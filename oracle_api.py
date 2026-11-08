"""Read-only clients for the external price oracles: Polymarket and Kalshi.

Both expose public, key-less REST reads. Every price is normalised into an
`OracleQuote` that is YES-denominated in [0, 1], the same convention SIG uses,
so strategy code can compare the three venues directly.

- Polymarket: Gamma API for discovery (events/markets/search), CLOB API for
  live order books. Prices live on outcome *token ids* (one per outcome).
- Kalshi: one REST API for markets, events and order books. Prices live on
  market *tickers*.

Kalshi's WebSocket feed requires an authenticated Kalshi account, so it is not
used; polling REST is enough for an oracle signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import aiohttp
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2"

REQUEST_TIMEOUT_SECONDS = 15
MAX_ATTEMPTS = 5
# Client-side throttle, well under each venue's public read limit, so bursts
# (e.g. market_mapper discovery) never trigger 429 in the first place.
POLYMARKET_MAX_REQUESTS_PER_SECOND = 20
KALSHI_MAX_REQUESTS_PER_SECOND = 4  # measured: 5/s sustained is fine, 8/s gets 429s
POLYMARKET_BATCH_SIZE = 100  # token ids per POST /books call
KALSHI_BATCH_SIZE = 50  # tickers per GET /markets?tickers= call
KALSHI_PAGE_LIMIT = 200

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

Source = Literal["polymarket", "kalshi"]


class OracleError(Exception):
    """Non-2xx response from an oracle API."""

    def __init__(
        self, source: Source, status: int, url: str, body: Any = None, retry_after: float | None = None
    ) -> None:
        super().__init__(f"{source} {status} for {url}")
        self.source = source
        self.status = status
        self.url = url
        self.body = body
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status in _RETRYABLE_STATUSES


class OracleTimeoutError(OracleError):
    """An oracle request timed out on every attempt. Not a TimeoutError on purpose -
    see exchange_api.RequestTimeoutError: main.py must not mistake it for its cycle timeout."""

    def __init__(self, source: Source, method: str, path: str, attempts: int, elapsed_s: float) -> None:
        super().__init__(source, 0, path)
        self.method = method
        self.path = path
        self.attempts = attempts
        self.elapsed_s = elapsed_s

    @property
    def retryable(self) -> bool:
        return True


@dataclass(frozen=True)
class OracleQuote:
    """Top of book for one YES contract, prices in [0, 1]. None = no liquidity on that side."""

    source: Source
    ref: str  # Polymarket outcome token id, or Kalshi market ticker
    best_bid: float | None
    best_ask: float | None
    bid_size: float | None
    ask_size: float | None
    last: float | None
    fetched_at: float  # unix seconds, local clock

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, OracleError):
        return exc.retryable
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


def _parse_retry_after(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


_backoff = wait_random_exponential(multiplier=0.5, max=20)


def _wait(retry_state: RetryCallState) -> float:
    delay = _backoff(retry_state)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, OracleError) and exc.status == 429:
        # Rate limited: never retry sooner than 1 s (or whatever the server asks for).
        delay = max(delay, exc.retry_after or 1.0)
    return delay


class _RateLimiter:
    """Spaces requests at least 1/rate seconds apart, across all concurrent callers."""

    def __init__(self, max_per_second: float) -> None:
        self._interval = 1.0 / max_per_second
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)


def _log_retry(retry_state: RetryCallState, method: str, path: str, started: float) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Oracle request failed, retrying",
        extra={
            "method": method,
            "path": path,
            "attempt": retry_state.attempt_number,
            "elapsed_s": round(time.monotonic() - started, 2),
            "sleep_s": round(retry_state.next_action.sleep, 2) if retry_state.next_action else None,
            "error": repr(exc),
        },
    )


class _JsonHttpClient:
    """Shared session handling + retrying JSON GET/POST. Use via `async with`."""

    source: Source
    max_requests_per_second: float

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._rate_limiter = _RateLimiter(self.max_requests_per_second)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        # Every call here is a read, so retrying is always safe.
        path = urlsplit(url).path  # logged without host or query string
        started = time.monotonic()
        attempts = 0
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                wait=_wait,
                stop=stop_after_attempt(MAX_ATTEMPTS),
                before_sleep=lambda state: _log_retry(state, method, path, started),
                reraise=True,
            ):
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    result = await self._send(method, url, params, json_body)
        except Exception as exc:
            elapsed_s = round(time.monotonic() - started, 2)
            if _is_retryable(exc):  # retries exhausted
                logger.error(
                    "Oracle request failed, giving up",
                    extra={
                        "source": self.source,
                        "method": method,
                        "path": path,
                        "attempts": attempts,
                        "elapsed_s": elapsed_s,
                        "error": repr(exc),
                    },
                )
            if isinstance(exc, asyncio.TimeoutError):
                raise OracleTimeoutError(self.source, method, path, attempts, elapsed_s) from exc
            raise
        return result

    async def _send(self, method: str, url: str, params: dict[str, Any] | None, json_body: Any) -> Any:
        if self._session is None:
            raise RuntimeError(f"{type(self).__name__} used outside 'async with'")

        clean = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in (params or {}).items() if v is not None}
        await self._rate_limiter.wait()
        started = time.monotonic()
        async with self._session.request(method, url, params=clean, json=json_body) as response:
            try:
                body = await response.json(content_type=None)
            except ValueError:
                body = None
            logger.debug(
                "Oracle call",
                extra={
                    "source": self.source,
                    "method": method,
                    "url": url,
                    "status": response.status,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                },
            )
            if response.status >= 400:
                raise OracleError(
                    self.source, response.status, url, body, _parse_retry_after(response.headers.get("Retry-After"))
                )
            return body


class PolymarketClient(_JsonHttpClient):
    source: Source = "polymarket"
    max_requests_per_second = POLYMARKET_MAX_REQUESTS_PER_SECOND

    # ------------------------------------------------------------ discovery (Gamma)

    async def search_events(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Full-text search; each event carries its nested `markets`."""
        body = await self._request(
            "GET", f"{POLYMARKET_GAMMA_URL}/public-search", params={"q": query, "limit_per_type": limit}
        )
        return body.get("events") or []

    async def get_event(self, slug: str) -> dict[str, Any] | None:
        events = await self._request("GET", f"{POLYMARKET_GAMMA_URL}/events", params={"slug": slug})
        return events[0] if events else None

    async def get_market(self, slug: str) -> dict[str, Any] | None:
        markets = await self._request("GET", f"{POLYMARKET_GAMMA_URL}/markets", params={"slug": slug})
        return markets[0] if markets else None

    async def list_markets(self, **params: Any) -> list[dict[str, Any]]:
        """Raw Gamma /markets query, e.g. `closed=False, limit=50, order="volume24hr"`."""
        return await self._request("GET", f"{POLYMARKET_GAMMA_URL}/markets", params=params)

    @staticmethod
    def outcome_tokens(market: dict[str, Any]) -> dict[str, str]:
        """Map outcome label -> CLOB token id, e.g. {"Yes": "5516...", "No": "3887..."}.

        Gamma returns `outcomes` and `clobTokenIds` as JSON-encoded strings.
        """
        outcomes = market.get("outcomes") or "[]"
        token_ids = market.get("clobTokenIds") or "[]"
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        return dict(zip(outcomes, token_ids))

    # ------------------------------------------------------------ prices (CLOB)

    async def get_book(self, token_id: str) -> dict[str, Any]:
        return await self._request("GET", f"{POLYMARKET_CLOB_URL}/book", params={"token_id": token_id})

    async def get_books(self, token_ids: list[str]) -> list[dict[str, Any]]:
        books: list[dict[str, Any]] = []
        for chunk in _chunks(token_ids, POLYMARKET_BATCH_SIZE):
            books.extend(
                await self._request("POST", f"{POLYMARKET_CLOB_URL}/books", json_body=[{"token_id": t} for t in chunk])
            )
        return books

    async def get_last_trade_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Last trade price per token id. Tokens that never traded are absent."""
        prices: dict[str, float] = {}
        for chunk in _chunks(token_ids, POLYMARKET_BATCH_SIZE):
            rows = await self._request(
                "POST", f"{POLYMARKET_CLOB_URL}/last-trades-prices", json_body=[{"token_id": t} for t in chunk]
            )
            for row in rows or []:
                price = _to_float(row.get("price"))
                if price is not None:
                    prices[row["token_id"]] = price
        return prices

    async def get_quote(self, token_id: str) -> OracleQuote:
        quotes = await self.get_quotes([token_id])
        if token_id not in quotes:
            raise OracleError("polymarket", 404, f"{POLYMARKET_CLOB_URL}/books (token {token_id})")
        return quotes[token_id]

    async def get_quotes(self, token_ids: list[str]) -> dict[str, OracleQuote]:
        """Quotes keyed by token id. Tokens without an order book (inactive) are absent."""
        books, last_prices = await asyncio.gather(self.get_books(token_ids), self.get_last_trade_prices(token_ids))
        return {book["asset_id"]: self.quote_from_book(book, last_prices.get(book["asset_id"])) for book in books}

    @staticmethod
    def quote_from_book(book: dict[str, Any], last: float | None = None) -> OracleQuote:
        # The book's own `last_trade_price` is not per-token on neg-risk markets
        # (all outcomes of an event report the same value), so `last` comes from
        # /last-trades-prices instead.
        # Do not rely on level ordering - pick best levels explicitly.
        bids = [(float(level["price"]), float(level["size"])) for level in book.get("bids") or []]
        asks = [(float(level["price"]), float(level["size"])) for level in book.get("asks") or []]
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        return OracleQuote(
            source="polymarket",
            ref=book["asset_id"],
            best_bid=best_bid[0] if best_bid else None,
            best_ask=best_ask[0] if best_ask else None,
            bid_size=best_bid[1] if best_bid else None,
            ask_size=best_ask[1] if best_ask else None,
            last=last,
            fetched_at=time.time(),
        )


class KalshiClient(_JsonHttpClient):
    source: Source = "kalshi"
    max_requests_per_second = KALSHI_MAX_REQUESTS_PER_SECOND

    # ------------------------------------------------------------ discovery

    async def list_series(self, category: str | None = None) -> list[dict[str, Any]]:
        """All series (e.g. category="Elections") in one call - cheap way to check tickers exist."""
        body = await self._request("GET", f"{KALSHI_URL}/series", params={"category": category})
        return body.get("series") or []

    async def list_events(
        self,
        *,
        series_ticker: str | None = None,
        status: str | None = "open",
        with_nested_markets: bool = True,
        limit: int = KALSHI_PAGE_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """One page: `{"events": [...], "cursor": "..."}`."""
        return await self._request(
            "GET",
            f"{KALSHI_URL}/events",
            params={
                "series_ticker": series_ticker,
                "status": status,
                "with_nested_markets": with_nested_markets,
                "limit": limit,
                "cursor": cursor,
            },
        )

    async def get_event(self, event_ticker: str, with_nested_markets: bool = True) -> dict[str, Any]:
        body = await self._request(
            "GET", f"{KALSHI_URL}/events/{event_ticker}", params={"with_nested_markets": with_nested_markets}
        )
        return body["event"]

    async def list_markets(
        self,
        *,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = "open",
        exclude_multivariate: bool = True,
        limit: int = KALSHI_PAGE_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """One page: `{"markets": [...], "cursor": "..."}`.

        Multivariate (parlay-style combo) markets are excluded by default - they
        flood the listing and are not useful as oracles.
        """
        return await self._request(
            "GET",
            f"{KALSHI_URL}/markets",
            params={
                "event_ticker": event_ticker,
                "series_ticker": series_ticker,
                "status": status,
                "mve_filter": "exclude" if exclude_multivariate else None,
                "limit": limit,
                "cursor": cursor,
            },
        )

    async def iter_markets(self, **filters: Any) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            page = await self.list_markets(**filters, cursor=cursor)
            for market in page.get("markets") or []:
                yield market
            cursor = page.get("cursor")
            if not cursor:
                return

    async def get_market(self, ticker: str) -> dict[str, Any]:
        body = await self._request("GET", f"{KALSHI_URL}/markets/{ticker}")
        return body["market"]

    async def get_markets(self, tickers: list[str]) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        for chunk in _chunks(tickers, KALSHI_BATCH_SIZE):
            body = await self._request(
                "GET", f"{KALSHI_URL}/markets", params={"tickers": ",".join(chunk), "limit": len(chunk)}
            )
            markets.extend(body.get("markets") or [])
        return markets

    async def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Raw book: `{"orderbook_fp": {"yes_dollars": [[price, size], ...], "no_dollars": [...]}}`.

        Kalshi books hold BIDS only; a YES ask is implied by a NO bid (ask = 1 - no_bid).
        """
        return await self._request("GET", f"{KALSHI_URL}/markets/{ticker}/orderbook")

    # ------------------------------------------------------------ prices

    async def get_quote(self, ticker: str) -> OracleQuote:
        return self.quote_from_market(await self.get_market(ticker))

    async def get_quotes(self, tickers: list[str]) -> dict[str, OracleQuote]:
        """Quotes keyed by ticker, one request per 50 tickers. Unknown tickers are absent."""
        return {market["ticker"]: self.quote_from_market(market) for market in await self.get_markets(tickers)}

    @staticmethod
    def quote_from_market(market: dict[str, Any]) -> OracleQuote:
        # Kalshi reports an empty side as 0 (bid) or 1 (ask); treat both as "no liquidity".
        bid = _to_float(market.get("yes_bid_dollars"))
        ask = _to_float(market.get("yes_ask_dollars"))
        if bid is not None and bid <= 0:
            bid = None
        if ask is not None and (ask <= 0 or ask >= 1):
            ask = None
        return OracleQuote(
            source="kalshi",
            ref=market["ticker"],
            best_bid=bid,
            best_ask=ask,
            bid_size=_to_float(market.get("yes_bid_size_fp")) if bid is not None else None,
            ask_size=_to_float(market.get("yes_ask_size_fp")) if ask is not None else None,
            last=_to_float(market.get("last_price_dollars")),
            fetched_at=time.time(),
        )


# ---------------------------------------------------------------------- smoke test


def _quote_fields(quote: OracleQuote) -> dict[str, Any]:
    return {
        "bid": quote.best_bid,
        "ask": quote.best_ask,
        "mid": round(quote.mid, 4) if quote.mid is not None else None,
        "last": quote.last,
    }


async def _smoke_test(polymarket_query: str, kalshi_series: str) -> int:
    from config import load_settings
    from logging_config import setup_logging

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    log = logging.getLogger("oracle_api.smoke_test")

    try:
        async with PolymarketClient() as poly:
            events = await poly.search_events(polymarket_query, limit=1)
            if not events:
                log.warning("Polymarket search returned nothing", extra={"query": polymarket_query})
            else:
                event = events[0]
                markets = [m for m in event.get("markets") or [] if not m.get("closed")][:5]
                yes_tokens = {PolymarketClient.outcome_tokens(m).get("Yes"): m for m in markets}
                yes_tokens.pop(None, None)
                quotes = await poly.get_quotes(list(yes_tokens))
                log.info("Polymarket event", extra={"title": event.get("title"), "slug": event.get("slug")})
                for token_id, market in yes_tokens.items():
                    if token_id in quotes:
                        log.info(market.get("question"), extra=_quote_fields(quotes[token_id]))

        async with KalshiClient() as kalshi:
            page = await kalshi.list_events(series_ticker=kalshi_series, limit=1)
            if not page.get("events"):
                log.warning("Kalshi series has no open events", extra={"series": kalshi_series})
            else:
                event = page["events"][0]
                tickers = [m["ticker"] for m in event.get("markets") or []][:5]
                quotes = await kalshi.get_quotes(tickers)
                log.info("Kalshi event", extra={"title": event.get("title"), "event_ticker": event.get("event_ticker")})
                for ticker in tickers:
                    if ticker in quotes:
                        log.info(ticker, extra=_quote_fields(quotes[ticker]))
    except OracleError as exc:
        log.error("Oracle call failed", extra={"source": exc.source, "status": exc.status, "url": exc.url})
        return 1
    return 0


if __name__ == "__main__":
    # Sanity check (read-only, no keys):
    #   python oracle_api.py                          defaults below
    #   python oracle_api.py "house 2026" KXFEDDECISION
    import sys

    args = sys.argv[1:]
    query = args[0] if len(args) > 0 else "senate 2026"
    series = args[1] if len(args) > 1 else "KXFEDDECISION"
    raise SystemExit(asyncio.run(_smoke_test(query, series)))
