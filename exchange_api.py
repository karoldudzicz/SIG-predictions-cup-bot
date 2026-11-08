"""Async REST client for the Super Market (SIG Predictions Cup) API.

Covers account and market discovery, prices and order books, portfolio and
tournament reads, the relationship graph and its constraints, order listing,
order placement (single and atomic multi-leg) and cancellation.

Error handling follows the API docs (branch on `error.code`, never `message`):
- 429, 5xx, network errors and 409 REQUEST_IN_FLIGHT are retried with exponential
  backoff + jitter, honouring `Retry-After` when the server sends it.
  This is safe for order placement too, because every placement carries an
  `idempotencyKey`: a retry with the same key never double-places.
- Other 4xx are never retried - a retry will not fix bad input, scope or membership.
- TERMS_NOT_ACKNOWLEDGED is raised as its own exception and logged as CRITICAL,
  because only a human can fix it (accept the terms on the website).

In DRY_RUN mode every mutating call (place / cancel) is logged and skipped,
and returns None. Reads always hit the real API.

Kill switch: once `halt()` is called (main.py does it the moment a stop is
requested), every placement is blocked and returns None - even mid-cycle.
Cancels keep working, so the shutdown cancel-all still runs.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from collections import Counter, deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

from config import (
    LADDER_BUCKET_RESERVE,
    SIG_GET_DEADLINE_S,
    SIG_GET_MAX_ATTEMPTS,
    SIG_GET_TIMEOUT_S,
    SIG_MAX_CONCURRENCY,
    SIG_READ_BURST,
    SIG_READ_PAUSE_MAX_S,
    SIG_READS_PER_MIN,
    SIG_POST_DEADLINE_S,
    SIG_POST_TIMEOUT_S,
    SIG_WRITE_BURST,
    SIG_WRITES_PER_MIN,
    Settings,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # orders and cancels; GETs use SIG_GET_MAX_ATTEMPTS
MARKETS_PAGE_LIMIT = 100  # API maximum
ORDERS_PAGE_LIMIT = 200  # API maximum
PRICES_BATCH_SIZE = 100  # API maximum ids per /exchanges/prices call
RELATIONSHIPS_PAGE_LIMIT = 200  # API maximum
MAX_MULTI_LEG_LEGS = 10
MAX_BATCH_ORDERS = 50  # API maximum for POST /orders/batch
FILLS_PAGE_LIMIT = 200  # API maximum
RATE_WINDOW_S = 60  # request counters cover the last minute
RATE_LOG_INTERVAL_S = 60  # "SIG request rate" INFO line at most this often
# Write-bucket ranks, best first. Quoter and ladder placements also keep
# LADDER_BUCKET_RESERVE tokens free for cancels and taker orders.
WRITE_RANK_CANCEL = 0
WRITE_RANK_ORDER = 1  # taker orders and everything else that is not a cancel
WRITE_RANK_QUOTER = 2
WRITE_RANK_LADDER = 3
# Read classes for the read bucket, best first: a read never takes a token while a read of
# a better class waits, so order books and background reads cannot starve essential ones.
READ_ESSENTIAL = 0  # positions, orders, pnl, tournament, account
READ_PRICES = 1  # bulk /exchanges/prices
READ_BOOK = 2  # /exchanges/{id}/orderbook
READ_OTHER = 3  # leaderboard, relationships, constraints, fills, research scripts

PRICE_TICK = 0.005
MIN_LIMIT_PRICE = 0.005
MAX_LIMIT_PRICE = 0.995

# The API holds an idempotency key's execution lease for 90 s, but waiting that long
# would not fit inside SIG_POST_DEADLINE_S. Re-asking sooner is cheap (another 409 or
# the replayed result); if the lease outlives the deadline, the order is journaled
# UNKNOWN and strategy.py blocks that market/side until it can be seen in positions.
REQUEST_IN_FLIGHT_WAIT_SECONDS = 10

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_AUTH_CODES = {
    "MISSING_API_KEY",
    "INVALID_API_KEY",
    "API_KEY_REVOKED",
    "API_KEY_EXPIRED",
    "INSUFFICIENT_SCOPES",
    "ACCOUNT_BANNED",
}

Side = Literal["yes", "no"]
Action = Literal["buy", "sell"]


class ApiError(Exception):
    """Error response from the API, carrying the stable `error.code`."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}
        self.retry_after = retry_after
        self.body = body  # raw response body, for endpoints with non-envelope errors

    @property
    def retryable(self) -> bool:
        return self.status in _RETRYABLE_STATUSES or self.code == "REQUEST_IN_FLIGHT"


class AuthError(ApiError):
    """The API key is missing, invalid, revoked, expired or lacks scopes."""


class TermsNotAcknowledgedError(ApiError):
    """The account has not accepted the current org terms - fix it on the website."""


class SelfCrossError(ApiError):
    """Not sent: the order could trade against one of our own resting orders.

    Raised locally before any request (status 0, not retryable), so callers that already
    handle ApiError record it as a rejection.
    """

    def __init__(self, order: dict[str, Any], resting_order_id: str) -> None:
        super().__init__(0, "SELF_CROSS_BLOCKED", f"would cross own resting order {resting_order_id}")
        self.order = order
        self.resting_order_id = resting_order_id


class ReadDeferredError(ApiError):
    """Not sent: no read token within the caller's short token wait (see get_orderbook's
    `token_wait_s`). Not retryable - the caller defers the read to its next cycle."""

    def __init__(self, path: str, waited_s: float) -> None:
        super().__init__(0, "READ_DEFERRED", f"GET {path} not sent: no read token within {waited_s:.1f} s")
        self.path = path

    @property
    def retryable(self) -> bool:
        return False


class ReadPausedError(ApiError):
    """Not sent: reads are paused after a 429 and this read is an order book or non-essential.

    Status 429 with `retry_after` = what is left of the pause, but NOT retryable here: the
    caller skips it this cycle (or, like edge_scan.SigReader, waits `retry_after` itself).
    """

    def __init__(self, path: str, remaining_s: float) -> None:
        super().__init__(429, "READ_PAUSED", f"GET {path} not sent: reads paused for {remaining_s:.1f} s", retry_after=remaining_s)
        self.path = path

    @property
    def retryable(self) -> bool:
        return False


class RequestTimeoutError(ApiError):
    """A SIG request timed out on every attempt (or hit its deadline).

    Deliberately NOT a TimeoutError: on Python 3.11+ asyncio.TimeoutError is TimeoutError,
    and main.py must be able to tell a failed request from its own cycle timeout.
    `retryable` is True: for a GET the next cycle can simply retry; for an order POST
    the outcome is unknown (strategy.py journals it UNKNOWN).
    """

    def __init__(self, method: str, path: str, attempts: int, elapsed_s: float) -> None:
        super().__init__(0, "REQUEST_TIMEOUT", f"{method} {path} timed out after {attempts} attempt(s), {elapsed_s} s")
        self.method = method
        self.path = path
        self.attempts = attempts
        self.elapsed_s = elapsed_s

    @property
    def retryable(self) -> bool:
        return True


class _WriteBucket:
    """Token bucket for SIG writes (and, with READ_* ranks, reads): `per_minute` refill, at
    most `burst` stored.

    Callers are ranked (WRITE_RANK_*): cancels, taker orders, quoter placements, ladder
    placements. A caller never takes a token while a better-ranked caller waits; quoter and
    ladder placements also keep `reserve` tokens in the bucket. One asyncio thread, no
    await between check and take, so no lock is needed.
    """

    def __init__(self, per_minute: float, burst: int) -> None:
        self._rate = per_minute / 60.0
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._waiting: Counter[int] = Counter()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now

    async def acquire(self, rank: int = WRITE_RANK_ORDER, reserve: int = 0) -> None:
        """Take one token. A caller waits while any caller of a better (lower) rank waits,
        and takes only when at least `reserve` tokens are left after its take."""
        needed = 1.0 + reserve
        self._waiting[rank] += 1
        try:
            while True:
                self._refill()
                ahead = any(count for r, count in self._waiting.items() if r < rank)
                if not ahead and self._tokens >= needed:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep(max((needed - self._tokens) / self._rate, 0.01))
        finally:
            self._waiting[rank] -= 1

    def available(self, rank: int) -> float:
        """Tokens a caller of `rank` could take right now without waiting (0 while a caller
        of a better rank waits). Takes nothing."""
        self._refill()
        if any(count for r, count in self._waiting.items() if r < rank):
            return 0.0
        return self._tokens

    @property
    def per_second(self) -> float:
        return self._rate

    def release(self) -> None:
        """Give back a token that was taken but never used for a request."""
        self._refill()
        self._tokens = min(self._capacity, self._tokens + 1.0)


def _log_rate_headers(method: str, path: str, response: aiohttp.ClientResponse) -> None:
    """Log rate-limit headers when the response carries any (SIG does not document them):
    DEBUG normally, INFO on a 429. Only response headers are inspected - never our own."""
    headers = {
        name: value
        for name, value in response.headers.items()
        if name.lower().startswith(("x-ratelimit", "ratelimit")) or name.lower() == "retry-after"
    }
    if headers:
        level = logging.INFO if response.status == 429 else logging.DEBUG
        logger.log(level, "Rate-limit headers", extra={"method": method, "path": path, "status": response.status, "headers": headers})


def new_idempotency_key(prefix: str = "order") -> str:
    """Fresh key for one logical placement. Persist it before sending if you may need to resume."""
    return f"{prefix}-{uuid.uuid4().hex}"


def _validate_limit_price(price: float) -> float:
    """Return the price snapped to the 0.005 grid, or raise if it is off-tick / out of range."""
    ticks = round(price / PRICE_TICK)
    if abs(ticks * PRICE_TICK - price) > 1e-9:
        raise ValueError(f"price {price} is not on the {PRICE_TICK} tick")
    snapped = round(ticks * PRICE_TICK, 3)
    if not MIN_LIMIT_PRICE <= snapped <= MAX_LIMIT_PRICE:
        raise ValueError(f"price {price} is outside [{MIN_LIMIT_PRICE}, {MAX_LIMIT_PRICE}]")
    return snapped


def yes_terms(side: str, action: str, price: float | None) -> tuple[Literal["bid", "ask"], float]:
    """Where an order sits in the YES-normalised book: (book side, YES price).

    buy YES @p and sell NO @p are YES bids at p and 1 - p; buy NO @p and sell YES @p are
    YES asks at 1 - p and p (the engine's own equivalence: sell yes q@p = buy no q@(1-p)).
    A market order (price None) is a bid at 1 or an ask at 0 - it crosses anything.
    """
    buys_yes = (side == "yes") == (action == "buy")
    if price is None:
        return ("bid", 1.0) if buys_yes else ("ask", 0.0)
    return ("bid" if buys_yes else "ask"), round(price if side == "yes" else 1.0 - price, 3)


def crosses(new: tuple[str, float], resting: tuple[str, float]) -> bool:
    """True if `new` (book side, YES price) could trade against `resting`."""
    if new[0] == resting[0]:
        return False
    bid, ask = (new[1], resting[1]) if new[0] == "bid" else (resting[1], new[1])
    return bid >= ask - 1e-9


def book_without_own(orderbook: dict[str, Any], own: Iterable[tuple[str, float, float]]) -> dict[str, Any]:
    """The YES-normalised book with our own resting quantity taken out.

    `own`: (book side, YES price, remaining quantity) of our resting orders on this
    exchange (ExchangeClient.own_orders). Levels left with nothing are dropped, so the
    first bid / ask of the result is the best OTHER bid / ask.
    """
    mine: dict[tuple[str, float], float] = {}
    for side, price, quantity in own:
        key = (side, round(price, 3))
        mine[key] = mine.get(key, 0.0) + abs(quantity)
    result = dict(orderbook)
    for book_side, key_side in (("bids", "bid"), ("asks", "ask")):
        levels = []
        for level in orderbook.get(book_side) or []:
            left = float(level["quantity"]) - mine.get((key_side, round(float(level["price"]), 3)), 0.0)
            if left > 1e-9:
                levels.append({**level, "quantity": left})
        result[book_side] = levels
    return result


@dataclass(frozen=True)
class OrderRequest:
    """One order (or one leg of a multi-leg order). `price=None` means a market order."""

    exchange_id: str
    side: Side
    action: Action
    quantity: int
    price: float | None = None
    tournament_id: str | None = None
    expiration_date: str | None = None  # ISO 8601, limit orders only

    def __post_init__(self) -> None:
        # Validate locally so a bad order never burns a rate-limited request.
        if not str(self.exchange_id).isdigit():
            raise ValueError(f"exchange_id must be a numeric string, got {self.exchange_id!r}")
        if self.side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got {self.side!r}")
        if self.action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {self.action!r}")
        if not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError(f"quantity must be a positive int, got {self.quantity!r}")
        if self.price is not None:
            object.__setattr__(self, "price", _validate_limit_price(self.price))
        elif self.expiration_date is not None:
            raise ValueError("expiration_date is not supported for market orders")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exchangeId": str(self.exchange_id),
            "side": self.side,
            "action": self.action,
            "quantity": self.quantity,
        }
        if self.price is not None:
            payload["price"] = self.price
        if self.tournament_id is not None:
            payload["tournamentId"] = self.tournament_id
        if self.expiration_date is not None:
            payload["expirationDate"] = self.expiration_date
        return payload


def _parse_retry_after(raw: str | None) -> float | None:
    # Only the delta-seconds form is handled; an HTTP-date falls back to normal backoff.
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ApiError):
        return exc.retryable
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


_backoff = wait_random_exponential(multiplier=0.25, max=30)
GET_TIMEOUT_RETRY_WAIT_MAX_S = 0.5  # a timed-out GET is retried almost at once


def _wait(retry_state: RetryCallState) -> float:
    delay = _backoff(retry_state)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, ApiError) and exc.retry_after is not None:
        delay = max(delay, exc.retry_after)
    return delay


def _get_wait(retry_state: RetryCallState) -> float:
    """GETs: retry a timeout after a short jitter; a 429 at once (the shared read pause in
    ExchangeClient._read_gate does the waiting - no read sleeps on its own); other errors
    (5xx) keep the normal backoff. Never sleep past the logical GET's deadline."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, ApiError) and exc.status == 429:
        return 0.0
    if isinstance(exc, asyncio.TimeoutError):
        delay = random.uniform(0.05, GET_TIMEOUT_RETRY_WAIT_MAX_S)
    else:
        delay = _wait(retry_state)
    remaining = SIG_GET_DEADLINE_S - (time.monotonic() - retry_state.start_time)
    return max(0.0, min(delay, remaining))


def _post_deadline_wait(retry_state: RetryCallState) -> float:
    """Order POSTs: normal backoff (incl. Retry-After / REQUEST_IN_FLIGHT), never past the deadline."""
    remaining = SIG_POST_DEADLINE_S - (time.monotonic() - retry_state.start_time)
    return max(0.0, min(_wait(retry_state), remaining))


def _log_retry(retry_state: RetryCallState, method: str, path: str, started: float) -> None:
    # `path` never carries the query string or credentials (params and headers are separate).
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Request failed, retrying",
        extra={
            "method": method,
            "path": path,
            "attempt": retry_state.attempt_number,
            "elapsed_s": round(time.monotonic() - started, 2),
            "sleep_s": round(retry_state.next_action.sleep, 2) if retry_state.next_action else None,
            "error": repr(exc),
        },
    )


def _clean_params(params: dict[str, Any]) -> dict[str, str]:
    # aiohttp rejects None and bool query values; the API expects "true"/"false".
    cleaned: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = "true" if value else "false"
        else:
            cleaned[key] = str(value)
    return cleaned


def _log_placed(result: dict[str, Any], idempotency_key: str, leg: int | None = None) -> None:
    # Log the CANONICAL order from the response - a SELL may have been rewritten
    # into its complement BUY, so the request payload is not the truth.
    fields = {
        "idempotency_key": idempotency_key,
        "order_id": result.get("orderId"),
        "exchange_id": result.get("exchangeId"),
        "action": result.get("action"),
        "side": result.get("side"),
        "price": result.get("price"),
        "quantity": result.get("quantity"),
        "traded": result.get("quantityTraded"),
        "remaining": result.get("remainingQuantity"),
        "open": result.get("open"),
        "terminal_reason": result.get("terminalReasonCode"),
    }
    if leg is not None:
        fields["leg"] = leg
    if result.get("terminalReasonCode") == "SelfTradePrevented":
        logger.warning("Order remainder cancelled by self-trade prevention", extra=fields)
    else:
        logger.info("Order placed", extra=fields)


class ExchangeClient:
    """Use as `async with ExchangeClient(settings) as client: ...`."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.sig_base_url
        self._api_key = settings.require_api_key()
        self._dry_run = settings.dry_run
        self._halted = False
        self._session: aiohttp.ClientSession | None = None
        # GETs only: cancels and order placement bypass it (the kill switch never queues).
        self._get_semaphore = asyncio.Semaphore(SIG_MAX_CONCURRENCY)
        # Every read attempt takes a token (see config.SIG_READS_PER_MIN, READ_* classes).
        self._read_bucket = _WriteBucket(SIG_READS_PER_MIN, SIG_READ_BURST)
        self._read_pause_until = 0.0  # monotonic; set by a 429 on any read (_start_read_pause)
        # Every write (orders, batches, cancels - each retry too) takes a token; see config.py.
        self._write_bucket = _WriteBucket(SIG_WRITES_PER_MIN, SIG_WRITE_BURST)
        # Per-cycle GET counters (strategy.py resets them at cycle start): cheap ints, no logging.
        self._get_retries = 0
        self._slowest_get_s = 0.0
        # Rolling 60 s request counters (every HTTP attempt, retries and timeouts included),
        # logged once a minute and reported in "Cycle done". SIG's per-account budget is
        # 100 reads + 30 writes per minute on a standard account.
        self._read_times: deque[float] = deque()
        self._write_times: deque[float] = deque()
        self._rate_limited_times: deque[float] = deque()
        self._rate_logged_at = time.monotonic()
        # Our own resting orders: exchange_id -> {order_id: (book side, YES price, remaining)}. Rebuilt
        # from every open-order read (note_open_orders), extended by placement responses,
        # pruned by clean cancels. Used only to refuse orders that would self-cross; a stale
        # entry can only block, never let a crossing order through.
        self._own_orders: dict[str, dict[str, tuple[str, float, float]]] = {}

    def _count_request(self, method: str) -> None:
        now = time.monotonic()
        (self._read_times if method in ("GET", "HEAD") else self._write_times).append(now)
        if now - self._rate_logged_at >= RATE_LOG_INTERVAL_S:
            self._rate_logged_at = now
            logger.info("SIG request rate", extra=self.rate_stats())

    def rate_stats(self) -> dict[str, int]:
        """Requests in the last 60 s: reads, writes and 429 responses."""
        now = time.monotonic()
        for times in (self._read_times, self._write_times, self._rate_limited_times):
            while times and now - times[0] > RATE_WINDOW_S:
                times.popleft()
        return {
            "reads_last_min": len(self._read_times),
            "writes_last_min": len(self._write_times),
            "rate_limited_last_min": len(self._rate_limited_times),
        }

    async def _acquire_write(self, method: str, path: str, write_rank: int = WRITE_RANK_ORDER) -> None:
        """Rate-limit one write. Cancels go first; once the kill switch is engaged they skip
        the bucket entirely, so shutdown's cancel-all never waits for a token. Quoter and
        ladder placements go after taker orders and keep LADDER_BUCKET_RESERVE tokens free."""
        is_cancel = method == "DELETE" or path == "/orders/cancel-all"
        if is_cancel and self._halted:
            return
        rank = WRITE_RANK_CANCEL if is_cancel else write_rank
        await self._write_bucket.acquire(rank, LADDER_BUCKET_RESERVE if rank >= WRITE_RANK_QUOTER else 0)

    async def acquire_order_token(self) -> bool:
        """Take the write token for one order up front (then `place_order(..., prepaid=True)`),
        so the caller can check its quote age AFTER any wait for the bucket. Returns False,
        taking nothing, in DRY_RUN or once halted. Give an unused token back with
        `refund_order_token()`."""
        if self._dry_run or self._halted:
            return False
        await self._write_bucket.acquire()
        return True

    def refund_order_token(self) -> None:
        self._write_bucket.release()

    # ------------------------------------------------------------------ self-cross guard

    def note_open_orders(self, orders: list[dict[str, Any]]) -> None:
        """Replace the own-order registry with an open-order read (GET /orders?status=open).

        `priceLimit` is in the order's own side terms, like the price we send.
        """
        registry: dict[str, dict[str, tuple[str, float, float]]] = {}
        for order in orders:
            if order.get("open") is False or not order.get("exchangeId"):
                continue
            side, price = yes_terms(order.get("side"), order.get("action"), order.get("priceLimit"))
            registry.setdefault(str(order["exchangeId"]), {})[str(order.get("id"))] = (side, price, abs(float(order.get("quantity") or 0)))
        self._own_orders = registry

    def own_orders(self, exchange_id: str) -> list[tuple[str, float, float]]:
        """(book side, YES price, remaining) of our resting orders on one exchange."""
        return list(self._own_orders.get(str(exchange_id), {}).values())

    def _note_placed(self, result: dict[str, Any] | None, request: OrderRequest) -> None:
        """Add a placed order that is still resting (canonical side / action / price)."""
        if not result or not result.get("open") or not result.get("remainingQuantity"):
            return
        exchange_id = str(result.get("exchangeId") or request.exchange_id)
        side, action = result.get("side") or request.side, result.get("action") or request.action
        book_side, yes_price = yes_terms(side, action, result.get("price", request.price))
        self._own_orders.setdefault(exchange_id, {})[str(result.get("orderId"))] = (book_side, yes_price, abs(float(result["remainingQuantity"])))

    def _forget_orders(self, order_id: str | None = None, exchange_id: str | None = None) -> None:
        if order_id is not None:
            for resting in self._own_orders.values():
                resting.pop(str(order_id), None)
        elif exchange_id is not None:
            self._own_orders.pop(str(exchange_id), None)
        else:
            self._own_orders.clear()

    def self_cross(self, order: OrderRequest, extra: dict[str, tuple[Any, ...]] | None = None) -> str | None:
        """Id of an own resting order (or of an `extra` one, e.g. an earlier leg of the same
        batch) that `order` could trade against, else None."""
        new = yes_terms(order.side, order.action, order.price)
        resting = {**self._own_orders.get(str(order.exchange_id), {}), **(extra or {})}
        for order_id, terms in resting.items():
            if crosses(new, terms):
                return order_id
        return None

    def _log_self_cross(self, order: OrderRequest, resting_order_id: str, idempotency_key: str) -> None:
        logger.warning(
            "self_cross_blocked",
            extra={"idempotency_key": idempotency_key, "payload": order.to_payload(), "resting_order_id": resting_order_id, "dry_run": self._dry_run},
        )

    # ------------------------------------------------------------------ read pause after a 429

    def read_pause_remaining(self) -> float:
        return max(0.0, self._read_pause_until - time.monotonic())

    def _start_read_pause(self, retry_after: float | None, path: str) -> None:
        """One shared pause for every read; one INFO line when a pause starts."""
        seconds = min(retry_after if retry_after is not None else SIG_READ_PAUSE_MAX_S, SIG_READ_PAUSE_MAX_S)
        now = time.monotonic()
        if now + seconds <= self._read_pause_until:
            return  # already paused at least this long
        if now >= self._read_pause_until:
            logger.info("SIG read pause", extra={"pause_s": round(seconds, 1), "reason": f"429 on GET {path}"})
        self._read_pause_until = now + seconds

    async def _read_gate(self, read_class: int, path: str) -> float:
        """During a read pause: order books and non-essential reads raise ReadPausedError
        (not sent), essential reads and bulk prices wait for the end. Returns seconds waited."""
        remaining = self.read_pause_remaining()
        if remaining <= 0:
            return 0.0
        if read_class >= READ_BOOK:
            raise ReadPausedError(path, remaining)
        await asyncio.sleep(remaining)
        return remaining

    def reset_get_stats(self) -> None:
        self._get_retries = 0
        self._slowest_get_s = 0.0

    def get_stats(self) -> dict[str, Any]:
        """`get_retries`: retried GET attempts; `slowest_get_s`: slowest successful logical GET."""
        return {"get_retries": self._get_retries, "slowest_get_s": round(self._slowest_get_s, 2), **self.rate_stats()}

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def halted(self) -> bool:
        return self._halted

    def halt(self, reason: str) -> None:
        """Kill switch: block all further order placement. Irreversible for this client."""
        if not self._halted:
            self._halted = True
            logger.warning("Order placement halted", extra={"reason": reason})

    async def __aenter__(self) -> ExchangeClient:
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=SIG_POST_TIMEOUT_S),
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------ core

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        bounded_post: bool = False,
        prepaid: bool = False,
        write_rank: int = WRITE_RANK_ORDER,
        read_class: int = READ_OTHER,
        token_wait_s: float | None = None,
    ) -> Any:
        """Send with retry on retryable errors.

        Only call this for requests that are safe to repeat: reads, cancels,
        and placements that carry an idempotencyKey.

        GET: SIG_GET_TIMEOUT_S per attempt, up to SIG_GET_MAX_ATTEMPTS, a timeout is
        retried after <= 0.5 s, the whole logical GET is bounded by SIG_GET_DEADLINE_S,
        and at most SIG_MAX_CONCURRENCY GETs are in flight.
        Order placement (bounded_post=True): SIG_POST_TIMEOUT_S per attempt, MAX_ATTEMPTS,
        normal backoff, but the whole logical POST - retries and waits included - ends
        within SIG_POST_DEADLINE_S. Same payload and idempotencyKey on every attempt.
        Everything else (cancels, cancel-all): SIG_POST_TIMEOUT_S, MAX_ATTEMPTS, normal
        backoff, no deadline, no concurrency cap - unchanged.
        Writes take a bucket token per attempt; `prepaid`: the caller already holds the
        token for the first attempt. `write_rank`: see WRITE_RANK_* and _acquire_write.
        Reads take a read-bucket token per attempt, by `read_class` (READ_*), waiting at
        most until the GET deadline. A 429 on a read starts the shared read pause
        (_read_gate): the retry goes through the gate instead of sleeping on its own; time
        spent waiting out a pause does not count against the GET deadline.
        """
        is_get = method == "GET"
        started = time.monotonic()
        if is_get:
            wait, stop = _get_wait, stop_after_attempt(SIG_GET_MAX_ATTEMPTS) | stop_after_delay(SIG_GET_DEADLINE_S)
        elif bounded_post:
            wait, stop = _post_deadline_wait, stop_after_attempt(MAX_ATTEMPTS) | stop_after_delay(SIG_POST_DEADLINE_S)
        else:
            wait, stop = _wait, stop_after_attempt(MAX_ATTEMPTS)

        attempts = 0
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                wait=wait,
                stop=stop,
                before_sleep=lambda state: _log_retry(state, method, path, started),
                reraise=True,
            ):
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    if not is_get and not (prepaid and attempts == 1):
                        await self._acquire_write(method, path, write_rank)
                    if is_get:
                        started += await self._read_gate(read_class, path)
                        remaining = SIG_GET_DEADLINE_S - (time.monotonic() - started)
                        # A token wait past the deadline ends as a timeout (-> RequestTimeoutError);
                        # past the caller's `token_wait_s` as ReadDeferredError (not sent, no retry).
                        token_limit = remaining if token_wait_s is None else min(remaining, token_wait_s)
                        try:
                            await asyncio.wait_for(self._read_bucket.acquire(read_class), timeout=max(0.01, token_limit))
                        except TimeoutError:
                            if token_wait_s is not None and token_wait_s < remaining:
                                raise ReadDeferredError(path, token_wait_s) from None
                            raise
                        remaining = SIG_GET_DEADLINE_S - (time.monotonic() - started)
                        timeout_s = max(0.1, min(SIG_GET_TIMEOUT_S, remaining))
                        try:
                            async with self._get_semaphore:
                                result = await self._send(method, path, params, json_body, timeout_s)
                        except ApiError as exc:
                            if exc.status == 429:
                                self._start_read_pause(exc.retry_after, path)
                            raise
                    elif bounded_post:
                        remaining = SIG_POST_DEADLINE_S - (time.monotonic() - started)
                        timeout_s = max(0.1, min(SIG_POST_TIMEOUT_S, remaining))
                        result = await self._send(method, path, params, json_body, timeout_s)
                    else:
                        result = await self._send(method, path, params, json_body, SIG_POST_TIMEOUT_S)
        except Exception as exc:
            elapsed_s = round(time.monotonic() - started, 2)
            if is_get:
                self._get_retries += max(0, attempts - 1)
            if _is_retryable(exc):  # retries (or the GET deadline) exhausted
                logger.error(
                    "Request failed, giving up",
                    extra={"method": method, "path": path, "attempts": attempts, "elapsed_s": elapsed_s, "error": repr(exc)},
                )
            if isinstance(exc, asyncio.TimeoutError):
                raise RequestTimeoutError(method, path, attempts, elapsed_s) from exc
            raise
        elapsed = time.monotonic() - started
        if is_get:
            self._get_retries += attempts - 1
            self._slowest_get_s = max(self._slowest_get_s, elapsed)
        if attempts > 1:
            # Measures how often the slow mode hits; DEBUG only, no INFO spam.
            logger.debug(
                "Request succeeded after retry",
                extra={"method": method, "path": path, "attempts": attempts, "elapsed_s": round(elapsed, 2)},
            )
        return result

    async def _send(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        json_body: Any,
        timeout_s: float,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("ExchangeClient used outside 'async with'")

        url = f"{self._base_url}{path}"
        self._count_request(method)
        started = time.monotonic()
        # asyncio.timeout bounds the whole attempt (connect + headers + body), whatever
        # the session's own timeout settings are.
        async with asyncio.timeout(timeout_s), self._session.request(
            method,
            url,
            params=_clean_params(params or {}),
            json=json_body,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as response:
            elapsed_ms = round((time.monotonic() - started) * 1000)
            try:
                body = await response.json(content_type=None)
            except ValueError:
                body = None  # non-JSON body, e.g. an HTML error page from a proxy

            logger.debug(
                "API call",
                extra={"method": method, "path": path, "status": response.status, "elapsed_ms": elapsed_ms},
            )
            if response.status == 429:
                self._rate_limited_times.append(time.monotonic())
            _log_rate_headers(method, path, response)

            if response.status < 400:
                return body
            raise self._build_error(response, body)

    @staticmethod
    def _build_error(response: aiohttp.ClientResponse, body: Any) -> ApiError:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        code = error.get("code") or f"HTTP_{response.status}"
        message = error.get("message") or response.reason or "Unknown error"
        details = error.get("details") or {}
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        if code == "REQUEST_IN_FLIGHT" and retry_after is None:
            retry_after = REQUEST_IN_FLIGHT_WAIT_SECONDS

        if code == "TERMS_NOT_ACKNOWLEDGED":
            logger.critical(
                "Account has not accepted the current terms - accept them on thesuper.market, "
                "the API cannot do it",
                extra={"missing_documents": details.get("missingDocuments")},
            )
            return TermsNotAcknowledgedError(response.status, code, message, details, retry_after, body)
        if code in _AUTH_CODES:
            return AuthError(response.status, code, message, details, retry_after, body)
        return ApiError(response.status, code, message, details, retry_after, body)

    # ------------------------------------------------------------- endpoints: reads

    async def get_account(self) -> dict[str, Any]:
        """Profile and balance. For an org-bound key, balance is the default tournament's."""
        return await self._request("GET", "/account", read_class=READ_ESSENTIAL)

    async def list_markets(
        self,
        *,
        status: str | None = None,
        tournament_id: str | None = None,
        search: str | None = None,
        category: str | None = None,
        is_composite: bool | None = None,
        is_multi_outcome: bool | None = None,
        sort: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """One page of markets: `{"data": [...], "pagination": {...}}`."""
        return await self._request(
            "GET",
            "/markets",
            params={
                "status": status,
                "tournamentId": tournament_id,
                "search": search,
                "category": category,
                "is_composite": is_composite,
                "is_multi_outcome": is_multi_outcome,
                "sort": sort,
                "limit": limit,
                "cursor": cursor,
            },
        )

    async def iter_markets(self, **filters: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield every market matching the filters, following pagination cursors."""
        cursor: str | None = None
        while True:
            page = await self.list_markets(**filters, limit=MARKETS_PAGE_LIMIT, cursor=cursor)
            for market in page["data"]:
                yield market
            pagination = page["pagination"]
            cursor = pagination.get("nextCursor")
            if not pagination.get("hasMore") or not cursor:
                return

    async def list_orders(
        self,
        *,
        status: Literal["open", "closed", "expired", "all"] = "open",
        exchange_id: str | None = None,
        market_id: str | None = None,
        tournament_id: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """One page of the caller's orders (default: open only)."""
        return await self._request(
            "GET",
            "/orders",
            params={
                "status": status,
                "exchangeId": exchange_id,
                "marketId": market_id,
                "tournamentId": tournament_id,
                "limit": limit,
                "cursor": cursor,
            },
            read_class=READ_ESSENTIAL,
        )

    async def iter_orders(self, **filters: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield every order matching the filters, following pagination cursors."""
        cursor: str | None = None
        while True:
            page = await self.list_orders(**filters, limit=ORDERS_PAGE_LIMIT, cursor=cursor)
            for order in page["data"]:
                yield order
            pagination = page["pagination"]
            cursor = pagination.get("nextCursor")
            if not pagination.get("hasMore") or not cursor:
                return

    async def get_order(self, order_id: int | str) -> dict[str, Any]:
        return await self._request("GET", f"/orders/{order_id}", read_class=READ_ESSENTIAL)

    async def get_prices(self, exchange_ids: list[str], tournament_id: str | None = None) -> dict[str, dict[str, Any]]:
        """Bulk top-of-book snapshot keyed by exchange id: latestPrice, bestBid, bestAsk, spread.

        ceil(N / 100) requests. Prices are YES-normalised; `tournament_id` scopes them to that
        tournament's book. Ids outside the context come back in missingIds (one WARNING with
        the count). Each row gets `receivedAt` (unix time its response arrived) - the age of
        a bulk price is measured from there.
        """
        prices: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for start in range(0, len(exchange_ids), PRICES_BATCH_SIZE):
            chunk = exchange_ids[start : start + PRICES_BATCH_SIZE]
            body = await self._request(
                "GET", "/exchanges/prices", params={"ids": ",".join(chunk), "tournamentId": tournament_id}, read_class=READ_PRICES
            )
            received_at = time.time()
            for row in body["data"]:
                prices[str(row["exchangeId"])] = {**row, "receivedAt": received_at}
            missing.extend(str(eid) for eid in body.get("missingIds") or [])
        if missing:
            logger.warning("Exchanges missing from price snapshot", extra={"missing": len(missing), "exchange_ids": missing[:20]})
        return prices

    def read_budget(self, read_class: int, window_s: float, reserve: int = 0) -> int:
        """How many reads of `read_class` the bucket can serve now plus what it refills within
        `window_s`, minus `reserve` tokens kept for other reads. 0 during a read pause."""
        if self.read_pause_remaining() > 0:
            return 0
        bucket = self._read_bucket
        return max(0, math.floor(bucket.available(read_class) + bucket.per_second * window_s + 1e-9) - reserve)

    async def get_orderbook(self, exchange_id: str, depth: int = 20, token_wait_s: float | None = None) -> dict[str, Any]:
        """YES-normalised book: bids descending, asks ascending, each level {price, quantity};
        `receivedAt` = unix time the response arrived. `token_wait_s`: wait at most this long
        for a read token, then ReadDeferredError (nothing sent) instead of waiting up to the
        GET deadline."""
        body = await self._request(
            "GET", f"/exchanges/{exchange_id}/orderbook", params={"depth": depth}, read_class=READ_BOOK, token_wait_s=token_wait_s
        )
        return {**body, "receivedAt": time.time()}

    async def get_positions(self) -> list[dict[str, Any]]:
        """Open positions in the default tournament. quantity > 0 = YES shares, < 0 = NO shares."""
        body = await self._request("GET", "/portfolio/positions", read_class=READ_ESSENTIAL)
        return body["positions"]

    async def get_pnl(self, period: str = "all") -> dict[str, Any]:
        """Account value, holdings value, unrealized and period P&L."""
        return await self._request("GET", "/portfolio/pnl", params={"period": period}, read_class=READ_ESSENTIAL)

    async def get_tournament(self, slug: str) -> dict[str, Any]:
        """Tournament metadata incl. `id` (UUID), start/end dates and `myBalance`."""
        return await self._request("GET", f"/tournaments/{slug}", read_class=READ_ESSENTIAL)

    async def get_tournament_fills(
        self, slug: str, cursor: str | None = None, limit: int = FILLS_PAGE_LIMIT, exchange_id: str | None = None
    ) -> dict[str, Any]:
        """One page of the caller's fills in this tournament, newest first (optionally one exchange).

        `{"data": [Fill...], "pagination": {...}}`; Fill: id, orderId, exchangeId, price,
        quantity (outcome-signed: < 0 = NO), side, filledAt.
        """
        return await self._request(
            "GET",
            f"/tournaments/{slug}/portfolio/fills",
            params={"limit": limit, "cursor": cursor, "exchangeId": exchange_id},
        )

    async def get_leaderboard(self, slug: str, period: str = "all", limit: int = 50) -> dict[str, Any]:
        """Tournament leaderboard incl. `myRank` (None until we trade) and `total` members ranked."""
        return await self._request("GET", f"/tournaments/{slug}/leaderboard", params={"period": period, "limit": limit})

    # ------------------------------------------------------------- endpoints: relationships

    async def iter_relationships(
        self, *, market_id: str | None = None, exchange_id: str | None = None, rel_type: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Every active SIG relationship (the engine's cross-market consistency graph)."""
        cursor: str | None = None
        while True:
            page = await self._request(
                "GET",
                "/relationships",
                params={
                    "marketId": market_id,
                    "exchangeId": exchange_id,
                    "type": rel_type,
                    "limit": RELATIONSHIPS_PAGE_LIMIT,
                    "cursor": cursor,
                },
            )
            for relationship in page["data"]:
                yield relationship
            pagination = page["pagination"]
            cursor = pagination.get("nextCursor")
            if not pagination.get("hasMore") or not cursor:
                return

    async def get_relationship_graph(self, market_id: str, depth: int = 2) -> dict[str, Any]:
        """`{"relationships": [...], "nodes": [...]}` reachable from one market."""
        return await self._request("GET", "/relationships/graph", params={"marketId": market_id, "depth": depth})

    async def get_constraints(
        self,
        *,
        violations_only: bool = True,
        min_violation: float | None = None,
        market_id: str | None = None,
        relationship_id: str | None = None,
    ) -> dict[str, Any]:
        """Engine-evaluated relationship constraints, incl. `suggestedCorrectiveTrades`.

        Returns `{"data": [...], "violationsCount": n, "computedAt": ...}`.
        `min_violation` defaults to 0.01 on the server when omitted.
        """
        return await self._request(
            "GET",
            "/relationships/constraints",
            params={
                "violationsOnly": violations_only,
                "minViolation": min_violation,
                "marketId": market_id,
                "relationshipId": relationship_id,
            },
        )

    # ------------------------------------------------------------- endpoints: trading

    async def place_order(self, order: OrderRequest, idempotency_key: str, prepaid: bool = False) -> dict[str, Any] | None:
        """Place one order. Returns the canonical engine response, or None in DRY_RUN or when halted.

        `prepaid`: the caller holds a write token from acquire_order_token() for the first
        attempt; it is given back if nothing is sent. On failure after all retries, keep
        `idempotency_key` and retry later with the identical order - the API guarantees it
        will not double-place.
        """
        payload = {**order.to_payload(), "idempotencyKey": idempotency_key}
        if self._halted:
            if prepaid:
                self.refund_order_token()
            logger.warning("Order blocked: shutdown in progress", extra={"payload": payload})
            return None
        crossing = self.self_cross(order)
        if crossing is not None:
            if prepaid:
                self.refund_order_token()
            self._log_self_cross(order, crossing, idempotency_key)
            raise SelfCrossError(payload, crossing)
        if self._dry_run:
            if prepaid:
                self.refund_order_token()
            logger.info("DRY_RUN: would place order", extra={"payload": payload})
            return None

        result = await self._request("POST", "/orders", json_body=payload, bounded_post=True, prepaid=prepaid)
        _log_placed(result, idempotency_key)
        self._note_placed(result, order)
        return result

    async def place_batch(
        self, orders: list[OrderRequest], idempotency_key: str, write_rank: int = WRITE_RANK_ORDER
    ) -> list[dict[str, Any]] | None:
        """POST /orders/batch: up to 50 independent orders under ONE idempotencyKey (one write).

        Returns per-item results `{index, ok, status, data}` in order, or None in DRY_RUN or
        when halted. 207 (some failed) and 422 (all failed) are returned, not raised; the
        caller resubmits failed items under a NEW key. 429 / 502 / 503 are retried by
        _request with the SAME key (the API replays terminal items, resumes the rest).
        `write_rank` WRITE_RANK_QUOTER / WRITE_RANK_LADDER: waits behind cancels and taker
        orders and leaves LADDER_BUCKET_RESERVE tokens in the bucket. Items that would
        self-cross are not sent and come back as failed with status 0 / SELF_CROSS_BLOCKED.
        """
        if not 1 <= len(orders) <= MAX_BATCH_ORDERS:
            raise ValueError(f"batch needs 1..{MAX_BATCH_ORDERS} orders, got {len(orders)}")
        payload = {"orders": [o.to_payload() for o in orders], "idempotencyKey": idempotency_key}
        if self._halted:
            logger.warning("Order blocked: shutdown in progress", extra={"payload": payload})
            return None
        # Self-cross: against our resting orders AND the earlier items of this batch.
        blocked: dict[int, str] = {}
        batch_terms: dict[str, dict[str, tuple[str, float]]] = {}
        for index, order in enumerate(orders):
            earlier = batch_terms.setdefault(str(order.exchange_id), {})
            crossing = self.self_cross(order, earlier)
            if crossing is not None:
                blocked[index] = crossing
                self._log_self_cross(order, crossing, idempotency_key)
            else:
                earlier[f"batch-{index}"] = yes_terms(order.side, order.action, order.price)
        if self._dry_run:
            logger.info("DRY_RUN: would place batch", extra={"idempotency_key": idempotency_key, "orders": len(orders) - len(blocked)})
            return None
        sendable = [i for i in range(len(orders)) if i not in blocked]
        body: dict[str, Any] = {"results": []}
        if sendable:
            if blocked:  # same key, but a different payload than a full batch would have had
                payload = {"orders": [orders[i].to_payload() for i in sendable], "idempotencyKey": idempotency_key}
            try:
                body = await self._request("POST", "/orders/batch", json_body=payload, bounded_post=True, write_rank=write_rank)
            except ApiError as exc:
                if exc.status == 422 and isinstance(exc.body, dict) and "results" in exc.body:
                    body = exc.body  # every item failed terminally: same shape as 207
                else:
                    raise
        # Back to the caller's indexes; blocked items become failed results.
        results = [{**item, "index": sendable[item.get("index", 0)]} for item in body.get("results") or []]
        results += [
            {"index": index, "ok": False, "status": 0, "data": {"code": "SELF_CROSS_BLOCKED", "restingOrderId": crossing}}
            for index, crossing in blocked.items()
        ]
        results.sort(key=lambda item: item["index"])
        for item in results:
            if item.get("ok"):
                _log_placed(item.get("data") or {}, idempotency_key, leg=item.get("index"))
                self._note_placed(item.get("data"), orders[item["index"]])
            else:
                logger.warning(
                    "Batch item failed",
                    extra={"idempotency_key": idempotency_key, "index": item.get("index"), "status": item.get("status"), "data": item.get("data")},
                )
        return results

    async def place_multi_leg(
        self,
        legs: list[OrderRequest],
        idempotency_key: str,
        relationship_constraint: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Place up to 10 legs atomically - all fill-or-rest, or none are placed.

        `relationship_constraint` is a relationship UUID from /relationships/constraints;
        the engine then rejects the whole request (422 RELATIONSHIP_VIOLATION) if the
        leg prices break that relationship. Returns per-leg canonical results in leg
        order, or None in DRY_RUN or when halted.
        """
        if not 1 <= len(legs) <= MAX_MULTI_LEG_LEGS:
            raise ValueError(f"multi-leg order needs 1..{MAX_MULTI_LEG_LEGS} legs, got {len(legs)}")

        payload: dict[str, Any] = {
            "legs": [leg.to_payload() for leg in legs],
            "idempotencyKey": idempotency_key,
        }
        if relationship_constraint is not None:
            payload["relationshipConstraint"] = relationship_constraint
        if self._halted:
            logger.warning("Order blocked: shutdown in progress", extra={"payload": payload})
            return None
        # Atomic: one crossing leg (against resting orders or an earlier leg) blocks them all.
        leg_terms: dict[str, dict[str, tuple[str, float]]] = {}
        for index, leg in enumerate(legs):
            earlier = leg_terms.setdefault(str(leg.exchange_id), {})
            crossing = self.self_cross(leg, earlier)
            if crossing is not None:
                self._log_self_cross(leg, crossing, idempotency_key)
                raise SelfCrossError(payload, crossing)
            earlier[f"leg-{index}"] = yes_terms(leg.side, leg.action, leg.price)
        if self._dry_run:
            logger.info("DRY_RUN: would place multi-leg order", extra={"payload": payload})
            return None

        response = await self._request("POST", "/orders/multi-leg", json_body=payload, bounded_post=True)
        results = [item["data"] for item in sorted(response["results"], key=lambda item: item["index"])]
        for index, result in enumerate(results):
            _log_placed(result, idempotency_key, leg=index)
            self._note_placed(result, legs[index])
        return results

    async def cancel_order(self, order_id: int | str) -> dict[str, Any] | None:
        """Cancel one open order. An order that is already filled/cancelled is not an error.

        Returns the API response (or `{"orderId", "alreadyClosed": True}`), or None in DRY_RUN.
        """
        if self._dry_run:
            logger.info("DRY_RUN: would cancel order", extra={"order_id": order_id})
            return None
        try:
            result = await self._request("DELETE", f"/orders/{order_id}")
        except ApiError as exc:
            if exc.status == 409:
                logger.info("Order already closed, nothing to cancel", extra={"order_id": order_id})
                self._forget_orders(order_id=str(order_id))
                return {"orderId": order_id, "alreadyClosed": True}
            raise
        logger.info("Order cancelled", extra={"order_id": order_id})
        self._forget_orders(order_id=str(order_id))
        return result

    async def cancel_all_orders(
        self,
        *,
        exchange_id: str | None = None,
        market_id: str | None = None,
        tournament_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Cancel open orders (whole default tournament, or narrowed to one exchange/market).

        Returns `{"cancelled": n, "errors": [...]}`, or None in DRY_RUN. Partial failure
        (207) and total failure (422) are returned, not raised, so the caller can inspect
        `errors`; both are logged as errors.
        """
        if exchange_id is not None and market_id is not None:
            raise ValueError("exchange_id and market_id are mutually exclusive")
        body = _clean_params(
            {"exchangeId": exchange_id, "marketId": market_id, "tournamentId": tournament_id}
        )
        if self._dry_run:
            logger.info("DRY_RUN: would cancel all orders", extra={"scope": body or "default tournament"})
            return None

        try:
            result = await self._request("POST", "/orders/cancel-all", json_body=body)
        except ApiError as exc:
            # 422 = every cancellation failed; body is the normal result shape, not an error envelope.
            if exc.status == 422 and isinstance(exc.body, dict) and "cancelled" in exc.body:
                result = exc.body
            else:
                raise

        if result.get("errors"):
            logger.error(
                "Some orders could not be cancelled",
                extra={"cancelled": result.get("cancelled"), "errors": result["errors"]},
            )
        else:
            logger.info("All orders cancelled", extra={"cancelled": result.get("cancelled")})
            # Clean: nothing in scope is resting any more. A market-scoped cancel cannot be
            # mapped to exchanges here; those entries go at the next open-order read.
            if exchange_id is not None:
                self._forget_orders(exchange_id=exchange_id)
            elif market_id is None:
                self._forget_orders()
        return result


# ---------------------------------------------------------------------- smoke tests


async def _smoke_test_read(client: ExchangeClient, log: logging.Logger) -> None:
    account = await client.get_account()
    log.info("Account OK", extra={"username": account.get("username"), "balance": account.get("balance")})

    page = await client.list_markets(status="open", limit=5)
    log.info("Open markets", extra={"total": page["pagination"].get("total")})
    for market in page["data"]:
        tournaments = sorted(
            {ctx.get("tournament", {}).get("name") for ctx in market.get("contexts", [])} - {None}
        )
        log.info(
            market["title"],
            extra={
                "market_id": market["id"],
                "exchange_ids": [ex["id"] for ex in market.get("exchanges", [])],
                "composite": market.get("isComposite"),
                "tournaments": tournaments,
            },
        )

    open_orders = [order async for order in client.iter_orders(status="open")]
    log.info("Open orders", extra={"count": len(open_orders)})

    exchange_ids = [ex["id"] for market in page["data"] for ex in market.get("exchanges", [])]
    for exchange_id, price in (await client.get_prices(exchange_ids)).items():
        log.info(
            "SIG price",
            extra={"exchange_id": exchange_id, "last": price["latestPrice"], "bid": price["bestBid"], "ask": price["bestAsk"]},
        )
    book = await client.get_orderbook(exchange_ids[0], depth=5)
    log.info("SIG orderbook", extra={"exchange_id": exchange_ids[0], "bids": book["bids"], "asks": book["asks"]})

    positions = await client.get_positions()
    pnl = await client.get_pnl()
    log.info("Portfolio", extra={"positions": len(positions), "account_value": pnl.get("totalAccountValue")})

    slug = page["data"][0]["contexts"][0]["tournament"]["slug"]
    tournament = await client.get_tournament(slug)
    leaderboard = await client.get_leaderboard(slug, limit=1)
    log.info(
        "Tournament",
        extra={
            "slug": slug,
            "id": tournament["id"],
            "status": tournament["status"],
            "start": tournament["startDate"],
            "end": tournament["endDate"],
            "members": tournament["memberCount"],
            "my_balance": tournament["myBalance"],
            "my_rank": leaderboard.get("myRank"),
            "ranked": leaderboard.get("total"),
        },
    )

    relationships = [r async for r in client.iter_relationships()]
    constraints = await client.get_constraints()
    log.info(
        "Relationships",
        extra={"relationships": len(relationships), "violations": constraints.get("violationsCount")},
    )


async def _smoke_test_order(client: ExchangeClient, log: logging.Logger, exchange_id: str) -> None:
    # Minimum-risk probe: 1 share, YES at the lowest price (0.005) - practically never
    # fills - then cancel it straight away.
    order = OrderRequest(exchange_id=exchange_id, side="yes", action="buy", quantity=1, price=0.005)
    result = await client.place_order(order, new_idempotency_key("smoke"))
    if result is None:
        log.info("DRY_RUN is on - nothing was sent. Set DRY_RUN=false to test for real.")
        return
    if result.get("open") and result.get("orderId") is not None:
        await client.cancel_order(result["orderId"])
    else:
        log.warning("Probe order is not resting - check it on the website", extra={"result": result})


async def _smoke_test(argv: list[str]) -> int:
    from config import load_settings
    from logging_config import setup_logging

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    log = logging.getLogger("exchange_api.smoke_test")
    log.info("Starting smoke test", extra={"dry_run": settings.dry_run})

    try:
        async with ExchangeClient(settings) as client:
            if len(argv) == 2 and argv[0] == "order-test":
                await _smoke_test_order(client, log, argv[1])
            elif not argv:
                await _smoke_test_read(client, log)
            else:
                log.error("Usage: python exchange_api.py [order-test <exchange_id>]")
                return 2
    except ApiError as exc:
        log.error(
            "API call failed",
            extra={"status": exc.status, "code": exc.code, "detail": exc.message, "details": exc.details},
        )
        return 1
    return 0


if __name__ == "__main__":
    # Sanity checks:
    #   python exchange_api.py                        read-only: account, markets, open orders
    #   python exchange_api.py order-test <exch_id>   place 1 share @ 0.005 and cancel it
    #                                                 (only logs when DRY_RUN=true)
    import sys

    raise SystemExit(asyncio.run(_smoke_test(sys.argv[1:])))
