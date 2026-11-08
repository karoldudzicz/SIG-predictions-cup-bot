"""READ-ONLY research: liquidity holes and D-vs-R parity gaps on SIG since tournament start.

This script only reads. It talks to SIG through `SigReader`, which exposes GET and nothing
else; on top of that the ExchangeClient is created with DRY_RUN forced on and its kill
switch engaged (`halt()`), so no order could be placed even by mistake. It never calls an
order or cancel method.

A) Liquidity holes - per exchange, 1m candles since --since. Reference = median close of
   the 30 surrounding candles (15 before / 15 after in the sparse series, the candle itself
   excluded; topped up from the other side at the edges). A hole is a candle with
   low <= ref - 0.10 (YES-down) or high >= ref + 0.10 (YES-up). For each hole the trades of
   that minute give: shares traded at least 0.10 beyond ref on the hole's side, captureable
   value = sum(size * |ref - price|), and minutes until a close came back within 0.03 of ref.
   -> data/edge_scan/holes.csv

B) Parity D vs R - two-party races only (no Independent market). 1m closes are put on a
   minute grid (last close carried forward for at most PARITY_MAX_STALENESS_MIN minutes)
   and gap = D_close - (1 - R_close), i.e. the price difference for the same bet.
   -> data/edge_scan/parity.csv (minutes with |gap| > 0.02) and parity_races.csv

Both write a summary to the log and to data/edge_scan/summary.json.

Request volume is kept modest (the live bot shares the API key): at most 3 concurrent
requests, ExchangeClient's GET timeouts/retries, and immutable responses are cached under
data/edge_scan/raw/ so reruns do not refetch them.

    python edge_scan.py [--since 2026-10-01T16:00:00Z] [--only-exchange 1077]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp

from config import load_settings
from exchange_api import ApiError, AuthError, ExchangeClient, TermsNotAcknowledgedError
from logging_config import setup_logging
from market_mapper import DATA_DIR, MarketMap

logger = logging.getLogger("edge_scan")

TOURNAMENT_ID = "bda92870-621e-47b0-bc3c-3602c5c26f55"  # midterm-elections
TOURNAMENT_START = "2026-10-01T16:00:00Z"

OUT_DIR = DATA_DIR / "edge_scan"
MAX_CONCURRENCY = 3  # the live bot shares the API key
DEFAULT_MAX_READS_PER_MIN = 30  # leaves the live bot (~40/min) room under SIG's 100 reads/min
RATE_LIMIT_RETRIES = 3
DEFAULT_RETRY_AFTER_S = 60.0  # SIG sends Retry-After: 60 when the account limit is hit
CANDLE_PAGE_LIMIT = 1000  # API maximum for price-history
TRADE_PAGE_LIMIT = 200  # API maximum for trades
# A window that ended at least this long ago can no longer change -> safe to cache.
IMMUTABLE_AFTER_S = 300

HOLE_THRESHOLD = 0.10
RECOVERY_BAND = 0.03
REF_WINDOW = 30  # neighbouring candles for the reference median
MIN_REF_NEIGHBOURS = 10  # fewer than this -> no reference, candle not judged
PARITY_THRESHOLDS = (0.02, 0.05)
PARITY_MAX_STALENESS_MIN = 30  # a close older than this does not take part in a gap
_EPS = 1e-9


# ---------------------------------------------------------------------- read-only API access


class SigReader:
    """GET-only, cached, rate- and concurrency-limited view of SIG. Exposes nothing that can trade.

    Requests are spaced at least 60 / max_reads_per_min seconds apart (the account has 100
    reads/min and the live bot uses ~40). A 429 pauses ALL of the scan's requests for its
    Retry-After (60 s if absent), then retries, at most RATE_LIMIT_RETRIES times.
    """

    def __init__(self, client: ExchangeClient, raw_dir: Path, max_reads_per_min: float = DEFAULT_MAX_READS_PER_MIN) -> None:
        self._client = client
        self._raw_dir = raw_dir
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._interval = 60.0 / max_reads_per_min
        self._next_slot = 0.0
        self._slot_lock = asyncio.Lock()
        self.requests = 0
        self.cache_hits = 0
        self.rate_limited = 0

    async def _throttle(self) -> None:
        async with self._slot_lock:
            now = time.monotonic()
            delay = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)

    def _pause_all(self, seconds: float) -> None:
        self._next_slot = max(self._next_slot, time.monotonic() + seconds)

    async def get(self, path: str, params: dict[str, Any], cache_key: str, cacheable: Callable[[Any], bool]) -> Any:
        """GET `path`; served from the raw cache when present. Stored only if `cacheable(body)`."""
        digest = hashlib.sha1(json.dumps([path, params], sort_keys=True).encode()).hexdigest()[:16]
        cache_file = self._raw_dir / f"{cache_key}_{digest}.json"
        if cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))
        for attempt in range(1, RATE_LIMIT_RETRIES + 2):
            await self._throttle()
            try:
                async with self._semaphore:
                    # ExchangeClient._request("GET") = the bot's GET timeouts, retries and deadline.
                    body = await self._client._request("GET", path, params=params)
                break
            except ApiError as exc:
                if exc.status != 429 or attempt > RATE_LIMIT_RETRIES:
                    raise
                self.rate_limited += 1
                wait = exc.retry_after if exc.retry_after is not None else DEFAULT_RETRY_AFTER_S
                logger.warning("Rate limited, pausing the scan", extra={"path": path, "attempt": attempt, "wait_s": wait})
                self._pause_all(wait)
        self.requests += 1
        if cacheable(body):
            self._raw_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(body), encoding="utf-8")
            tmp.replace(cache_file)
        return body


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _parse_time(raw: str) -> datetime:
    moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def fetch_candles(reader: SigReader, exchange_id: str, since: datetime) -> list[dict[str, Any]]:
    """All 1m candles from `since` to now, following the `coverage.complete` continuation."""
    candles: list[dict[str, Any]] = []
    start = _iso(since)
    while True:
        params = {"resolution": "1m", "from": start, "limit": CANDLE_PAGE_LIMIT, "tournamentId": TOURNAMENT_ID}
        body = await reader.get(
            f"/exchanges/{exchange_id}/price-history",
            params,
            cache_key=f"candles_{exchange_id}",
            # A truncated page covers a closed window in the past; the last page reaches "now".
            cacheable=lambda b: not (b.get("coverage") or {}).get("complete", True),
        )
        candles.extend(body.get("candles") or [])
        if (body.get("coverage") or {}).get("complete", True):
            return candles
        next_start = body.get("to")
        if not next_start or next_start == start:
            logger.warning("Candle pagination did not advance, stopping", extra={"exchange_id": exchange_id, "from": start})
            return candles
        start = next_start


async def fetch_trades(reader: SigReader, exchange_id: str, start: datetime, end: datetime, now: datetime) -> list[dict[str, Any]]:
    """Every trade in [start, end), following cursor pagination."""
    trades: list[dict[str, Any]] = []
    cursor: str | None = None
    immutable = (now - end).total_seconds() >= IMMUTABLE_AFTER_S
    while True:
        params: dict[str, Any] = {"from": _iso(start), "to": _iso(end), "limit": TRADE_PAGE_LIMIT, "tournamentId": TOURNAMENT_ID}
        if cursor:
            params["cursor"] = cursor
        body = await reader.get(f"/exchanges/{exchange_id}/trades", params, cache_key=f"trades_{exchange_id}", cacheable=lambda b: immutable)
        trades.extend(body.get("data") or [])
        pagination = body.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            return trades


# ---------------------------------------------------------------------- analysis A: holes


@dataclass(frozen=True)
class Candle:
    time: datetime
    high: float
    low: float
    close: float


@dataclass
class Hole:
    exchange_id: str
    direction: str  # "down" (YES traded far below ref) or "up" (far above)
    time: datetime
    ref: float
    extreme: float
    depth: float
    recovery_min: float | None
    shares: int | None = None
    value: float | None = None


def parse_candles(rows: list[dict[str, Any]]) -> list[Candle]:
    """Candles with a full high/low/close, oldest first, de-duplicated by time."""
    by_time: dict[datetime, Candle] = {}
    for row in rows:
        if row.get("close") is None or row.get("high") is None or row.get("low") is None:
            continue
        moment = _parse_time(row["time"])
        by_time[moment] = Candle(moment, float(row["high"]), float(row["low"]), float(row["close"]))
    return [by_time[t] for t in sorted(by_time)]


def reference_price(closes: list[float], i: int, window: int = REF_WINDOW) -> float | None:
    """Median close of up to `window` neighbouring candles, candle i excluded."""
    half = window // 2
    before = closes[max(0, i - half) : i]
    after = closes[i + 1 : i + 1 + half]
    if len(before) < half:  # near the start: take more from after
        after = closes[i + 1 : i + 1 + window - len(before)]
    elif len(after) < half:  # near the end: take more from before
        before = closes[max(0, i - (window - len(after))) : i]
    neighbours = before + after
    if len(neighbours) < MIN_REF_NEIGHBOURS:
        return None
    return statistics.median(neighbours)


def find_holes(exchange_id: str, candles: list[Candle], threshold: float = HOLE_THRESHOLD, band: float = RECOVERY_BAND) -> list[Hole]:
    closes = [c.close for c in candles]
    holes: list[Hole] = []
    for i, candle in enumerate(candles):
        ref = reference_price(closes, i)
        if ref is None:
            continue
        for direction, extreme, hit in (
            ("down", candle.low, candle.low <= ref - threshold + _EPS),
            ("up", candle.high, candle.high >= ref + threshold - _EPS),
        ):
            if hit:
                holes.append(
                    Hole(exchange_id, direction, candle.time, ref, extreme, abs(extreme - ref), _recovery_minutes(candles, i, ref, band))
                )
    return holes


def _recovery_minutes(candles: list[Candle], i: int, ref: float, band: float) -> float | None:
    """Minutes from candle i until a close within `band` of ref (0 if its own close is). None if never."""
    start = candles[i].time
    for candle in candles[i:]:
        if abs(candle.close - ref) <= band + _EPS:
            return round((candle.time - start).total_seconds() / 60, 1)
    return None


def hole_trade_stats(hole: Hole, trades: list[dict[str, Any]], threshold: float = HOLE_THRESHOLD) -> tuple[int, float]:
    """(shares, captureable value) of the trades at least `threshold` beyond ref on the hole's side."""
    shares, value = 0, 0.0
    for trade in trades:
        price, size = trade.get("price"), trade.get("size") or 0
        if price is None:
            continue
        beyond = price <= hole.ref - threshold + _EPS if hole.direction == "down" else price >= hole.ref + threshold - _EPS
        if beyond:
            shares += int(size)
            value += size * abs(hole.ref - price)
    return shares, round(value, 4)


# ---------------------------------------------------------------------- analysis B: parity


@dataclass(frozen=True)
class ParityPoint:
    time: datetime
    d_close: float
    r_close: float

    @property
    def gap(self) -> float:
        """D_close - (1 - R_close): > 0 means D YES is dearer than the same bet via R NO."""
        return self.d_close - (1.0 - self.r_close)


def parity_series(d: list[Candle], r: list[Candle], end: datetime, max_stale_min: int = PARITY_MAX_STALENESS_MIN) -> list[ParityPoint]:
    """One point per minute where both markets have a close no older than `max_stale_min`."""
    if not d or not r:
        return []
    d_by_minute = {c.time: c.close for c in d}
    r_by_minute = {c.time: c.close for c in r}
    minute = max(d[0].time, r[0].time)
    stale = timedelta(minutes=max_stale_min)
    last_d = last_r = None  # (time, close)
    for candle in d:
        if candle.time <= minute:
            last_d = (candle.time, candle.close)
    for candle in r:
        if candle.time <= minute:
            last_r = (candle.time, candle.close)
    points: list[ParityPoint] = []
    while minute < end:
        if minute in d_by_minute:
            last_d = (minute, d_by_minute[minute])
        if minute in r_by_minute:
            last_r = (minute, r_by_minute[minute])
        if last_d and last_r and minute - last_d[0] <= stale and minute - last_r[0] <= stale:
            points.append(ParityPoint(minute, last_d[1], last_r[1]))
        minute += timedelta(minutes=1)
    return points


def two_party_races(market_map: MarketMap) -> dict[str, tuple[str, str]]:
    """race key -> (D exchange id, R exchange id) for races with exactly D and R."""
    races = {}
    for race_key, mappings in market_map.by_race().items():
        parties = {m.party: m.sig_exchange_id for m in mappings}
        if set(parties) == {"D", "R"}:
            races[race_key] = (parties["D"], parties["R"])
    return races


# ---------------------------------------------------------------------- pipeline


async def _safe(coro: Any, what: dict[str, Any]) -> Any:
    """Await a read; on a transient failure log one WARNING and return None (scan continues)."""
    try:
        return await coro
    except (AuthError, TermsNotAcknowledgedError):
        raise
    except (ApiError, aiohttp.ClientError, TimeoutError) as exc:
        logger.warning("Read failed, skipped", extra={**what, "error": repr(exc)})
        return None


async def run_scan(
    reader: SigReader,
    market_map: MarketMap,
    since: datetime,
    now: datetime,
    out_dir: Path = OUT_DIR,
    only_exchange: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    mappings = {m.sig_exchange_id: m for m in market_map}
    races = two_party_races(market_map)
    if only_exchange:
        races = {k: v for k, v in races.items() if only_exchange in v}
        wanted = {only_exchange} | {eid for pair in races.values() for eid in pair}
    else:
        wanted = set(mappings)
    exchange_ids = sorted(e for e in wanted if e in mappings)
    logger.info("Scan starting", extra={"exchanges": len(exchange_ids), "parity_races": len(races), "since": _iso(since), "now": _iso(now)})

    raw = await asyncio.gather(*(_safe(fetch_candles(reader, eid, since), {"exchange_id": eid, "read": "candles"}) for eid in exchange_ids))
    candles = {eid: parse_candles(rows) for eid, rows in zip(exchange_ids, raw) if rows is not None}
    skipped = sorted(set(exchange_ids) - set(candles))

    # A) holes, then the trades of each hole minute.
    hole_exchanges = [only_exchange] if only_exchange else list(candles)
    holes = [hole for eid in hole_exchanges if eid in candles for hole in find_holes(eid, candles[eid])]
    tapes = await asyncio.gather(
        *(
            _safe(fetch_trades(reader, h.exchange_id, h.time, h.time + timedelta(minutes=1), now), {"exchange_id": h.exchange_id, "read": "trades", "time": _iso(h.time)})
            for h in holes
        )
    )
    for hole, trades in zip(holes, tapes):
        if trades is not None:
            hole.shares, hole.value = hole_trade_stats(hole, trades)

    # B) parity.
    parity_rows: list[dict[str, Any]] = []
    race_rows: list[dict[str, Any]] = []
    for race_key, (d_id, r_id) in sorted(races.items()):
        if d_id not in candles or r_id not in candles:
            continue
        points = parity_series(candles[d_id], candles[r_id], now)
        if not points:
            continue
        gaps = [p.gap for p in points]
        worst = max(points, key=lambda p: abs(p.gap))
        race_rows.append(
            {
                "race": race_key,
                "d_exchange_id": d_id,
                "r_exchange_id": r_id,
                "minutes": len(points),
                "minutes_gt_0.02": sum(abs(g) > PARITY_THRESHOLDS[0] for g in gaps),
                "minutes_gt_0.05": sum(abs(g) > PARITY_THRESHOLDS[1] for g in gaps),
                "median_gap": round(statistics.median(gaps), 4),
                "max_abs_gap": round(abs(worst.gap), 4),
                "max_gap_time": _iso(worst.time),
            }
        )
        parity_rows.extend(
            {"race": race_key, "d_exchange_id": d_id, "r_exchange_id": r_id, "time": _iso(p.time), "d_close": p.d_close, "r_close": p.r_close, "gap": round(p.gap, 4)}
            for p in points
            if abs(p.gap) > PARITY_THRESHOLDS[0]
        )

    summary = _summarize(holes, mappings, race_rows, parity_rows)
    summary["requests"], summary["cache_hits"], summary["skipped_exchanges"] = reader.requests, reader.cache_hits, skipped
    summary["since"], summary["now"], summary["duration_s"] = _iso(since), _iso(now), round(time.monotonic() - started, 1)
    _write_outputs(out_dir, holes, mappings, parity_rows, race_rows, summary)
    _log_summary(summary)
    return summary


def _summarize(holes: list[Hole], mappings: dict[str, Any], race_rows: list[dict[str, Any]], parity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_market: dict[str, dict[str, Any]] = defaultdict(lambda: {"holes": 0, "value": 0.0, "shares": 0})
    by_hour: dict[int, dict[str, Any]] = {h: {"holes": 0, "value": 0.0} for h in range(24)}
    for hole in holes:
        market = by_market[hole.exchange_id]
        market["holes"] += 1
        market["value"] += hole.value or 0.0
        market["shares"] += hole.shares or 0
        by_hour[hole.time.hour]["holes"] += 1
        by_hour[hole.time.hour]["value"] += hole.value or 0.0
    top = sorted(by_market.items(), key=lambda item: -item[1]["value"])[:15]
    largest_gaps = sorted(parity_rows, key=lambda row: -abs(row["gap"]))[:10]
    return {
        "holes": {
            "count": len(holes),
            "by_direction": dict(Counter(h.direction for h in holes)),
            "without_trade_data": sum(h.value is None for h in holes),
            "total_value": round(sum(h.value or 0.0 for h in holes), 2),
            "top_markets": [
                {"exchange_id": eid, "title": mappings[eid].title if eid in mappings else None, **{k: round(v, 2) for k, v in stats.items()}}
                for eid, stats in top
            ],
            "by_hour_utc": {str(h): {"holes": v["holes"], "value": round(v["value"], 2)} for h, v in by_hour.items()},
        },
        "parity": {
            "races": len(race_rows),
            "minutes": sum(r["minutes"] for r in race_rows),
            "minutes_gt_0.02": sum(r["minutes_gt_0.02"] for r in race_rows),
            "minutes_gt_0.05": sum(r["minutes_gt_0.05"] for r in race_rows),
            "largest_gaps": largest_gaps,
            "per_race_median_gap": {r["race"]: r["median_gap"] for r in race_rows},
        },
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_outputs(out_dir: Path, holes: list[Hole], mappings: dict[str, Any], parity_rows, race_rows, summary) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hole_rows = [
        {
            "exchange_id": h.exchange_id,
            "race": mappings[h.exchange_id].race.key if h.exchange_id in mappings else "",
            "party": mappings[h.exchange_id].party if h.exchange_id in mappings else "",
            "time": _iso(h.time),
            "direction": h.direction,
            "ref": round(h.ref, 4),
            "extreme": h.extreme,
            "depth": round(h.depth, 4),
            "shares": h.shares,
            "value": h.value,
            "recovery_min": h.recovery_min,
        }
        for h in sorted(holes, key=lambda h: (h.time, h.exchange_id))
    ]
    _write_csv(out_dir / "holes.csv", ["exchange_id", "race", "party", "time", "direction", "ref", "extreme", "depth", "shares", "value", "recovery_min"], hole_rows)
    _write_csv(out_dir / "parity.csv", ["race", "d_exchange_id", "r_exchange_id", "time", "d_close", "r_close", "gap"], parity_rows)
    _write_csv(
        out_dir / "parity_races.csv",
        ["race", "d_exchange_id", "r_exchange_id", "minutes", "minutes_gt_0.02", "minutes_gt_0.05", "median_gap", "max_abs_gap", "max_gap_time"],
        race_rows,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def _log_summary(summary: dict[str, Any]) -> None:
    holes, parity = summary["holes"], summary["parity"]
    logger.info(
        "Holes summary",
        extra={k: holes[k] for k in ("count", "by_direction", "without_trade_data", "total_value")},
    )
    for rank, market in enumerate(holes["top_markets"], start=1):
        logger.info("Top hole market", extra={"rank": rank, **market})
    logger.info("Holes by hour (UTC)", extra={"by_hour_utc": holes["by_hour_utc"]})
    logger.info("Parity summary", extra={k: parity[k] for k in ("races", "minutes", "minutes_gt_0.02", "minutes_gt_0.05")})
    for row in parity["largest_gaps"]:
        logger.info("Largest parity gap", extra=row)
    logger.info(
        "Scan done",
        extra={k: summary[k] for k in ("requests", "cache_hits", "skipped_exchanges", "duration_s")},
    )


# ---------------------------------------------------------------------- CLI


async def _amain(args: argparse.Namespace) -> int:
    # Read-only on purpose: DRY_RUN forced on and the kill switch engaged before any request.
    settings = replace(load_settings(), dry_run=True)
    setup_logging(settings.log_level, settings.log_dir)
    since = _parse_time(args.since)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    market_map = MarketMap.load()
    if args.only_exchange and market_map.get(args.only_exchange) is None:
        logger.error("Exchange not in the market map", extra={"exchange_id": args.only_exchange})
        return 2
    async with ExchangeClient(settings) as client:
        client.halt("edge_scan.py is read-only research")
        reader = SigReader(client, OUT_DIR / "raw", max_reads_per_min=args.max_reads_per_min)
        await run_scan(reader, market_map, since, now, OUT_DIR, args.only_exchange)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only scan for liquidity holes and D/R parity gaps on SIG.")
    parser.add_argument("--since", default=TOURNAMENT_START, help=f"ISO start (default: tournament start {TOURNAMENT_START})")
    parser.add_argument("--only-exchange", help="scan just this exchange id (and its race for parity)")
    parser.add_argument(
        "--max-reads-per-min",
        type=float,
        default=DEFAULT_MAX_READS_PER_MIN,
        help=f"request spacing (default {DEFAULT_MAX_READS_PER_MIN}; SIG allows 100/min per account, the bot uses ~40)",
    )
    args = parser.parse_args(argv)
    if not 0 < args.max_reads_per_min <= 100:
        parser.error("--max-reads-per-min must be in (0, 100]: SIG allows 100 reads/min for the whole account")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
