"""READ-ONLY collector: SIG quote snapshots for an offline re-test of maker markouts.

Long-running and nohup-friendly; SIGINT / SIGTERM stop it gracefully after the current read.
Everything goes through research.reader (any write raises; RESEARCH_READS_PER_MIN, default 10,
hard max 15; Retry-After honoured, 429 backs off, repeated 429 abort the run). Nothing runs
without --yes; the estimated reads per hour are logged first.

What is polled: GET /exchanges/prices - the bulk call the bot uses - returns bestBid / bestAsk /
latestPrice / spread for up to 100 exchanges per read, but NO sizes. That is enough for a quote
mid and a Lee-Ready classification, so every POLL_S (30 s) the bulk call covers every market
(the market map plus every exchange in the saved tapes). Sizes need order books: with
--book-markets N (<= 25) the reads left in the budget after the bulk call go to the books
(depth 5) of up to N cheap markets (mid < 0.15 or > 0.85), cycling through them.

One JSONL line per market per poll (and per book read) is appended to
data/edge_scan/quote_snapshots/<UTC date of the poll>/snapshots.jsonl:
{"ts", "exchange_id", "bid", "ask", "mid", "spread_ticks", "latest", "source": "bulk" | "book",
 "bid_size", "ask_size"}. The raw responses also go to raw.jsonl under the start date
(research.reader). Disk: roughly 5 MB per hour each at 30 s.

    nohup .venv/bin/python scripts/quote_snapshots.py --yes [--hours 24] [--book-markets 10] > /dev/null 2>&1 &
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from exchange_api import ApiError  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, ResearchAborted, ResearchReader, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min  # noqa: E402

logger = logging.getLogger("quote_snapshots")

SCRIPT = "quote_snapshots"
POLL_S = 30.0
PRICES_CHUNK = 100
MAX_BOOK_MARKETS = 25
BOOK_DEPTH = 5
TICK = 0.005


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------- plan (pure)


def universe_ids(market_map: MarketMap, tape_records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Mapped exchanges plus every exchange with trades in the saved tapes (so tapes can be matched)."""
    ids = {m.sig_exchange_id for m in market_map}
    for rec in tape_records:
        path = str(rec.get("path") or "")
        if path.startswith("/exchanges/") and path.endswith("/trades"):
            ids.add(path.split("/")[2])
    return sorted(ids)


def plan(n_ids: int, reads_per_min: float, book_markets: int, poll_s: float = POLL_S) -> dict[str, Any]:
    """Reads per poll: the bulk chunks, plus whatever book reads still fit in the budget."""
    chunks = math.ceil(n_ids / PRICES_CHUNK)
    budget = math.floor(poll_s * reads_per_min / 60.0 + 1e-9)
    if chunks > budget:
        raise ValueError(f"{chunks} bulk reads per poll do not fit {reads_per_min:g} reads/min at a {poll_s:g} s poll")
    books = min(book_markets, budget - chunks)
    per_poll = chunks + books
    return {"chunks": chunks, "books_per_poll": books, "reads_per_poll": per_poll, "reads_per_hour": round(per_poll * 3600 / poll_s)}


def snapshot_line(ts: str, exchange_id: str, bid: Any, ask: Any, latest: Any = None, source: str = "bulk",
                  bid_size: Any = None, ask_size: Any = None) -> dict[str, Any]:
    bid = None if bid is None else float(bid)
    ask = None if ask is None else float(ask)
    return {"ts": ts, "exchange_id": exchange_id, "bid": bid, "ask": ask,
            "mid": None if bid is None or ask is None else round((bid + ask) / 2, 6),
            "spread_ticks": None if bid is None or ask is None else round((ask - bid) / TICK), "latest": None if latest is None else float(latest),
            "source": source, "bid_size": None if bid_size is None else float(bid_size), "ask_size": None if ask_size is None else float(ask_size)}


def cheap_ids(lines: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(l["exchange_id"] for l in lines if l["mid"] is not None and (l["mid"] < 0.15 or l["mid"] > 0.85))


def append_lines(lines: Sequence[Mapping[str, Any]], root: Path) -> Path:
    """Append to <root>/<UTC date of the first line>/snapshots.jsonl (a long run rolls over at midnight)."""
    day = lines[0]["ts"][:10]
    path = root / day / "snapshots.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, separators=(",", ":")) + "\n")
    return path


# ---------------------------------------------------------------------- the loop


async def poll_once(reader: ResearchReader, ids: Sequence[str], tid: str | None, books: int, cursor: int, root: Path, now: Callable[[], datetime]) -> tuple[int, int]:
    """One poll: bulk prices for every id, then up to `books` cheap books (cycling from `cursor`).
    -> (lines written, next cursor)."""
    ts = _iso(now())
    lines: list[dict[str, Any]] = []
    for start in range(0, len(ids), PRICES_CHUNK):
        body = await reader.get("/exchanges/prices", {"ids": ",".join(ids[start : start + PRICES_CHUNK]), "tournamentId": tid})
        for row in body.get("data") or []:
            if row.get("bestBid") is not None or row.get("bestAsk") is not None:
                lines.append(snapshot_line(ts, str(row["exchangeId"]), row.get("bestBid"), row.get("bestAsk"), row.get("latestPrice")))
    cheap = cheap_ids(lines)
    for k in range(min(books, len(cheap))):
        eid = cheap[(cursor + k) % len(cheap)]
        book = await reader.get(f"/exchanges/{eid}/orderbook", {"depth": BOOK_DEPTH, "tournamentId": tid})
        bids, asks = book.get("bids") or [], book.get("asks") or []
        lines.append(snapshot_line(_iso(now()), eid, bids[0]["price"] if bids else None, asks[0]["price"] if asks else None, None, "book",
                                   bids[0]["quantity"] if bids else None, asks[0]["quantity"] if asks else None))
    if lines:
        append_lines(lines, root)
    return len(lines), cursor + min(books, len(cheap))


async def run(reader: ResearchReader, ids: Sequence[str], books: int, stop: asyncio.Event, max_polls: int | None = None, root: Path | None = None,
              clock: Callable[[], float] = time.monotonic, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
              poll_s: float = POLL_S) -> dict[str, Any]:
    root = root or OUT_ROOT / SCRIPT
    tid = (await reader.get(f"/tournaments/{reader.settings.tournament_slug}")).get("id") if hasattr(reader, "settings") else None
    polls = lines = errors = 0
    cursor = 0
    while not stop.is_set() and (max_polls is None or polls < max_polls):
        started = clock()
        try:
            written, cursor = await poll_once(reader, ids, tid, books, cursor, root, now)
            lines += written
        except ResearchAborted as exc:
            logger.error("Repeated 429: stopping the collector", extra={"error": str(exc), "polls": polls})
            break
        except ApiError as exc:
            errors += 1
            logger.warning("Poll failed, next poll as planned", extra={"status": exc.status, "code": exc.code})
        polls += 1
        if polls % 20 == 0:
            logger.info("Collector alive", extra={"polls": polls, "lines": lines, "errors": errors})
        if max_polls is not None and polls >= max_polls:
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, poll_s - (clock() - started)))
        except TimeoutError:
            pass
    return {"polls": polls, "lines": lines, "errors": errors}


def install_stop_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # Windows: no loop signal handlers
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))


async def _amain(ids: Sequence[str], reads_per_min: float, books: int, hours: float | None) -> int:
    stop = asyncio.Event()
    install_stop_handlers(stop)
    max_polls = None if hours is None else max(1, int(hours * 3600 // POLL_S))
    async with research_reader(SCRIPT, reads_per_min) as reader:
        result = await run(reader, ids, books, stop, max_polls)
    logger.info("Collector stopped", extra=result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only collector of SIG quote snapshots (bulk prices every 30 s, optional books).")
    add_research_args(parser)
    parser.add_argument("--hours", type=float, default=None, help="stop after this many hours (default: until SIGINT / SIGTERM)")
    parser.add_argument("--book-markets", type=int, default=0, help=f"cheap markets whose books (sizes) are read with the leftover budget (<= {MAX_BOOK_MARKETS})")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.book_markets <= MAX_BOOK_MARKETS:
        parser.error(f"--book-markets must be in [0, {MAX_BOOK_MARKETS}]")
    tape_records = [rec for p in sorted((OUT_ROOT / "tape_patterns").glob("*/raw.jsonl")) for rec in read_jsonl(p)]
    ids = universe_ids(MarketMap.load(), tape_records)
    try:
        p = plan(len(ids), reads_per_min, args.book_markets)
    except ValueError as exc:
        parser.error(str(exc))
    logger.info("Collector plan", extra={"exchanges": len(ids), **p, "poll_s": POLL_S, "sizes": "book reads only" if p["books_per_poll"] else "none (bulk has no sizes)"})
    hours = args.hours or 1.0
    estimate = ReadEstimate(1 + int(p["reads_per_poll"] * hours * 3600 / POLL_S), reads_per_min, extra_s=max(0.0, hours * 3600 - p["reads_per_poll"] * hours * 3600 / POLL_S * 60 / reads_per_min))
    if not confirm_fetch(estimate, args.yes):
        logger.info("Estimate shown for one hour" if args.hours is None else "Estimate shown for --hours", extra={"reads_per_hour": p["reads_per_hour"]})
        return 0
    return asyncio.run(_amain(ids, reads_per_min, p["books_per_poll"], args.hours))


if __name__ == "__main__":
    raise SystemExit(main())
