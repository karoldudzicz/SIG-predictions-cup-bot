"""READ-ONLY research: do the parties' YES prices in a race add up to more (or less) than 1?

Polls SIG's top of book (GET /exchanges/prices, one request per poll for up to 100
exchanges) every --interval-s for --duration-min. For every race with at least two mapped
parties, all priced:

    sum_bid = sum of the parties' best YES bids    -> an edge when > 1 + BASKET_THRESHOLD
              (sell YES on every party for more than the at most 1 we can owe)
    sum_ask = sum of the parties' best YES asks    -> an edge when < 1 - BASKET_THRESHOLD
              (buy YES on every party for less than 1; only a sure 1 if one of the LISTED
              parties wins - an unlisted winner pays nothing)

Each observation of an edge is logged with its timestamp and written to
data/edge_scan/basket.csv. Consecutive polls with the same race and kind form an episode;
when an episode starts the order books of its legs are read once (depth 1) and depth =
the smallest best-level quantity across the legs (shares that could be done on every leg).
At the end: how many episodes, how long they lasted, how deep they were -> log and
data/edge_scan/basket_summary.json.

Requests go through edge_scan.SigReader (GET only, spaced by --max-reads-per-min, 429 ->
pause for Retry-After); the client is created with DRY_RUN forced on and its kill switch
engaged, and this script never calls an order or cancel method.

    python scripts/basket_scan.py [--duration-min 60] [--interval-s 15] [--max-reads-per-min 30]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from config import load_settings  # noqa: E402
from edge_scan import DEFAULT_MAX_READS_PER_MIN, OUT_DIR, SigReader  # noqa: E402
from exchange_api import ExchangeClient  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402

logger = logging.getLogger("basket_scan")

BASKET_THRESHOLD = 0.01
PRICES_BATCH_SIZE = 100  # API maximum ids per /exchanges/prices call
CSV_FIELDS = ("time", "race", "kind", "sum", "excess", "legs", "depth")


@dataclass(frozen=True)
class Observation:
    time: float
    race: str
    kind: str  # "bid_over" (sum_bid > 1 + t) or "ask_under" (sum_ask < 1 - t)
    total: float
    excess: float  # how far past 1, always > 0
    legs: tuple[tuple[str, str, float], ...]  # (party, exchange_id, price)


@dataclass
class Episode:
    race: str
    kind: str
    start: float
    last: float
    max_excess: float
    depth: float | None = None
    polls: int = 1


def basket_observations(now: float, races: dict[str, list[tuple[str, str]]], prices: dict[str, dict[str, Any]], threshold: float = BASKET_THRESHOLD) -> list[Observation]:
    """races: race key -> [(party, exchange_id)]. prices: exchange_id -> {bestBid, bestAsk}.
    A race is judged on a side only when every party has a price on that side."""
    found = []
    for race, legs in races.items():
        if len(legs) < 2:
            continue
        for kind, key in (("bid_over", "bestBid"), ("ask_under", "bestAsk")):
            quoted = [(party, eid, (prices.get(eid) or {}).get(key)) for party, eid in legs]
            if any(price is None for _, _, price in quoted):
                continue
            total = round(sum(float(price) for _, _, price in quoted), 4)
            excess = total - 1.0 if kind == "bid_over" else 1.0 - total
            if excess > threshold + 1e-9:
                found.append(Observation(now, race, kind, total, round(excess, 4), tuple((p, e, float(x)) for p, e, x in quoted)))
    return found


@dataclass
class EpisodeTracker:
    """Open episodes per (race, kind); one that is not seen in a poll is closed."""

    open: dict[tuple[str, str], Episode] = field(default_factory=dict)
    closed: list[Episode] = field(default_factory=list)

    def observe(self, now: float, observations: list[Observation]) -> list[Episode]:
        """Returns the episodes that STARTED this poll."""
        seen = {(o.race, o.kind): o for o in observations}
        for key in [k for k in self.open if k not in seen]:
            self.closed.append(self.open.pop(key))
        started = []
        for key, obs in seen.items():
            episode = self.open.get(key)
            if episode is None:
                episode = self.open[key] = Episode(obs.race, obs.kind, now, now, obs.excess)
                started.append(episode)
            else:
                episode.last, episode.polls = now, episode.polls + 1
                episode.max_excess = max(episode.max_excess, obs.excess)
        return started

    def finish(self) -> list[Episode]:
        self.closed.extend(self.open.values())
        self.open = {}
        return self.closed


def summarize(episodes: list[Episode], interval_s: float) -> dict[str, Any]:
    """Durations count one poll interval per poll seen (a single sighting = one interval)."""
    durations = [e.last - e.start + interval_s for e in episodes]
    depths = [e.depth for e in episodes if e.depth is not None]
    return {
        "episodes": len(episodes),
        "by_kind": {kind: sum(1 for e in episodes if e.kind == kind) for kind in ("bid_over", "ask_under")},
        "races": sorted({e.race for e in episodes}),
        "duration_s": _stats(durations),
        "max_excess": _stats([e.max_excess for e in episodes]),
        "depth_shares": _stats(depths),
        "episodes_without_depth": len(episodes) - len(depths),
    }


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": round(min(values), 4), "median": round(statistics.median(values), 4), "mean": round(statistics.fmean(values), 4), "max": round(max(values), 4)}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def _read_prices(reader: SigReader, exchange_ids: list[str]) -> dict[str, dict[str, Any]]:
    prices: dict[str, dict[str, Any]] = {}
    for start in range(0, len(exchange_ids), PRICES_BATCH_SIZE):
        chunk = exchange_ids[start : start + PRICES_BATCH_SIZE]
        body = await reader.get("/exchanges/prices", {"ids": ",".join(chunk)}, cache_key="prices", cacheable=lambda b: False)
        for row in body.get("data") or []:
            prices[str(row["exchangeId"])] = row
    return prices


async def _depth(reader: SigReader, observation: Observation) -> float | None:
    """Smallest best-level quantity across the legs, on the side the edge trades against."""
    book_side = "bids" if observation.kind == "bid_over" else "asks"
    quantities = []
    for _, exchange_id, _ in observation.legs:
        book = await reader.get(f"/exchanges/{exchange_id}/orderbook", {"depth": 1}, cache_key="book", cacheable=lambda b: False)
        levels = book.get(book_side) or []
        if not levels:
            return None
        quantities.append(float(levels[0]["quantity"]))
    return min(quantities)


async def run_scan(
    reader: SigReader,
    market_map: MarketMap,
    polls: int,
    interval_s: float,
    out_dir: Path = OUT_DIR,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    races = {key: [(m.party, m.sig_exchange_id) for m in group] for key, group in market_map.by_race().items()}
    exchange_ids = sorted({eid for legs in races.values() for _, eid in legs})
    tracker = EpisodeTracker()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "basket.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for poll in range(polls):
            started = clock()
            prices = await _read_prices(reader, exchange_ids)
            observations = basket_observations(started, races, prices)
            new_episodes = tracker.observe(started, observations)
            by_key = {(o.race, o.kind): o for o in observations}
            for episode in new_episodes:
                episode.depth = await _depth(reader, by_key[(episode.race, episode.kind)])
            depth_of = {(e.race, e.kind): e.depth for e in tracker.open.values()}
            for obs in observations:
                logger.info(
                    "Basket edge",
                    extra={"time": _iso(obs.time), "race": obs.race, "kind": obs.kind, "sum": obs.total, "excess": obs.excess, "legs": obs.legs},
                )
                writer.writerow({
                    "time": _iso(obs.time), "race": obs.race, "kind": obs.kind, "sum": obs.total, "excess": obs.excess,
                    "legs": ";".join(f"{party}:{eid}@{price}" for party, eid, price in obs.legs),
                    "depth": depth_of.get((obs.race, obs.kind)),
                })
            fh.flush()
            if poll < polls - 1:
                await sleep(max(0.0, interval_s - (clock() - started)))
    summary = {**summarize(tracker.finish(), interval_s), "polls": polls, "interval_s": interval_s, "races_watched": len(races)}
    (out_dir / "basket_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Basket scan done", extra=summary)
    return summary


async def _amain(args: argparse.Namespace) -> int:
    # Read-only on purpose: DRY_RUN forced on and the kill switch engaged before any request.
    settings = replace(load_settings(), dry_run=True)
    setup_logging(settings.log_level, settings.log_dir)
    market_map = MarketMap.load()
    async with ExchangeClient(settings) as client:
        client.halt("basket_scan.py is read-only research")
        reader = SigReader(client, OUT_DIR / "raw", max_reads_per_min=args.max_reads_per_min)
        polls = max(1, int(args.duration_min * 60 // args.interval_s))
        await run_scan(reader, market_map, polls, args.interval_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only scan for races whose YES prices sum past 1 on SIG.")
    parser.add_argument("--duration-min", type=float, default=60.0, help="how long to poll (default 60)")
    parser.add_argument("--interval-s", type=float, default=15.0, help="seconds between polls (default 15)")
    parser.add_argument(
        "--max-reads-per-min",
        type=float,
        default=DEFAULT_MAX_READS_PER_MIN,
        help=f"request spacing (default {DEFAULT_MAX_READS_PER_MIN}; SIG allows 100/min per account, the bot uses ~45)",
    )
    args = parser.parse_args(argv)
    if not 0 < args.max_reads_per_min <= 100:
        parser.error("--max-reads-per-min must be in (0, 100]: SIG allows 100 reads/min for the whole account")
    if args.interval_s < 1 or args.duration_min <= 0:
        parser.error("--interval-s must be >= 1 and --duration-min > 0")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
