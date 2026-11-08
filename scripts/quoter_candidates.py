"""READ-ONLY: which markets look worth quoting in the quoter micro-pilot?

A market is listed when ALL of these hold:
  - it is a D or R market of the market map (the quoter quotes nothing else)
  - we hold no position on it
  - both sides of the SIG book are quoted and the YES mid is within [MID_LOW, MID_HIGH]
  - the SIG spread is at least MIN_SPREAD_TICKS ticks
  - a fresh oracle fair exists and its uncertainty band u <= MAX_U
  - the directional cap (max_directional_fraction of the bankroll) has free room in BOTH
    directions, D and R - measured like the taker: held market value per direction

Columns: exchange_id, race, party, yes_bid, yes_ask, spread_ticks, fair, u, bid_depth,
ask_depth (shares at the best level, from one order book read per listed market, at most
--max-books of them, widest spreads first), free_room_D, free_room_R. Sorted by spread,
widest first -> log and data/edge_scan/quoter_candidates.csv.

Every bid / ask is the OTHER participants' one: where one of our open orders sits at the
bulk top of book, that market's order book is read and our orders taken out.

--ignore-room: skip the directional-room filter (free_room_D / R are still reported).
--spread-histogram: the other participants' spread in ticks (1..7, 8+; count and share)
over every mapped market with both sides quoted, and the 15 widest of them with full data
(race, party, bid, ask, fair, u, depth) whatever the room or the other filters ->
log, data/edge_scan/quoter_spread_widest.csv and quoter_spread_histogram.json.

Reads only: SIG through edge_scan.SigReader (GET only, spaced by --max-reads-per-min,
default 15, 429 -> pause), the client created with DRY_RUN forced on and its kill switch
engaged, plus the public Polymarket / Kalshi read endpoints. Nothing is sent but GETs.

    python scripts/quoter_candidates.py [--max-reads-per-min 15] [--max-books 20] [--ignore-room] [--spread-histogram]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from config import load_settings  # noqa: E402
from edge_scan import OUT_DIR, SigReader  # noqa: E402
from exchange_api import PRICE_TICK, ExchangeClient, book_without_own, yes_terms  # noqa: E402
from ladder import uncertainty_band  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from oracle_api import KalshiClient, OracleQuote, PolymarketClient  # noqa: E402
from risk import Direction, RiskLimits, directional_exposure  # noqa: E402
from strategy import compute_fair_price, holdings_from_positions  # noqa: E402

logger = logging.getLogger("quoter_candidates")

DEFAULT_MAX_READS_PER_MIN = 15  # the live bot shares the account's 100 reads/min
DEFAULT_MAX_BOOKS = 20
MID_LOW, MID_HIGH = 0.15, 0.85
MIN_SPREAD_TICKS = 3
MAX_U = 0.02
CSV_PATH = OUT_DIR / "quoter_candidates.csv"
WIDEST_CSV_PATH = OUT_DIR / "quoter_spread_widest.csv"
HISTOGRAM_PATH = OUT_DIR / "quoter_spread_histogram.json"
WIDEST_COUNT = 15
HISTOGRAM_BUCKETS = ("1", "2", "3", "4", "5", "6", "7", "8+")
PAGE_LIMIT = 200
CSV_FIELDS = ("exchange_id", "race", "party", "yes_bid", "yes_ask", "spread_ticks", "fair", "u", "bid_depth", "ask_depth", "free_room_D", "free_room_R")


@dataclass
class CandidateInputs:
    market_map: MarketMap
    positions: list[dict[str, Any]]
    bankroll: float
    prices: Mapping[str, Mapping[str, Any]]  # exchange_id -> bulk row (bestBid / bestAsk)
    pm_quotes: Mapping[str, OracleQuote]
    ks_quotes: Mapping[str, OracleQuote]
    limits: RiskLimits = field(default_factory=RiskLimits)
    staleness_s: float = 10.0


@dataclass
class Candidate:
    exchange_id: str
    race: str
    party: str
    yes_bid: float
    yes_ask: float
    spread_ticks: int
    fair: float | None  # None only in the widest-spread list: no fresh fair
    u: float | None
    bid_depth: float | None = None
    ask_depth: float | None = None
    free_room_D: float = 0.0
    free_room_R: float = 0.0


def free_directional_room(inp: CandidateInputs) -> dict[str, float]:
    """max_directional_fraction * bankroll minus the held market value per direction."""
    holdings = holdings_from_positions(inp.positions)
    used = directional_exposure(inp.market_map, ((eid, h.side, h.market_value) for eid, h in holdings.items()))
    cap = inp.limits.max_directional_fraction * inp.bankroll
    return {d.value: round(cap - used.get(d, 0.0), 2) for d in (Direction.PRO_DEMOCRAT, Direction.PRO_REPUBLICAN)}


def find_candidates(inp: CandidateInputs, ignore_room: bool = False) -> tuple[list[Candidate], dict[str, Any]]:
    """Listed markets (widest spread first) and a summary with the count dropped per filter.
    `ignore_room`: no directional-room filter (the room is still reported)."""
    held = set(holdings_from_positions(inp.positions))
    room = free_directional_room(inp)
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    if min(room.values()) <= 0 and not ignore_room:
        summary = {"candidates": 0, "free_room": room, "dropped": {"no directional room in both directions": len(inp.market_map)}}
        return [], summary

    found: list[Candidate] = []
    for mapping in inp.market_map:
        eid = mapping.sig_exchange_id
        if mapping.party not in ("D", "R"):
            drop("not D / R")
            continue
        if eid in held:
            drop("position held")
            continue
        row = inp.prices.get(eid) or {}
        bid, ask = row.get("bestBid"), row.get("bestAsk")
        if bid is None or ask is None:
            drop("one-sided or empty book")
            continue
        bid, ask = float(bid), float(ask)
        if not MID_LOW <= (bid + ask) / 2 <= MID_HIGH:
            drop("price outside 0.15-0.85")
            continue
        spread_ticks = round((ask - bid) / PRICE_TICK)
        if spread_ticks < MIN_SPREAD_TICKS:
            drop("spread < 3 ticks")
            continue
        pm = inp.pm_quotes.get(mapping.polymarket_token) if mapping.polymarket_token else None
        ks = inp.ks_quotes.get(mapping.kalshi_ticker) if mapping.kalshi_ticker else None
        fair, _ = compute_fair_price(pm, ks, inp.staleness_s)
        if fair is None:
            drop("no fresh fair")
            continue
        u = uncertainty_band(pm, ks, inp.staleness_s)
        if u > MAX_U + 1e-9:
            drop("u > 0.02")
            continue
        found.append(Candidate(eid, mapping.race.key, mapping.party, bid, ask, spread_ticks, round(fair.prob, 4), round(u, 4), free_room_D=room["D"], free_room_R=room["R"]))
    found.sort(key=lambda c: (-c.spread_ticks, c.exchange_id))
    return found, {"candidates": len(found), "free_room": room, "dropped": dropped, "room_filter": not ignore_room}


def _fair_and_u(inp: CandidateInputs, mapping: Any) -> tuple[float | None, float | None]:
    pm = inp.pm_quotes.get(mapping.polymarket_token) if mapping.polymarket_token else None
    ks = inp.ks_quotes.get(mapping.kalshi_ticker) if mapping.kalshi_ticker else None
    fair, _ = compute_fair_price(pm, ks, inp.staleness_s)
    return (None, None) if fair is None else (round(fair.prob, 4), round(uncertainty_band(pm, ks, inp.staleness_s), 4))


def spread_histogram(inp: CandidateInputs, widest: int = WIDEST_COUNT) -> tuple[dict[str, Any], list[Candidate]]:
    """The other participants' spread in ticks over every mapped market with both sides
    quoted (buckets 1..7 and 8+, plus "0" for a locked / crossed book if any), and the
    `widest` markets with full data - no room, price, party or fair filter."""
    room = free_directional_room(inp)
    rows: list[Candidate] = []
    for mapping in inp.market_map:
        row = inp.prices.get(mapping.sig_exchange_id) or {}
        bid, ask = row.get("bestBid"), row.get("bestAsk")
        if bid is None or ask is None:
            continue
        bid, ask = float(bid), float(ask)
        fair, u = _fair_and_u(inp, mapping)
        rows.append(Candidate(mapping.sig_exchange_id, mapping.race.key, mapping.party, bid, ask, round((ask - bid) / PRICE_TICK),
                              fair, u, free_room_D=room["D"], free_room_R=room["R"]))
    counts = {label: 0 for label in HISTOGRAM_BUCKETS}
    for row in rows:
        label = "0" if row.spread_ticks <= 0 else ("8+" if row.spread_ticks >= 8 else str(row.spread_ticks))
        counts[label] = counts.get(label, 0) + 1
    total = len(rows)
    histogram = {label: {"count": n, "share": round(n / total, 3) if total else 0.0} for label, n in counts.items()}
    rows.sort(key=lambda c: (-c.spread_ticks, c.exchange_id))
    return {"two_sided_markets": total, "histogram": histogram}, rows[:widest]


# ---------------------------------------------------------------------- reads


async def gather_inputs(reader: SigReader, polymarket: Any, kalshi: Any, market_map: MarketMap, slug: str, staleness_s: float) -> CandidateInputs:
    def get(path: str, params: dict[str, Any] | None = None) -> Any:
        return reader.get(path, params or {}, cache_key="none", cacheable=lambda b: False)

    tournament = await get(f"/tournaments/{slug}")
    bankroll = float((await get("/portfolio/pnl", {"period": "all"})).get("totalAccountValue") or 0.0)
    positions = (await get("/portfolio/positions")).get("positions") or []
    open_orders: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = await get("/orders", {"status": "open", "limit": PAGE_LIMIT, "cursor": cursor})
        open_orders.extend(page.get("data") or [])
        cursor = (page.get("pagination") or {}).get("nextCursor")
        if not (page.get("pagination") or {}).get("hasMore") or not cursor:
            break
    ids = [m.sig_exchange_id for m in market_map]
    prices: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 100):
        body = await get("/exchanges/prices", {"ids": ",".join(ids[start : start + 100]), "tournamentId": tournament.get("id")})
        prices.update({str(row["exchangeId"]): row for row in body.get("data") or []})
    await _others_only(reader, prices, open_orders)
    pm_quotes, ks_quotes = await asyncio.gather(polymarket.get_quotes(market_map.polymarket_tokens()), kalshi.get_quotes(market_map.kalshi_tickers()))
    return CandidateInputs(market_map, positions, bankroll, prices, pm_quotes, ks_quotes, staleness_s=staleness_s)


def own_at_top(prices: Mapping[str, Mapping[str, Any]], open_orders: list[dict[str, Any]]) -> dict[str, list[tuple[str, float, float]]]:
    """exchange_id -> our resting orders there (book side, YES price, remaining), for every
    market where one of them IS the bulk best bid or ask."""
    own: dict[str, list[tuple[str, float, float]]] = {}
    for order in open_orders:
        if order.get("open") is False:
            continue
        side, price = yes_terms(order.get("side"), order.get("action"), order.get("priceLimit"))
        own.setdefault(str(order.get("exchangeId")), []).append((side, price, abs(float(order.get("quantity") or 0))))
    at_top = {}
    for eid, orders in own.items():
        row = prices.get(eid) or {}
        if any((side == "bid" and price == row.get("bestBid")) or (side == "ask" and price == row.get("bestAsk")) for side, price, _ in orders):
            at_top[eid] = orders
    return at_top


async def _others_only(reader: SigReader, prices: dict[str, dict[str, Any]], open_orders: list[dict[str, Any]]) -> None:
    """Replace bestBid / bestAsk with the other participants' where we sit at the top."""
    for eid, orders in own_at_top(prices, open_orders).items():
        book = book_without_own(await reader.get(f"/exchanges/{eid}/orderbook", {"depth": 20}, cache_key="none", cacheable=lambda b: False), orders)
        bids, asks = book.get("bids") or [], book.get("asks") or []
        prices[eid] = {**prices[eid], "bestBid": float(bids[0]["price"]) if bids else None, "bestAsk": float(asks[0]["price"]) if asks else None, "ownRemoved": True}


async def add_depth(reader: SigReader, candidates: list[Candidate], max_books: int) -> None:
    """Shares at the best bid / ask, one order book read per candidate (widest spreads first)."""
    for candidate in candidates[:max_books]:
        book = await reader.get(f"/exchanges/{candidate.exchange_id}/orderbook", {"depth": 1}, cache_key="none", cacheable=lambda b: False)
        bids, asks = book.get("bids") or [], book.get("asks") or []
        candidate.bid_depth = float(bids[0]["quantity"]) if bids else 0.0
        candidate.ask_depth = float(asks[0]["quantity"]) if asks else 0.0


def write_histogram(histogram: dict[str, Any], widest: list[Candidate], csv_path: Path = WIDEST_CSV_PATH, json_path: Path = HISTOGRAM_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in widest:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps({**histogram, "widest": [asdict(r) for r in widest]}, indent=2), encoding="utf-8")
    logger.info("Spread histogram", extra=histogram)
    for row in widest:
        logger.info("Widest spread", extra=asdict(row))


def write_outputs(candidates: list[Candidate], summary: dict[str, Any], csv_path: Path = CSV_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))
    for candidate in candidates:
        logger.info("Quoter candidate", extra=asdict(candidate))
    logger.info("Quoter candidates done", extra=summary)


async def _amain(args: argparse.Namespace) -> int:
    # Read-only on purpose: DRY_RUN forced on and the kill switch engaged before any request.
    settings = replace(load_settings(), dry_run=True)
    setup_logging(settings.log_level, settings.log_dir)
    market_map = MarketMap.load()
    async with ExchangeClient(settings) as client, PolymarketClient() as polymarket, KalshiClient() as kalshi:
        client.halt("quoter_candidates.py is read-only")
        reader = SigReader(client, OUT_DIR / "raw", max_reads_per_min=args.max_reads_per_min)
        inp = await gather_inputs(reader, polymarket, kalshi, market_map, settings.tournament_slug, settings.oracle_staleness_threshold_s)
        candidates, summary = find_candidates(inp, ignore_room=args.ignore_room)
        await add_depth(reader, candidates, args.max_books)
        write_outputs(candidates, summary)
        if args.spread_histogram:
            histogram, widest = spread_histogram(inp)
            await add_depth(reader, widest, WIDEST_COUNT)
            write_histogram(histogram, widest)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only list of markets worth quoting in the quoter pilot.")
    parser.add_argument("--max-reads-per-min", type=float, default=DEFAULT_MAX_READS_PER_MIN, help=f"SIG read spacing (default {DEFAULT_MAX_READS_PER_MIN})")
    parser.add_argument("--max-books", type=int, default=DEFAULT_MAX_BOOKS, help=f"order books read for depth, widest spreads first (default {DEFAULT_MAX_BOOKS})")
    parser.add_argument("--ignore-room", action="store_true", help="do not filter on directional room (still reported)")
    parser.add_argument("--spread-histogram", action="store_true", help=f"histogram of the other participants' spreads + the {WIDEST_COUNT} widest")
    args = parser.parse_args(argv)
    if not 0 < args.max_reads_per_min <= 30:
        parser.error("--max-reads-per-min must be in (0, 30]: the live bot uses up to ~80 of the account's 100 reads/min")
    if not 0 <= args.max_books <= 100:
        parser.error("--max-books must be in [0, 100]")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
