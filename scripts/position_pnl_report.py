"""READ-ONLY: unrealized P&L of every held position, and whether SIG converged to the oracle.

For every position in /portfolio/positions:
  1. Entry: average buy price on the held side from our REST fills (fills.py: a NO fill
     carries a NO price; also shown YES-denominated), first fill time, size. The API's
     average cost (costBasis / shares) is listed next to it as a check.
  2. Now: SIG bid / ask / mid (bulk /exchanges/prices with the tournament id) and the
     oracle fair (strategy.compute_fair_price), all in the held side's terms too.
  3. Gap at entry = fair at entry - entry price; gap now = fair now - SIG mid now; both in
     the held side's terms (> 0: the oracle says the position is worth more than the price).
     Fair at entry = the fill-weighted fair_at_fill of data/markout.jsonl; the API keeps no
     history of our oracle fair, so fills older than the markout journal have none.
  4. Unrealized P&L by mid and by bid (what a sale would realize), in SUSQies and in ticks
     per share.
  5. Entry groups: "old" (first fill before --old-until, default 2026-10-04) vs "new" (first
     fill at or after --new-since, default 2026-10-05 13:41 UTC) vs "other".
  6. Old positions: SIG move since entry (mid now - entry price; the entry price stands in
     for SIG at entry, which the API does not keep - it includes the spread paid) vs oracle
     move since entry (fair now - fair at entry), and which of them closed the gap.
  7. Summary: P&L by bid (also per group), share of positions whose gap closed >= 50%,
     median gap change, 10 worst and 10 best by bid -> log, data/edge_scan/position_pnl.csv
     and position_pnl_summary.json, with notes on every piece of history that is missing.

Fills whose action is not in the order journal (data/order_journal.jsonl) are taken as
buys and counted in the notes. Reads only: SIG through edge_scan.SigReader (GET only,
default 15 reads/min), DRY_RUN forced on and the kill switch engaged, the public oracle
read endpoints, and the local journals.

    python scripts/position_pnl_report.py [--old-until ISO] [--new-since ISO] [--max-reads-per-min 15]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere

from config import load_settings  # noqa: E402
from edge_scan import OUT_DIR, SigReader  # noqa: E402
from exchange_api import PRICE_TICK, ExchangeClient  # noqa: E402
from fills import fill_price_side, fill_shares  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from markout import MARKOUT_JOURNAL_PATH, unix_time  # noqa: E402
from oracle_api import KalshiClient, OracleQuote, PolymarketClient  # noqa: E402
from strategy import JOURNAL_PATH, OrderJournal, SigBook, compute_fair_price, holdings_from_positions  # noqa: E402

logger = logging.getLogger("position_pnl_report")

DEFAULT_MAX_READS_PER_MIN = 15  # the live bot shares the account's 100 reads/min
DEFAULT_OLD_UNTIL = "2026-10-04T00:00:00Z"  # old = first fill on Oct 1-3
DEFAULT_NEW_SINCE = "2026-10-05T13:41:00Z"
CONVERGED_SHARE = 0.5  # a gap that closed by at least this much has "converged"
TOP_N = 10
PAGE_LIMIT = 200
CSV_PATH = OUT_DIR / "position_pnl.csv"
SUMMARY_PATH = OUT_DIR / "position_pnl_summary.json"


@dataclass
class PnlInputs:
    market_map: MarketMap
    positions: list[dict[str, Any]]
    fills: list[dict[str, Any]]  # REST fills, any order
    prices: Mapping[str, Mapping[str, Any]]  # exchange_id -> bulk row (bestBid / bestAsk)
    pm_quotes: Mapping[str, OracleQuote]
    ks_quotes: Mapping[str, OracleQuote]
    fair_at_fill: Mapping[Any, float] = field(default_factory=dict)  # fill id -> YES fair when the fill was seen (markout.jsonl)
    order_actions: Mapping[str, str] = field(default_factory=dict)  # order id -> "buy" / "sell" (order journal)
    staleness_s: float = 10.0
    old_until: float = 0.0  # unix
    new_since: float = float("inf")  # unix


@dataclass
class PositionPnl:
    exchange_id: str
    race: str
    party: str
    side: str
    shares: int
    group: str  # old | new | other | unknown (no fill found)
    first_fill: str | None
    buy_fills: int
    entry_side: float | None  # average buy price, held side's terms
    entry_yes: float | None
    api_avg_cost_side: float | None  # costBasis / shares
    yes_bid: float | None
    yes_ask: float | None
    mid_side: float | None
    bid_side: float | None
    fair_side_now: float | None
    fair_side_entry: float | None  # fill-weighted, from markout.jsonl
    gap_entry: float | None  # fair at entry - entry price
    gap_now: float | None  # fair now - SIG mid now
    gap_change: float | None  # gap now - gap at entry
    gap_closed_share: float | None  # (gap at entry - gap now) / gap at entry, when gap at entry > 0
    pnl_mid: float | None
    pnl_bid: float | None
    pnl_mid_ticks: float | None  # per share
    pnl_bid_ticks: float | None
    sig_move: float | None  # mid now - entry price (old positions)
    oracle_move: float | None  # fair now - fair at entry (old positions)
    driver: str = ""  # old positions: what moved the gap


def _iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _side(value_yes: float | None, side: str) -> float | None:
    return None if value_yes is None else (value_yes if side == "yes" else 1.0 - value_yes)


def _driver(sig_move: float | None, oracle_move: float | None, gap_entry: float | None, gap_now: float | None) -> str:
    """Which side closed (or widened) the gap since entry."""
    if gap_entry is None or gap_now is None or sig_move is None or oracle_move is None:
        return "unknown (no fair at entry)"
    closed = gap_entry - gap_now
    if abs(closed) < PRICE_TICK:
        return "neither (gap unchanged)"
    if closed > 0:  # converged: SIG up toward the oracle, or the oracle down toward SIG
        return "SIG moved toward the oracle" if sig_move >= -oracle_move else "oracle moved toward SIG"
    return "SIG moved away from the oracle" if sig_move <= -oracle_move else "oracle moved away from SIG"


def build_report(inp: PnlInputs) -> tuple[list[PositionPnl], dict[str, Any]]:
    holdings = holdings_from_positions(inp.positions)
    notes: dict[str, int] = {"fills_without_action_taken_as_buy": 0, "positions_without_buy_fills": 0,
                             "positions_without_fair_at_entry": 0, "positions_without_fair_now": 0, "positions_without_sig_price": 0}
    by_exchange: dict[str, list[dict[str, Any]]] = {}
    for fill in inp.fills:
        by_exchange.setdefault(str(fill.get("exchangeId")), []).append(fill)

    rows: list[PositionPnl] = []
    for eid, held in sorted(holdings.items()):
        mapping = inp.market_map.get(eid)
        buys = []
        for fill in sorted(by_exchange.get(eid, []), key=lambda f: unix_time(f.get("filledAt"))):
            if fill.get("side") != held.side:
                continue
            action = inp.order_actions.get(str(fill.get("orderId")))
            if action is None:
                notes["fills_without_action_taken_as_buy"] += 1
            elif action != "buy":
                continue
            price = fill_price_side(fill)
            if price is not None and fill_shares(fill) > 0:
                buys.append((fill, price, fill_shares(fill)))
        bought = sum(shares for _, _, shares in buys)
        entry_side = sum(price * shares for _, price, shares in buys) / bought if bought else None
        first_ts = unix_time(buys[0][0].get("filledAt")) if buys else None
        if not buys:
            notes["positions_without_buy_fills"] += 1
        group = "unknown" if first_ts is None else ("old" if first_ts < inp.old_until else ("new" if first_ts >= inp.new_since else "other"))

        known = [(inp.fair_at_fill[f.get("id")], shares) for f, _, shares in buys if inp.fair_at_fill.get(f.get("id")) is not None]
        fair_side_entry = _side(sum(fair * shares for fair, shares in known) / sum(s for _, s in known), held.side) if known else None
        if fair_side_entry is None:
            notes["positions_without_fair_at_entry"] += 1

        row = inp.prices.get(eid) or {}
        book = SigBook(row.get("bestBid"), row.get("bestAsk"))
        if book.yes_bid is None and book.yes_ask is None:
            notes["positions_without_sig_price"] += 1
        bid_side, ask_side = book.sell_price(held.side), book.buy_price(held.side)
        mid_side = (bid_side + ask_side) / 2 if bid_side is not None and ask_side is not None else None
        fair = None
        if mapping is not None:
            pm = inp.pm_quotes.get(mapping.polymarket_token) if mapping.polymarket_token else None
            ks = inp.ks_quotes.get(mapping.kalshi_ticker) if mapping.kalshi_ticker else None
            fair, _ = compute_fair_price(pm, ks, inp.staleness_s)
        fair_side_now = _side(fair.prob, held.side) if fair is not None else None
        if fair_side_now is None:
            notes["positions_without_fair_now"] += 1

        gap_entry = fair_side_entry - entry_side if fair_side_entry is not None and entry_side is not None else None
        gap_now = fair_side_now - mid_side if fair_side_now is not None and mid_side is not None else None
        gap_change = gap_now - gap_entry if gap_now is not None and gap_entry is not None else None
        closed_share = (gap_entry - gap_now) / gap_entry if gap_change is not None and gap_entry > PRICE_TICK / 2 else None
        pnl_mid = held.shares * (mid_side - entry_side) if mid_side is not None and entry_side is not None else None
        pnl_bid = held.shares * (bid_side - entry_side) if bid_side is not None and entry_side is not None else None
        sig_move = mid_side - entry_side if group == "old" and mid_side is not None and entry_side is not None else None
        oracle_move = fair_side_now - fair_side_entry if group == "old" and fair_side_now is not None and fair_side_entry is not None else None
        rows.append(PositionPnl(
            eid, mapping.race.key if mapping else "", mapping.party if mapping else "", held.side, held.shares, group, _iso(first_ts), len(buys),
            _r(entry_side), _r(_side(entry_side, held.side)), _r(held.cost_basis / held.shares) if held.shares else None,
            book.yes_bid, book.yes_ask, _r(mid_side), _r(bid_side), _r(fair_side_now), _r(fair_side_entry),
            _r(gap_entry), _r(gap_now), _r(gap_change), _r(closed_share, 3),
            _r(pnl_mid, 2), _r(pnl_bid, 2),
            _r((mid_side - entry_side) / PRICE_TICK, 2) if pnl_mid is not None else None,
            _r((bid_side - entry_side) / PRICE_TICK, 2) if pnl_bid is not None else None,
            _r(sig_move), _r(oracle_move),
            _driver(sig_move, oracle_move, gap_entry, gap_now) if group == "old" else "",
        ))

    def total(items: list[PositionPnl], name: str) -> float:
        return round(sum(getattr(r, name) or 0.0 for r in items), 2)

    groups = {g: [r for r in rows if r.group == g] for g in ("old", "new", "other", "unknown")}
    with_gap = [r for r in rows if r.gap_closed_share is not None]
    changes = [r.gap_change for r in rows if r.gap_change is not None]
    priced = sorted((r for r in rows if r.pnl_bid is not None), key=lambda r: r.pnl_bid)
    old = groups["old"]
    drivers: dict[str, int] = {}
    for r in old:
        drivers[r.driver] = drivers.get(r.driver, 0) + 1
    summary = {
        "positions": len(rows),
        "pnl_bid_total": total(rows, "pnl_bid"),
        "pnl_mid_total": total(rows, "pnl_mid"),
        "by_group": {g: {"positions": len(items), "pnl_bid": total(items, "pnl_bid"), "pnl_mid": total(items, "pnl_mid")} for g, items in groups.items()},
        "converged_share": round(sum(1 for r in with_gap if r.gap_closed_share >= CONVERGED_SHARE) / len(with_gap), 3) if with_gap else None,
        "positions_with_gap_at_entry": len(with_gap),
        "median_gap_change": round(statistics.median(changes), 4) if changes else None,
        "old_positions_drivers": drivers,
        "old_median_sig_move": round(statistics.median([r.sig_move for r in old if r.sig_move is not None]), 4) if any(r.sig_move is not None for r in old) else None,
        "old_median_oracle_move": round(statistics.median([r.oracle_move for r in old if r.oracle_move is not None]), 4) if any(r.oracle_move is not None for r in old) else None,
        "worst_by_bid": [_brief(r) for r in priced[:TOP_N]],
        "best_by_bid": [_brief(r) for r in reversed(priced[-TOP_N:])],
        "notes": notes,
    }
    return rows, summary


def _brief(row: PositionPnl) -> dict[str, Any]:
    return {"exchange_id": row.exchange_id, "race": row.race, "party": row.party, "side": row.side, "shares": row.shares,
            "group": row.group, "pnl_bid": row.pnl_bid, "pnl_mid": row.pnl_mid, "gap_entry": row.gap_entry, "gap_now": row.gap_now}


# ---------------------------------------------------------------------- local journals


def fair_at_fill_from_markout(lines: list[str]) -> dict[Any, float]:
    """fill id -> YES fair when the bot saw the fill (markout.jsonl "fill" lines)."""
    fairs: dict[Any, float] = {}
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") == "fill" and entry.get("fair_at_fill") is not None:
            fairs[entry.get("fill_id")] = float(entry["fair_at_fill"])
    return fairs


def order_actions_from_journal(path: Path, fills: list[dict[str, Any]]) -> dict[str, str]:
    """order id -> canonical action for the orders behind `fills` (OrderJournal.action_for_order)."""
    if not path.exists():
        return {}
    journal = OrderJournal(path)
    actions = {}
    for order_id in {str(f.get("orderId")) for f in fills}:
        action = journal.action_for_order(order_id)
        if action is not None:
            actions[order_id] = action
    return actions


# ---------------------------------------------------------------------- reads


async def _pages(reader: SigReader, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    items, cursor = [], None
    while True:
        body = await reader.get(path, {**params, "limit": PAGE_LIMIT, "cursor": cursor}, cache_key="none", cacheable=lambda b: False)
        items.extend(body.get("data") or [])
        pagination = body.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            return items


async def gather_inputs(reader: SigReader, polymarket: Any, kalshi: Any, market_map: MarketMap, slug: str, staleness_s: float) -> PnlInputs:
    def get(path: str, params: dict[str, Any] | None = None) -> Any:
        return reader.get(path, params or {}, cache_key="none", cacheable=lambda b: False)

    tournament = await get(f"/tournaments/{slug}")
    positions = (await get("/portfolio/positions")).get("positions") or []
    fills = await _pages(reader, f"/tournaments/{slug}/portfolio/fills", {})
    held = sorted({str(p.get("exchangeId")) for p in positions})
    prices: dict[str, dict[str, Any]] = {}
    for start in range(0, len(held), 100):
        body = await get("/exchanges/prices", {"ids": ",".join(held[start : start + 100]), "tournamentId": tournament.get("id")})
        prices.update({str(row["exchangeId"]): row for row in body.get("data") or []})
    pm_quotes, ks_quotes = await asyncio.gather(polymarket.get_quotes(market_map.polymarket_tokens()), kalshi.get_quotes(market_map.kalshi_tickers()))
    return PnlInputs(market_map, positions, fills, prices, pm_quotes, ks_quotes, staleness_s=staleness_s)


def write_outputs(rows: list[PositionPnl], summary: dict[str, Any], csv_path: Path = CSV_PATH, json_path: Path = SUMMARY_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PositionPnl.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for row in rows:
        logger.info("Position P&L", extra={k: v for k, v in asdict(row).items() if v not in (None, "")})
    logger.info("Position P&L summary", extra={k: v for k, v in summary.items() if k not in ("worst_by_bid", "best_by_bid")})
    for item in summary["worst_by_bid"]:
        logger.info("Worst position by bid", extra=item)
    for item in summary["best_by_bid"]:
        logger.info("Best position by bid", extra=item)


async def _amain(args: argparse.Namespace) -> int:
    # Read-only on purpose: DRY_RUN forced on and the kill switch engaged before any request.
    settings = replace(load_settings(), dry_run=True)
    setup_logging(settings.log_level, settings.log_dir)
    market_map = MarketMap.load()
    async with ExchangeClient(settings) as client, PolymarketClient() as polymarket, KalshiClient() as kalshi:
        client.halt("position_pnl_report.py is read-only")
        reader = SigReader(client, OUT_DIR / "raw", max_reads_per_min=args.max_reads_per_min)
        inp = await gather_inputs(reader, polymarket, kalshi, market_map, settings.tournament_slug, settings.oracle_staleness_threshold_s)
    markout_lines = MARKOUT_JOURNAL_PATH.read_text(encoding="utf-8").splitlines() if MARKOUT_JOURNAL_PATH.exists() else []
    inp.fair_at_fill = fair_at_fill_from_markout(markout_lines)
    inp.order_actions = order_actions_from_journal(JOURNAL_PATH, inp.fills)
    inp.old_until, inp.new_since = unix_time(args.old_until), unix_time(args.new_since)
    rows, summary = build_report(inp)
    summary["notes"]["markout_journal_lines"] = len(markout_lines)
    write_outputs(rows, summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only unrealized P&L per position and SIG / oracle convergence.")
    parser.add_argument("--old-until", default=DEFAULT_OLD_UNTIL, help=f"first fill before this = old (default {DEFAULT_OLD_UNTIL})")
    parser.add_argument("--new-since", default=DEFAULT_NEW_SINCE, help=f"first fill at or after this = new (default {DEFAULT_NEW_SINCE})")
    parser.add_argument("--max-reads-per-min", type=float, default=DEFAULT_MAX_READS_PER_MIN, help=f"SIG read spacing (default {DEFAULT_MAX_READS_PER_MIN})")
    args = parser.parse_args(argv)
    if not 0 < args.max_reads_per_min <= 30:
        parser.error("--max-reads-per-min must be in (0, 30]: the live bot uses up to ~80 of the account's 100 reads/min")
    if not unix_time(args.old_until) or not unix_time(args.new_since):
        parser.error("--old-until / --new-since must be ISO times, e.g. 2026-10-05T13:41:00Z")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
