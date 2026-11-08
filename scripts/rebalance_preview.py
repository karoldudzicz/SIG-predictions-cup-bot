"""READ-ONLY: what would selling positions down to their targets free, and at what cost?

Rebuilds the taker's view of one cycle without trading: positions, open orders, bankroll,
tournament standing, SIG top of book (bulk /exchanges/prices with the tournament id), the
oracle fair and relationships -> the same Kelly targets as strategy.py (size_portfolio
with every held position counted). For every position holding more shares than its
target (no target on that side = target 0) it reports, per row, to the log and to
data/edge_scan/rebalance_preview.csv:

    excess shares, best bid / ask, fair, uncertainty u, the sell price at the bid,
    sell cost vs fair = (fair - bid) * excess   (NO positions in NO terms),
    freed gross = cost basis of the excess, freed race worst case, freed directional room

and a summary: (a) rows ranked by sell cost per freed gross dollar, (b) totals freed by
selling every position with fair - bid <= 0.01 and <= 0.02, (c) how many targets that
the caps block today (allowed = 0) would get room after such a sale - re-sized and
re-capped with the reduced holdings: room on their own, and accepted within one cycle; also
targets that the sizing caps cut to nothing today and that come back after the sale -
(d) positions that cannot be sold sensibly (no bid, spread wider than UNSELLABLE_SPREAD,
or no fair / no SIG price).

Reads only: SIG through edge_scan.SigReader (GET only, spaced by --max-reads-per-min,
default 15, 429 -> pause), the client created with DRY_RUN forced on and its kill switch
engaged, plus the public Polymarket / Kalshi read endpoints. Nothing is sent but GETs.
Assumes the taker owns every position (ladder and quoter off).

    python scripts/rebalance_preview.py [--max-reads-per-min 15] [--with-depth]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from config import load_settings  # noqa: E402
from edge_scan import OUT_DIR, SigReader  # noqa: E402
from exchange_api import ExchangeClient  # noqa: E402
from ladder import uncertainty_band  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from oracle_api import KalshiClient, OracleQuote, PolymarketClient  # noqa: E402
from risk import (  # noqa: E402
    Direction,
    Opportunity,
    RaceLeg,
    RiskLimits,
    build_clusters,
    direction_of,
    leaderboard_standing,
    race_legs_from_holdings,
    race_worst_case_loss,
    size_portfolio,
    tournament_kelly_fraction,
)
from strategy import Holding, SigBook, apply_exposure_caps, compute_fair_price, holdings_from_positions, plan_orders  # noqa: E402
from strategy import MIN_ORDER_SHARES, REBALANCE_BAND  # noqa: E402

logger = logging.getLogger("rebalance_preview")

DEFAULT_MAX_READS_PER_MIN = 15  # the live bot shares the account's 100 reads/min
UNSELLABLE_SPREAD = 0.05  # YES ask - bid wider than this: no sensible exit at the bid
COST_BANDS = (0.01, 0.02)  # (b): fair - bid thresholds
PAGE_LIMIT = 200
CSV_PATH = OUT_DIR / "rebalance_preview.csv"
CSV_FIELDS = (
    "exchange_id", "race", "party", "side", "held_shares", "target_shares", "excess_shares", "note",
    "yes_bid", "yes_ask", "fair_yes", "fair_side", "u", "sell_price", "fair_minus_bid", "sell_cost_vs_fair",
    "freed_gross", "freed_race_worst_case", "direction", "freed_directional", "cost_per_freed_dollar",
    "sellable", "unsellable_reason", "sold_today_with_flag", "bid_depth",
)


@dataclass
class PreviewInputs:
    market_map: MarketMap
    positions: list[dict[str, Any]]
    open_orders: list[dict[str, Any]]
    bankroll: float
    prices: Mapping[str, Mapping[str, Any]]  # exchange_id -> bulk row (bestBid / bestAsk)
    pm_quotes: Mapping[str, OracleQuote]
    ks_quotes: Mapping[str, OracleQuote]
    relationships: list[dict[str, Any]] = field(default_factory=list)
    standing: float | None = None
    time_remaining: float = 1.0
    limits: RiskLimits = field(default_factory=RiskLimits)
    staleness_s: float = 10.0
    bid_depth: Mapping[str, float] = field(default_factory=dict)  # --with-depth: shares at the best bid (side terms)


@dataclass
class PreviewRow:
    exchange_id: str
    race: str
    party: str
    side: str
    held_shares: int
    target_shares: int | None  # None: no fair / SIG price, target unknown
    excess_shares: int | None
    note: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    fair_yes: float | None = None
    fair_side: float | None = None
    u: float | None = None
    sell_price: float | None = None  # best bid in the held side's terms
    fair_minus_bid: float | None = None
    sell_cost_vs_fair: float | None = None
    freed_gross: float | None = None
    freed_race_worst_case: float | None = None
    direction: str = ""
    freed_directional: float | None = None
    cost_per_freed_dollar: float | None = None
    sellable: bool = False
    unsellable_reason: str = ""
    sold_today_with_flag: bool = False  # SELL_ABOVE_TARGET=True would sell this (target on this side, past the band)
    bid_depth: float | None = None


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _scaled(holding: Holding, keep: int) -> Holding | None:
    if keep <= 0:
        return None
    scale = keep / holding.shares
    return Holding(holding.side, keep, holding.market_value * scale, holding.cost_basis * scale)


@dataclass
class _Cycle:
    """The taker's view of one cycle, computed the way strategy.py computes it."""

    fair: dict[str, float]
    margins: dict[str, float]
    u: dict[str, float]
    books: dict[str, SigBook]
    limits: RiskLimits
    clusters: list
    skip: set[str]


def _cycle_view(inp: PreviewInputs) -> _Cycle:
    books = {eid: SigBook(row.get("bestBid"), row.get("bestAsk")) for eid, row in inp.prices.items()}
    fair, margins, u = {}, {}, {}
    for mapping in inp.market_map:
        eid = mapping.sig_exchange_id
        pm = inp.pm_quotes.get(mapping.polymarket_token) if mapping.polymarket_token else None
        ks = inp.ks_quotes.get(mapping.kalshi_ticker) if mapping.kalshi_ticker else None
        price, _ = compute_fair_price(pm, ks, inp.staleness_s)
        if price is None or eid not in books:
            continue
        fair[eid], margins[eid], u[eid] = price.prob, price.margin, uncertainty_band(pm, ks, inp.staleness_s)
    limits = replace(inp.limits, kelly_fraction=tournament_kelly_fraction(inp.limits.kelly_fraction, inp.standing, inp.time_remaining))
    skip = {str(o.get("exchangeId")) for o in inp.open_orders} & set(fair)
    return _Cycle(fair, margins, u, books, limits, build_clusters(inp.market_map, inp.relationships), skip)


def _targets(view: _Cycle, inp: PreviewInputs, holdings: dict[str, Holding]) -> dict[str, Any]:
    opportunities = {
        eid: Opportunity(eid, fair_prob=p, yes_price=view.books[eid].buy_price("yes"), no_price=view.books[eid].buy_price("no"), margin=view.margins[eid])
        for eid, p in view.fair.items()
    }
    held = {eid: (h.side, h.market_value) for eid, h in holdings.items()}
    return {bet.exchange_id: bet for bet in size_portfolio(view.clusters, opportunities, inp.bankroll, inp.market_map, view.limits, held)}


def _increases(view: _Cycle, inp: PreviewInputs, holdings: dict[str, Holding]) -> tuple[set[str], set[str], set[str]]:
    """(exchanges with a planned increase, those with room under the caps on their own,
    those accepted within one cycle) - SELL_ABOVE_TARGET off, as today."""
    targets = _targets(view, inp, holdings)
    planned = [o for o in plan_orders(view.fair, targets, holdings, view.books, view.skip, inp.market_map, sell_above_target=False) if o.kind == "increase"]
    caps = dict(market_map=inp.market_map, fair=view.fair, margins=view.margins, bankroll=inp.bankroll, limits=view.limits, log=False)
    alone = {o.exchange_id for o in planned if apply_exposure_caps([o], holdings, **caps)}
    accepted = {o.exchange_id for o in apply_exposure_caps(planned, holdings, **caps)}
    return {o.exchange_id for o in planned}, alone, accepted


def _after_sale(holdings: dict[str, Holding], rows: Iterable[PreviewRow]) -> dict[str, Holding]:
    after = dict(holdings)
    for row in rows:
        kept = _scaled(holdings[row.exchange_id], row.held_shares - row.excess_shares)
        if kept is None:
            after.pop(row.exchange_id, None)
        else:
            after[row.exchange_id] = kept
    return after


def _freed(inp: PreviewInputs, holdings: dict[str, Holding], rows: list[PreviewRow]) -> dict[str, Any]:
    """Totals for selling every row in `rows` at the bid: per race worst case computed on
    the whole set (legs in one race are not additive)."""
    race_parties = {key: [m.party for m in group] for key, group in inp.market_map.by_race().items()}
    legs = race_legs_from_holdings(inp.market_map, holdings)
    freed_race = 0.0
    for race in {row.race for row in rows}:
        before = race_worst_case_loss(race_parties[race], legs[race])
        sells = [RaceLeg(row.party, row.side, -row.excess_shares, -row.excess_shares * row.sell_price) for row in rows if row.race == race]
        freed_race += before - race_worst_case_loss(race_parties[race], legs[race] + sells)
    directional: dict[str, float] = {}
    for row in rows:
        directional[row.direction] = round(directional.get(row.direction, 0.0) + (row.freed_directional or 0.0), 2)
    return {
        "positions": len(rows),
        "excess_shares": sum(row.excess_shares for row in rows),
        "sell_cost_vs_fair": round(sum(row.sell_cost_vs_fair or 0.0 for row in rows), 2),
        "freed_gross": round(sum(row.freed_gross or 0.0 for row in rows), 2),
        "freed_race_worst_case": round(freed_race, 2),
        "freed_directional": directional,
    }


def build_preview(inp: PreviewInputs) -> tuple[list[PreviewRow], dict[str, Any]]:
    holdings = holdings_from_positions(inp.positions)
    view = _cycle_view(inp)
    targets = _targets(view, inp, holdings)
    race_parties = {key: [m.party for m in group] for key, group in inp.market_map.by_race().items()}
    legs = race_legs_from_holdings(inp.market_map, holdings)

    rows: list[PreviewRow] = []
    for eid, held in sorted(holdings.items()):
        mapping = inp.market_map.get(eid)
        race, party = (mapping.race.key, mapping.party) if mapping else ("", "")
        if eid not in view.fair:
            reason = "not in market map" if mapping is None else ("no SIG price" if eid not in view.books else "no fresh fair")
            rows.append(PreviewRow(eid, race, party, held.side, held.shares, None, None, reason, unsellable_reason=reason))
            continue
        target = targets.get(eid)
        if target is not None and target.side == held.side:
            target_shares, note = target.quantity, "above target"
        elif target is not None:
            target_shares, note = 0, "flip: sold today regardless of SELL_ABOVE_TARGET"
        else:
            target_shares, note = 0, "no target: sold today only at a bid >= fair"
        excess = held.shares - target_shares
        if excess <= 0:
            continue
        book = view.books[eid]
        fair_yes = view.fair[eid]
        fair_side = fair_yes if held.side == "yes" else 1.0 - fair_yes
        bid = book.sell_price(held.side)
        row = PreviewRow(
            eid, race, party, held.side, held.shares, target_shares, excess, note,
            yes_bid=book.yes_bid, yes_ask=book.yes_ask, fair_yes=_r(fair_yes), fair_side=_r(fair_side), u=_r(view.u[eid]),
            sell_price=bid, freed_gross=_r(held.cost_basis * excess / held.shares, 2),
            direction=direction_of(inp.market_map, eid, held.side).value,
            freed_directional=_r(held.market_value * excess / held.shares, 2),
            sold_today_with_flag=target is not None and target.side == held.side and excess >= max(MIN_ORDER_SHARES, REBALANCE_BAND * target.quantity),
            bid_depth=inp.bid_depth.get(eid),
        )
        if bid is not None:
            row.fair_minus_bid = _r(fair_side - bid)
            row.sell_cost_vs_fair = _r((fair_side - bid) * excess, 2)
            if mapping is not None:
                before = race_worst_case_loss(race_parties[race], legs[race])
                after = race_worst_case_loss(race_parties[race], legs[race] + [RaceLeg(party, held.side, -excess, -excess * bid)])
                row.freed_race_worst_case = _r(before - after, 2)
            if row.freed_gross:
                row.cost_per_freed_dollar = _r(row.sell_cost_vs_fair / row.freed_gross)
        spread = None if book.yes_bid is None or book.yes_ask is None else book.yes_ask - book.yes_bid
        if bid is None:
            row.unsellable_reason = "no bid"
        elif spread is None:
            row.unsellable_reason = "one-sided book"
        elif spread > UNSELLABLE_SPREAD + 1e-9:
            row.unsellable_reason = f"spread {spread:.3f} > {UNSELLABLE_SPREAD}"
        row.sellable = not row.unsellable_reason
        rows.append(row)

    measured = [row for row in rows if row.excess_shares is not None]
    sellable = [row for row in measured if row.sellable]
    ranked = sorted(sellable, key=lambda row: (row.cost_per_freed_dollar is None, row.cost_per_freed_dollar or 0.0))

    # (c) blocked targets: an increase planned today with no room under the caps on its own.
    planned_before, alone_before, _ = _increases(view, inp, holdings)
    blocked = planned_before - alone_before
    scenarios: dict[str, Any] = {}
    sets = {f"fair_minus_bid<={band}": [r for r in sellable if r.fair_minus_bid is not None and r.fair_minus_bid <= band + 1e-9] for band in COST_BANDS}
    sets["all_sellable"] = sellable
    for name, chosen in sets.items():
        planned_after, alone_after, accepted_after = _increases(view, inp, _after_sale(holdings, chosen))
        scenarios[name] = {
            **_freed(inp, holdings, chosen),
            "blocked_targets_with_room": len(blocked & alone_after),
            "blocked_targets_accepted_in_one_cycle": len(blocked & accepted_after),
            # Targets the sizing caps (cluster / directional / portfolio, held positions
            # counted) cut to nothing today, that come back once the excess is sold.
            "new_targets_with_room": len((planned_after - planned_before) & alone_after),
        }

    summary = {
        "positions": len(holdings),
        "above_target": len(measured),
        "sellable": len(sellable),
        "ranked_by_cost_per_freed_dollar": [(r.exchange_id, r.cost_per_freed_dollar) for r in ranked],
        "blocked_targets_today": len(blocked),
        "scenarios": scenarios,
        "unsellable": [(r.exchange_id, r.unsellable_reason) for r in rows if not r.sellable],
    }
    return ranked + [r for r in rows if r not in ranked], summary


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


def _time_remaining(tournament: Mapping[str, Any]) -> float:
    start, end = tournament.get("startDate"), tournament.get("endDate")
    if not start or not end:
        return 1.0
    start_dt, end_dt = (datetime.fromisoformat(x.replace("Z", "+00:00")) for x in (start, end))
    total = (end_dt - start_dt).total_seconds()
    left = (end_dt - datetime.now(timezone.utc)).total_seconds()
    return min(max(left / total, 0.0), 1.0) if total > 0 else 1.0


async def gather_inputs(reader: SigReader, polymarket: Any, kalshi: Any, market_map: MarketMap, slug: str, staleness_s: float, with_depth: bool) -> PreviewInputs:
    def get(path: str, params: dict[str, Any] | None = None) -> Any:
        return reader.get(path, params or {}, cache_key="none", cacheable=lambda b: False)

    tournament = await get(f"/tournaments/{slug}")
    bankroll = float((await get("/portfolio/pnl", {"period": "all"})).get("totalAccountValue") or 0.0)
    positions = (await get("/portfolio/positions")).get("positions") or []
    open_orders = await _pages(reader, "/orders", {"status": "open"})
    board = await get(f"/tournaments/{slug}/leaderboard", {"period": "all", "limit": 1})
    relationships = await _pages(reader, "/relationships", {})
    ids = [m.sig_exchange_id for m in market_map]
    prices: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 100):
        body = await get("/exchanges/prices", {"ids": ",".join(ids[start : start + 100]), "tournamentId": tournament.get("id")})
        prices.update({str(row["exchangeId"]): row for row in body.get("data") or []})
    pm_quotes, ks_quotes = await asyncio.gather(polymarket.get_quotes(market_map.polymarket_tokens()), kalshi.get_quotes(market_map.kalshi_tickers()))
    inp = PreviewInputs(
        market_map, positions, open_orders, bankroll, prices, pm_quotes, ks_quotes, relationships,
        leaderboard_standing(board.get("myRank"), board.get("total")), _time_remaining(tournament), staleness_s=staleness_s,
    )
    if with_depth:  # one order book per position above target: shares resting at the best bid
        rows, _ = build_preview(inp)
        depth: dict[str, float] = {}
        for row in rows:
            if row.excess_shares and row.sell_price is not None:
                book = await get(f"/exchanges/{row.exchange_id}/orderbook", {"depth": 1})
                levels = book.get("bids" if row.side == "yes" else "asks") or []
                depth[row.exchange_id] = float(levels[0]["quantity"]) if levels else 0.0
        inp.bid_depth = depth
    return inp


def write_outputs(rows: list[PreviewRow], summary: dict[str, Any], csv_path: Path = CSV_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    for row in rows:
        logger.info("Rebalance candidate", extra={k: v for k, v in asdict(row).items() if v not in (None, "")})
    for name, scenario in summary["scenarios"].items():
        logger.info("Rebalance scenario", extra={"scenario": name, **scenario})
    logger.info(
        "Rebalance preview done",
        extra={k: summary[k] for k in ("positions", "above_target", "sellable", "blocked_targets_today", "ranked_by_cost_per_freed_dollar", "unsellable")},
    )
    (csv_path.with_suffix(".json")).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


async def _amain(args: argparse.Namespace) -> int:
    # Read-only on purpose: DRY_RUN forced on and the kill switch engaged before any request.
    settings = replace(load_settings(), dry_run=True)
    setup_logging(settings.log_level, settings.log_dir)
    market_map = MarketMap.load()
    async with ExchangeClient(settings) as client, PolymarketClient() as polymarket, KalshiClient() as kalshi:
        client.halt("rebalance_preview.py is read-only")
        reader = SigReader(client, OUT_DIR / "raw", max_reads_per_min=args.max_reads_per_min)
        inp = await gather_inputs(reader, polymarket, kalshi, market_map, settings.tournament_slug, settings.oracle_staleness_threshold_s, args.with_depth)
        rows, summary = build_preview(inp)
        write_outputs(rows, summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only preview of selling positions down to their targets.")
    parser.add_argument("--max-reads-per-min", type=float, default=DEFAULT_MAX_READS_PER_MIN, help=f"SIG read spacing (default {DEFAULT_MAX_READS_PER_MIN})")
    parser.add_argument("--with-depth", action="store_true", help="also read one order book per position above target (best-bid size)")
    args = parser.parse_args(argv)
    if not 0 < args.max_reads_per_min <= 30:
        parser.error("--max-reads-per-min must be in (0, 30]: the live bot uses up to ~80 of the account's 100 reads/min")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
