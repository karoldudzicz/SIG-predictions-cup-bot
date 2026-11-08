"""READ-ONLY simulation: a looser taker gross cap on top of the stress-based directional limit.

Nothing in risk.py / strategy.py / config.py changes; every limit below is overridden in the
simulation only.

1. Variants --gross-frac X (default 0.30 = baseline, 0.40, 0.50, 0.60), each with the
   stress-based directional limit of directional_stress_preview.py. X replaces the taker's
   30% caps - the portfolio cap in sizing (max_total_fraction, held market value) and the
   gross cap on orders (max_gross_exposure_fraction, hold-book cost) - and lifts the global
   cap (max_total_gross_exposure_fraction) to at least X. Per variant, against today's real
   limits: targets unlocked (orders the caps accept once the per-cycle pacing is ignored),
   their stake, expected profit (shares * (fair - margin - price), as in sizing), stress loss
   at s = 0.5 and 0.75 per direction after the change, gross after the change as % of
   bankroll, and what still blocks (first blocking limit, counted).
2. Every blocked target under today's limits -> data/edge_scan/blocked_targets.csv:
   exchange_id, race, party, side, YES bid / ask, buy price, fair (side), margin, net edge,
   target stake / shares (what Kelly wants with the sizing caps relaxed), expected profit,
   first blocking limit. Pipeline order: edge filter -> Kelly -> cluster cap -> directional
   cap -> portfolio cap (sizing) -> open order / held at target -> race / gross / per-cycle
   caps on the order alone -> the caps together with the cycle's other orders -> pacing.
3. --explain ID[,ID]: that pipeline step by step for these exchanges, logged (INFO).
4. Oracle-error scenario per variant (today's positions + the unlocked orders): per
   direction, the sum over races of the worst-case loss of that direction's legs in the race
   (D YES and R NO are pro-D; legs of the other direction in the same race are NOT netted -
   the conservative view of "our side was wrong everywhere"), and the loss if every held
   favourite (side price >= 0.5) of a direction loses at once (cost of those positions);
   both also as % of bankroll.

Reads only (the same reads as rebalance_preview.py): SIG through edge_scan.SigReader (GET
only, default 15 reads/min), DRY_RUN forced on and the kill switch engaged, plus the public
oracle read endpoints. Assumes the taker owns every position (ladder and quoter off).

    python scripts/gross_cap_preview.py [--gross-frac 0.30,0.40,0.50,0.60] [--explain 1000,936]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere
sys.path.insert(0, str(ROOT / "scripts"))

from config import load_settings  # noqa: E402
from directional_stress_preview import (  # noqa: E402
    ALT_STRESS_S,
    PositionRow,
    _key,
    adverse_wave,
    position_rows,
    positions_label,
    side_loss_per_share,
    targets_alternative,
    targets_today,
    wave_label,
)
from edge_scan import OUT_DIR, SigReader  # noqa: E402
from exchange_api import ExchangeClient  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from oracle_api import KalshiClient, PolymarketClient  # noqa: E402
from rebalance_preview import PreviewInputs, _Cycle, _cycle_view, gather_inputs  # noqa: E402
from risk import (  # noqa: E402
    RaceLeg,
    RiskLimits,
    SizedBet,
    check_buy,
    direction_of,
    max_shares_within_race_cap,
    race_legs_from_holdings,
    race_worst_case_loss,
)
from strategy import MIN_ORDER_SHARES, Holding, PlannedOrder, apply_exposure_caps, holdings_from_positions, plan_orders  # noqa: E402

logger = logging.getLogger("gross_cap_preview")

DEFAULT_MAX_READS_PER_MIN = 15  # the live bot shares the account's 100 reads/min
DEFAULT_GROSS_FRACS = (0.30, 0.40, 0.50, 0.60)
STRESS_SIZES = (0.5, 0.75)
FAVOURITE_PRICE = 0.5  # a position whose side is priced at or above this is a "favourite"
TOP_BLOCKED = 15
BLOCKED_CSV = OUT_DIR / "blocked_targets.csv"
VARIANTS_JSON = OUT_DIR / "gross_cap_variants.json"
_RELAXED = 1e9  # a cap switched off in the simulation
BLOCKED_FIELDS = (
    "exchange_id", "race", "party", "side", "yes_bid", "yes_ask", "buy_price", "fair_side", "margin", "edge",
    "target_stake", "target_shares", "expected_profit", "first_block",
)


# ---------------------------------------------------------------------- limits per variant


def variant_limits(base: RiskLimits, gross_frac: float) -> RiskLimits:
    """X replaces both 30% caps of the taker; the global cap is lifted to at least X."""
    return replace(
        base,
        max_total_fraction=gross_frac,
        max_gross_exposure_fraction=gross_frac,
        max_total_gross_exposure_fraction=max(base.max_total_gross_exposure_fraction, gross_frac),
    )


def _unpaced(limits: RiskLimits) -> RiskLimits:
    """The per-cycle notional cap only paces deployment: off for "what fits at all"."""
    return replace(limits, max_new_notional_per_cycle_fraction=_RELAXED)


# ---------------------------------------------------------------------- one blocked target


@dataclass
class BlockedTarget:
    exchange_id: str
    race: str
    party: str
    side: str
    yes_bid: float | None
    yes_ask: float | None
    buy_price: float
    fair_side: float
    margin: float
    edge: float  # net per share: fair_side - margin - buy_price
    target_stake: float  # what Kelly wants with the sizing caps relaxed (per-contract cap kept)
    target_shares: int
    expected_profit: float  # target_shares * edge
    first_block: str


def _targets(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], rows: list[PositionRow], mode: str, limits: RiskLimits) -> dict[str, SizedBet]:
    staged = replace(view, limits=limits)
    if mode == "stress":
        return targets_alternative(inp, staged, holdings, rows)[0]
    return targets_today(inp, staged, holdings)


def _sizing_stages(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], rows: list[PositionRow], mode: str, limits: RiskLimits) -> list[tuple[str, dict[str, SizedBet]]]:
    """Targets with the sizing caps switched on one by one, in size_portfolio's order."""
    relaxed = replace(limits, max_fraction_per_cluster=_RELAXED, max_directional_fraction=_RELAXED, max_total_fraction=_RELAXED)
    cluster = replace(relaxed, max_fraction_per_cluster=limits.max_fraction_per_cluster)
    directional = replace(cluster, max_directional_fraction=limits.max_directional_fraction)
    directional_name = "directional stress 4% (sizing)" if mode == "stress" else "directional 15% (sizing, market value)"
    return [
        ("kelly / exclusive-outcome sizing", _targets(inp, view, holdings, rows, "nominal", relaxed)),
        ("cluster 3% (sizing, held market value)", _targets(inp, view, holdings, rows, "nominal", cluster)),
        (directional_name, _targets(inp, view, holdings, rows, mode, directional)),
        (f"portfolio {limits.max_total_fraction:.0%} (sizing, held market value)", _targets(inp, view, holdings, rows, mode, limits)),
    ]


def _order_cap(order: PlannedOrder, inp: PreviewInputs, holdings: dict[str, Holding], limits: RiskLimits) -> str | None:
    """The first cap that leaves this order, on its own, below MIN_ORDER_SHARES (None = fits)."""
    mapping = inp.market_map.get(order.exchange_id)
    if mapping is None:
        return "not in market map"
    legs = race_legs_from_holdings(inp.market_map, holdings)[mapping.race.key]
    parties = [m.party for m in inp.market_map.by_race().get(mapping.race.key, [])]
    bankroll = inp.bankroll
    if max_shares_within_race_cap(parties, legs, mapping.party, order.side, order.price, order.quantity, limits.max_race_worst_loss_fraction * bankroll) < MIN_ORDER_SHARES:
        return f"race {limits.max_race_worst_loss_fraction:.0%} (order)"
    gross = sum(h.cost_basis for h in holdings.values())
    gross_cap = min(limits.max_gross_exposure_fraction, limits.max_total_gross_exposure_fraction) * bankroll
    if math.floor(max(0.0, gross_cap - gross) / order.price) < MIN_ORDER_SHARES:
        return f"gross {limits.max_gross_exposure_fraction:.0%} (order, hold-book cost)"
    if math.floor(limits.max_new_notional_per_cycle_fraction * bankroll / order.price) < MIN_ORDER_SHARES:
        return "per-cycle notional (order)"
    return None


def _accept(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], planned: list[PlannedOrder], limits: RiskLimits) -> list[PlannedOrder]:
    return apply_exposure_caps(planned, holdings, inp.market_map, view.fair, view.margins, inp.bankroll, limits, log=False)


def classify_targets(
    inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], rows: list[PositionRow], mode: str, limits: RiskLimits
) -> tuple[list[BlockedTarget], dict[str, list[dict[str, Any]]]]:
    """Every market with a side that passes the buy filter and still gets no (full) order,
    with the first limit that stops it; plus a step-by-step trace per exchange."""
    stages = _sizing_stages(inp, view, holdings, rows, mode, limits)
    full_targets = stages[-1][1]
    planned = {o.exchange_id: o for o in plan_orders(view.fair, full_targets, holdings, view.books, view.skip, inp.market_map, sell_above_target=False) if o.kind == "increase"}
    unpaced = _unpaced(limits)
    accepted_steady = {o.exchange_id for o in _accept(inp, view, holdings, list(planned.values()), unpaced)}
    accepted_cycle = {o.exchange_id for o in _accept(inp, view, holdings, list(planned.values()), limits)}

    blocked: list[BlockedTarget] = []
    traces: dict[str, list[dict[str, Any]]] = {}
    for eid, fair_yes in sorted(view.fair.items()):
        mapping = inp.market_map.get(eid)
        book = view.books[eid]
        margin = view.margins[eid]
        trace: list[dict[str, Any]] = []
        passing = []
        for side in ("yes", "no"):
            price = book.buy_price(side)
            fair_side = fair_yes if side == "yes" else 1.0 - fair_yes
            if price is None:
                trace.append({"step": "buy filter", "side": side, "result": "no ask on this side"})
                continue
            ok, edge, reason = check_buy(price, fair_side, margin, view.limits)
            trace.append({"step": "buy filter", "side": side, "price": price, "fair_side": round(fair_side, 4), "margin": round(margin, 4), "edge": round(edge, 4), "result": reason})
            if ok:
                passing.append((side, price, fair_side, edge))
        traces[eid] = trace
        if not passing:
            continue
        desired = stages[0][1].get(eid)
        side, price, fair_side, edge = next((p for p in passing if desired is None or p[0] == desired.side), passing[0])
        first_block = None
        for name, targets in stages:
            bet = targets.get(eid)
            trace.append({"step": name, "target_shares": bet.quantity if bet and bet.side == side else 0})
            if first_block is None and (bet is None or bet.side != side):
                first_block = name
        if first_block is None:
            order = planned.get(eid)
            if order is None:
                held = holdings.get(eid)
                first_block = "open order on this exchange" if eid in view.skip else ("held at / above target" if held and held.side == side else "below the rebalance band")
            else:
                first_block = _order_cap(order, inp, holdings, unpaced)
                if first_block is None and eid not in accepted_steady:
                    first_block = "race / gross with the cycle's other orders"
                elif first_block is None and eid not in accepted_cycle:
                    first_block = "per-cycle notional (paced, later cycles)"
        trace.append({"step": "result", "first_block": first_block})
        if first_block in (None, "held at / above target"):
            continue  # an order goes out, or the target is already held
        shares = desired.quantity if desired is not None and desired.side == side else 0
        blocked.append(BlockedTarget(
            eid, mapping.race.key if mapping else "", mapping.party if mapping else "", side, book.yes_bid, book.yes_ask,
            price, round(fair_side, 4), round(margin, 4), round(edge, 4), round(shares * price, 2), shares,
            round(shares * edge, 2), first_block,
        ))
    blocked.sort(key=lambda b: (-b.expected_profit, b.exchange_id))
    return blocked, traces


# ---------------------------------------------------------------------- variants


def _order_loss(order: PlannedOrder, inp: PreviewInputs, view: _Cycle, wave: str, s: float) -> float:
    mapping = inp.market_map.get(order.exchange_id)
    return 0.0 if mapping is None else order.quantity * side_loss_per_share(view.fair[order.exchange_id], mapping.party, order.side, wave, s)


def oracle_error(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], rows: list[PositionRow], orders: list[PlannedOrder]) -> dict[str, Any]:
    """Per direction: sum over races of the worst-case loss of that direction's legs (the other
    direction's legs in the same race not netted), and the loss if every favourite (side
    price >= FAVOURITE_PRICE) of that direction loses."""
    legs = race_legs_from_holdings(inp.market_map, holdings)
    for order in orders:
        mapping = inp.market_map.get(order.exchange_id)
        if mapping is not None:
            legs[mapping.race.key].append(RaceLeg(mapping.party, order.side, order.quantity, order.quantity * order.price))
    parties = {key: [m.party for m in group] for key, group in inp.market_map.by_race().items()}

    def leg_direction(leg: RaceLeg) -> str:
        if leg.party not in ("D", "R"):
            return "N"
        return "D" if (leg.party == "D") == (leg.side == "yes") else "R"

    race_loss = {"D": 0.0, "R": 0.0}
    for race, race_legs in legs.items():
        for direction in race_loss:
            mine = [leg for leg in race_legs if leg_direction(leg) == direction]
            if mine:
                race_loss[direction] += race_worst_case_loss(parties.get(race, []), mine)
    favourites = {"D": 0.0, "R": 0.0}
    for row in rows:
        side_price = None if row.p_yes is None else (row.p_yes if row.side == "yes" else 1.0 - row.p_yes)
        if row.direction in favourites and side_price is not None and side_price >= FAVOURITE_PRICE:
            favourites[row.direction] += holdings[row.exchange_id].cost_basis
    for order in orders:
        direction = direction_of(inp.market_map, order.exchange_id, order.side).value
        fair_side = view.fair[order.exchange_id] if order.side == "yes" else 1.0 - view.fair[order.exchange_id]
        if direction in favourites and fair_side >= FAVOURITE_PRICE:
            favourites[direction] += order.quantity * order.price
    bankroll = inp.bankroll or 1.0
    return {
        "race_worst_loss_by_direction": {d: round(v, 2) for d, v in race_loss.items()},
        "race_worst_loss_pct": {d: round(100 * v / bankroll, 2) for d, v in race_loss.items()},
        "all_favourites_lose": {d: round(v, 2) for d, v in favourites.items()},
        "all_favourites_lose_pct": {d: round(100 * v / bankroll, 2) for d, v in favourites.items()},
    }


def run_variant(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], rows: list[PositionRow], gross_frac: float, today_accepted: set[str]) -> dict[str, Any]:
    limits = variant_limits(view.limits, gross_frac)
    targets = _targets(inp, view, holdings, rows, "stress", limits)
    planned = [o for o in plan_orders(view.fair, targets, holdings, view.books, view.skip, inp.market_map, sell_above_target=False) if o.kind == "increase"]
    accepted = _accept(inp, view, holdings, planned, _unpaced(limits))
    unlocked = [o for o in accepted if o.exchange_id not in today_accepted]
    bankroll = inp.bankroll or 1.0
    stress = {}
    for direction in ("D", "R"):
        wave = adverse_wave(direction)
        for s in STRESS_SIZES:
            held = sum(r.losses.get(_key(wave, s), 0.0) for r in rows if r.direction == direction)
            new = sum(_order_loss(o, inp, view, wave, s) for o in accepted if direction_of(inp.market_map, o.exchange_id, o.side).value == direction)
            # Key "D@0.5" here = D-DIRECTIONAL positions in their adverse wave (wave R), not wave D.
            stress[f"{direction}@{s}"] = {
                "label": f"{positions_label(direction)} in {wave_label(wave)}, s={s}",
                "loss": round(held + new, 2), "pct_bankroll": round(100 * (held + new) / bankroll, 2),
            }
    blocked, _ = classify_targets(inp, view, holdings, rows, "stress", limits)
    counts: dict[str, int] = {}
    for item in blocked:
        counts[item.first_block] = counts.get(item.first_block, 0) + 1
    gross_after = sum(h.cost_basis for h in holdings.values()) + sum(o.quantity * o.price for o in accepted)
    return {
        "gross_frac": gross_frac,
        "unlocked": len(unlocked),
        "unlocked_stake": round(sum(o.quantity * o.price for o in unlocked), 2),
        "unlocked_expected_profit": round(sum(o.quantity * (o.edge or 0.0) for o in unlocked), 2),
        "stress_after": stress,
        "gross_after_pct": round(100 * gross_after / bankroll, 2),
        "still_blocked_by": counts,
        "oracle_error": oracle_error(inp, view, holdings, rows, accepted),
        "unlocked_orders": [
            {"exchange_id": o.exchange_id, "race": o.race_key, "side": o.side, "quantity": o.quantity, "price": o.price, "edge": None if o.edge is None else round(o.edge, 4)}
            for o in sorted(unlocked, key=lambda o: -(o.quantity * (o.edge or 0.0)))
        ],
    }


def build_gross_preview(inp: PreviewInputs, gross_fracs: tuple[float, ...] = DEFAULT_GROSS_FRACS) -> tuple[list[BlockedTarget], dict[str, Any], dict[str, list]]:
    holdings = holdings_from_positions(inp.positions)
    view = _cycle_view(inp)
    rows = position_rows(inp, view, holdings)
    # Today: the real limits (nominal 15% directional, 30% caps), pacing ignored.
    today_targets = targets_today(inp, view, holdings)
    today_planned = [o for o in plan_orders(view.fair, today_targets, holdings, view.books, view.skip, inp.market_map, sell_above_target=False) if o.kind == "increase"]
    today_accepted = {o.exchange_id for o in _accept(inp, view, holdings, today_planned, _unpaced(view.limits))}
    blocked, traces = classify_targets(inp, view, holdings, rows, "nominal", view.limits)
    variants = [run_variant(inp, view, holdings, rows, frac, today_accepted) for frac in gross_fracs]
    summary = {
        "bankroll": round(inp.bankroll, 2),
        "today": {
            "orders_that_fit": len(today_accepted),
            "blocked_targets": len(blocked),
            "oracle_error": oracle_error(inp, view, holdings, rows, [o for o in today_planned if o.exchange_id in today_accepted]),
        },
        "variants": variants,
        "top_blocked_by_expected_profit": [asdict(b) for b in blocked[:TOP_BLOCKED]],
    }
    return blocked, summary, traces


# ---------------------------------------------------------------------- outputs and CLI


def write_outputs(blocked: list[BlockedTarget], summary: dict[str, Any], traces: dict[str, list], explain: tuple[str, ...] = (),
                  csv_path: Path = BLOCKED_CSV, json_path: Path = VARIANTS_JSON) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=BLOCKED_FIELDS)
        writer.writeheader()
        for item in blocked:
            writer.writerow(asdict(item))
    json_path.write_text(json.dumps({**summary, "explain": {eid: traces.get(eid) for eid in explain}}, indent=2, default=str), encoding="utf-8")
    for variant in summary["variants"]:
        logger.info("Gross cap variant", extra={k: v for k, v in variant.items() if k != "unlocked_orders"})
    for item in summary["top_blocked_by_expected_profit"]:
        logger.info("Blocked target", extra=item)
    for eid in explain:
        logger.info("Explain target", extra={"exchange_id": eid, "trace": traces.get(eid, [{"step": "no fresh fair or SIG price this run"}])})
    logger.info("Gross cap preview done", extra={"today": summary["today"], "blocked_targets": len(blocked)})


async def _amain(args: argparse.Namespace) -> int:
    # Read-only on purpose: DRY_RUN forced on and the kill switch engaged before any request.
    settings = replace(load_settings(), dry_run=True)
    setup_logging(settings.log_level, settings.log_dir)
    market_map = MarketMap.load()
    async with ExchangeClient(settings) as client, PolymarketClient() as polymarket, KalshiClient() as kalshi:
        client.halt("gross_cap_preview.py is read-only")
        reader = SigReader(client, OUT_DIR / "raw", max_reads_per_min=args.max_reads_per_min)
        inp = await gather_inputs(reader, polymarket, kalshi, market_map, settings.tournament_slug, settings.oracle_staleness_threshold_s, with_depth=False)
        blocked, summary, traces = build_gross_preview(inp, args.gross_frac)
        write_outputs(blocked, summary, traces, args.explain)
    return 0


def _fractions(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values or not all(0.05 <= v <= 1.0 for v in values):
        raise argparse.ArgumentTypeError("gross fractions must be in [0.05, 1.0]")
    return values


def _ids(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not all(v.isdigit() for v in values):
        raise argparse.ArgumentTypeError("exchange ids must be numeric")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: looser taker gross cap + stress directional limit, simulated.")
    parser.add_argument("--max-reads-per-min", type=float, default=DEFAULT_MAX_READS_PER_MIN, help=f"SIG read spacing (default {DEFAULT_MAX_READS_PER_MIN})")
    parser.add_argument("--gross-frac", type=_fractions, default=DEFAULT_GROSS_FRACS, help="comma-separated variants (default 0.30,0.40,0.50,0.60)")
    parser.add_argument("--explain", type=_ids, default=(), help="comma-separated exchange ids to trace step by step")
    args = parser.parse_args(argv)
    if not 0 < args.max_reads_per_min <= 30:
        parser.error("--max-reads-per-min must be in (0, 30]: the live bot uses up to ~80 of the account's 100 reads/min")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
