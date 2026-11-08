"""READ-ONLY simulation: a stress-loss directional limit instead of 15% nominal - what would it free?

1. Today's positions: direction (D / R / N, risk.direction_of), market value, and the
   directional exposure and room as the taker counts them: held market value per
   direction against max_directional_fraction (15%) of the bankroll.
2. National-wave stress: for a D-wave the logit of every D party's win probability moves
   +s and every R party's -s (an R-wave the other way; others unchanged), for s in
   WAVE_SIZES: p -> logistic(logit(p) +/- s), positions priced at the oracle fair (or their
   mark when there is no fresh fair). Loss = shares * max(0, price_now - price_after) on
   the side we hold. Totals per wave and s (also as % of bankroll), the 10 largest
   contributions, and senate:RI and house:NH-01 separately.
3. Alternative: the directional limit as "stress loss of that direction at s = ALT_STRESS_S
   in its adverse wave <= ALT_STRESS_FRAC of the bankroll". Free stress budget, turned into
   nominal with the direction's own loss-per-dollar, vs. today's free nominal room.
4. Simulation (nothing is sent; risk.py / strategy.py unchanged): size_portfolio with the
   nominal directional cap switched off, then the stress cap applied per direction the way
   size_portfolio applies its own (every bet of the direction scaled by one factor), then
   plan_orders and apply_exposure_caps exactly as the taker runs them. Reported: targets
   that get room only under the alternative, their stake, the expected profit
   (shares * shrunk edge = fair_side - margin - price, as in sizing), the stress they add,
   and how many new targets the race (2%) / gross (30%) / per-cycle caps still block.
5. Summary: does the alternative free real capital for markets with a clear edge, and the
   expected profit per SUSQie of added stress loss.

Reads only (the same reads as rebalance_preview.py): SIG through edge_scan.SigReader (GET
only, default 15 reads/min), DRY_RUN forced on and the kill switch engaged, plus the public
oracle read endpoints. Assumes the taker owns every position (ladder and quoter off).

    python scripts/directional_stress_preview.py [--max-reads-per-min 15]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere
sys.path.insert(0, str(ROOT / "scripts"))

from config import load_settings  # noqa: E402
from edge_scan import OUT_DIR, SigReader  # noqa: E402
from exchange_api import ExchangeClient  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from oracle_api import KalshiClient, PolymarketClient  # noqa: E402
from rebalance_preview import PreviewInputs, _Cycle, _cycle_view, gather_inputs  # noqa: E402
from risk import (  # noqa: E402
    Direction,
    Opportunity,
    SizedBet,
    adverse_wave,
    directional_exposure,
    direction_of,
    logistic,
    logit,
    max_shares_within_race_cap,
    race_legs_from_holdings,
    side_loss_per_share,
    size_portfolio,
    stressed_yes,
)
from strategy import MIN_ORDER_SHARES, Holding, PlannedOrder, apply_exposure_caps, holdings_from_positions, plan_orders  # noqa: E402

logger = logging.getLogger("directional_stress_preview")

DEFAULT_MAX_READS_PER_MIN = 15  # the live bot shares the account's 100 reads/min
WAVE_SIZES = (0.25, 0.5, 0.75)
WAVES = ("D", "R")
ALT_STRESS_S = 0.5
ALT_STRESS_FRAC = 0.04
FOCUS_RACES = ("senate:RI", "house:NH-01")
TOP_CONTRIBUTIONS = 10
CLEAR_EDGE = 0.03  # per share, after shrink and margin: what (5) calls "a clear edge"
POSITIONS_CSV = OUT_DIR / "directional_stress_positions.csv"
SUMMARY_JSON = OUT_DIR / "directional_stress_summary.json"
_P_CLAMP = 1e-6


# ---------------------------------------------------------------------- stress maths
# logit / logistic / stressed_yes / side_loss_per_share / adverse_wave live in risk.py and
# are shared with the bot's "stress" directional limit (imported above).


def wave_label(wave: str) -> str:
    return f"wave {wave} ({wave} wins)"


def positions_label(direction: str) -> str:
    return f"positions {direction}-directional"


# ---------------------------------------------------------------------- positions


@dataclass
class PositionRow:
    exchange_id: str
    race: str
    party: str
    side: str
    shares: int
    market_value: float
    direction: str
    p_yes: float | None  # oracle fair, or the mark (market value per share) when no fresh fair
    price_source: str  # "fair" | "mark" | "none"
    losses: dict[str, float] = field(default_factory=dict)  # "D@0.5" -> stress loss


def _key(wave: str, s: float) -> str:
    return f"{wave}@{s}"


def position_rows(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding]) -> list[PositionRow]:
    rows = []
    for eid, held in sorted(holdings.items()):
        mapping = inp.market_map.get(eid)
        race, party = (mapping.race.key, mapping.party) if mapping else ("", "")
        if eid in view.fair:
            p_yes, source = view.fair[eid], "fair"
        elif held.shares:
            mark_side = held.market_value / held.shares
            p_yes, source = (mark_side if held.side == "yes" else 1.0 - mark_side), "mark"
        else:
            p_yes, source = None, "none"
        row = PositionRow(eid, race, party, held.side, held.shares, round(held.market_value, 2),
                          direction_of(inp.market_map, eid, held.side).value, None if p_yes is None else round(p_yes, 4), source)
        for wave in WAVES:
            for s in WAVE_SIZES:
                row.losses[_key(wave, s)] = 0.0 if p_yes is None else round(held.shares * side_loss_per_share(p_yes, party, held.side, wave, s), 2)
        rows.append(row)
    return rows


def directional_room(inp: PreviewInputs, holdings: dict[str, Holding]) -> dict[str, dict[str, float]]:
    """Today, as the taker counts it: held market value per direction vs 15% of bankroll."""
    used = directional_exposure(inp.market_map, ((eid, h.side, h.market_value) for eid, h in holdings.items()))
    cap = inp.limits.max_directional_fraction * inp.bankroll
    return {
        d.value: {"exposure": round(used.get(d, 0.0), 2), "cap": round(cap, 2), "room": round(cap - used.get(d, 0.0), 2)}
        for d in (Direction.PRO_DEMOCRAT, Direction.PRO_REPUBLICAN)
    }


def direction_stress(rows: list[PositionRow], direction: str, s: float = ALT_STRESS_S) -> float:
    """Stress loss of one direction's positions in its adverse wave."""
    wave = adverse_wave(direction)
    return sum(r.losses.get(_key(wave, s), 0.0) for r in rows if r.direction == direction) if wave else 0.0


# ---------------------------------------------------------------------- alternative sizing


def _bet_stress(bet: SizedBet, inp: PreviewInputs, view: _Cycle) -> float:
    mapping = inp.market_map.get(bet.exchange_id)
    wave = adverse_wave(bet.direction.value)
    if mapping is None or wave is None:
        return 0.0
    return bet.quantity * side_loss_per_share(view.fair[bet.exchange_id], mapping.party, bet.side, wave, ALT_STRESS_S)


def _order_stress(order: PlannedOrder, inp: PreviewInputs, view: _Cycle) -> float:
    mapping = inp.market_map.get(order.exchange_id)
    wave = adverse_wave(direction_of(inp.market_map, order.exchange_id, order.side).value)
    if mapping is None or wave is None:
        return 0.0
    return order.quantity * side_loss_per_share(view.fair[order.exchange_id], mapping.party, order.side, wave, ALT_STRESS_S)


def _opportunities(view: _Cycle) -> dict[str, Opportunity]:
    return {
        eid: Opportunity(eid, fair_prob=p, yes_price=view.books[eid].buy_price("yes"), no_price=view.books[eid].buy_price("no"), margin=view.margins[eid])
        for eid, p in view.fair.items()
    }


def targets_today(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding]) -> dict[str, SizedBet]:
    held = {eid: (h.side, h.market_value) for eid, h in holdings.items()}
    return {b.exchange_id: b for b in size_portfolio(view.clusters, _opportunities(view), inp.bankroll, inp.market_map, view.limits, held)}


def targets_alternative(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], rows: list[PositionRow] | None = None) -> tuple[dict[str, SizedBet], dict[str, Any]]:
    """The bot's own sizing in DIRECTIONAL_LIMIT_MODE=stress (risk.size_portfolio ->
    risk.stress_directional_cap) at ALT_STRESS_S / ALT_STRESS_FRAC: cluster cap, portfolio
    cap, then per direction the adverse-wave stress of the held positions no target
    replaces + the direction's targets kept within ALT_STRESS_FRAC of the bankroll (every
    bet of the direction scaled by one factor). `rows` is not needed any more (kept for
    callers)."""
    held = {eid: (h.side, h.market_value) for eid, h in holdings.items()}
    held_shares = {eid: h.shares for eid, h in holdings.items()}
    stress_limits = replace(view.limits, directional_mode="stress", directional_stress_s=ALT_STRESS_S, directional_stress_frac=ALT_STRESS_FRAC)
    info: dict[str, Any] = {}
    sized = size_portfolio(view.clusters, _opportunities(view), inp.bankroll, inp.market_map, stress_limits, held, held_shares, report=info)
    return {b.exchange_id: b for b in sized}, info


def _increases(inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding], targets: dict[str, SizedBet]) -> tuple[list[PlannedOrder], dict[str, PlannedOrder], set[str]]:
    """(planned increases, eid -> the order capped on its own, eids accepted in one cycle)."""
    planned = [o for o in plan_orders(view.fair, targets, holdings, view.books, view.skip, inp.market_map, sell_above_target=False) if o.kind == "increase"]
    caps = dict(market_map=inp.market_map, fair=view.fair, margins=view.margins, bankroll=inp.bankroll, limits=view.limits, log=False)
    alone = {}
    for order in planned:
        result = apply_exposure_caps([order], holdings, **caps)
        if result:
            alone[order.exchange_id] = result[0]
    accepted = {o.exchange_id for o in apply_exposure_caps(planned, holdings, **caps)}
    return planned, alone, accepted


def blocking_cap(order: PlannedOrder, inp: PreviewInputs, view: _Cycle, holdings: dict[str, Holding]) -> str:
    """Which order cap leaves an increase (on its own) below MIN_ORDER_SHARES."""
    mapping = inp.market_map.get(order.exchange_id)
    if mapping is None:
        return "not in market map"
    legs = race_legs_from_holdings(inp.market_map, holdings)[mapping.race.key]
    parties = [m.party for m in inp.market_map.by_race().get(mapping.race.key, [])]
    limits, bankroll = view.limits, inp.bankroll
    if max_shares_within_race_cap(parties, legs, mapping.party, order.side, order.price, order.quantity, limits.max_race_worst_loss_fraction * bankroll) < MIN_ORDER_SHARES:
        return "race (2%)"
    gross = sum(h.cost_basis for h in holdings.values())
    if math.floor(max(0.0, limits.max_gross_exposure_fraction * bankroll - gross) / order.price) < MIN_ORDER_SHARES:
        return "gross (30%)"
    if math.floor(limits.max_new_notional_per_cycle_fraction * bankroll / order.price) < MIN_ORDER_SHARES:
        return "per-cycle notional"
    return "other"


# ---------------------------------------------------------------------- the preview


def build_stress_preview(inp: PreviewInputs) -> tuple[list[PositionRow], dict[str, Any]]:
    holdings = holdings_from_positions(inp.positions)
    view = _cycle_view(inp)
    rows = position_rows(inp, view, holdings)
    bankroll = inp.bankroll

    # 2. Stress totals, top contributions, focus races.
    stress: dict[str, Any] = {}
    for wave in WAVES:
        for s in WAVE_SIZES:
            total = sum(r.losses[_key(wave, s)] for r in rows)
            stress[_key(wave, s)] = {
                "label": f"{wave_label(wave)}, s={s}: loss of every position",
                "loss": round(total, 2), "pct_bankroll": round(100 * total / bankroll, 2) if bankroll else None,
            }
    top = {
        wave: [
            {"race": r.race, "party": r.party, "side": r.side, "exchange_id": r.exchange_id, "loss": r.losses[_key(wave, ALT_STRESS_S)]}
            for r in sorted(rows, key=lambda r: -r.losses[_key(wave, ALT_STRESS_S)])[:TOP_CONTRIBUTIONS]
            if r.losses[_key(wave, ALT_STRESS_S)] > 0
        ]
        for wave in WAVES
    }
    focus = {
        race: {_key(wave, s): round(sum(r.losses[_key(wave, s)] for r in rows if r.race == race), 2) for wave in WAVES for s in WAVE_SIZES}
        for race in FOCUS_RACES
    }

    # 3. Alternative capacity per direction.
    room_today = directional_room(inp, holdings)
    alternative: dict[str, Any] = {}
    for direction in ("D", "R"):
        loss = direction_stress(rows, direction)
        exposure = room_today[direction]["exposure"]
        ratio = loss / exposure if exposure > 0 else None  # stress loss per nominal dollar held
        free_stress = ALT_STRESS_FRAC * bankroll - loss
        free_nominal = free_stress / ratio if ratio else None
        alternative[direction] = {
            "stress_loss_s0.5": round(loss, 2),
            "stress_budget": round(ALT_STRESS_FRAC * bankroll, 2),
            "free_stress": round(free_stress, 2),
            "loss_per_nominal_dollar": None if ratio is None else round(ratio, 4),
            "free_nominal_alternative": None if free_nominal is None else round(free_nominal, 2),
            "free_nominal_today": room_today[direction]["room"],
            "nominal_gained": None if free_nominal is None else round(free_nominal - room_today[direction]["room"], 2),
        }

    # 4. Simulation.
    today_targets = targets_today(inp, view, holdings)
    alt_targets, alt_scaling = targets_alternative(inp, view, holdings, rows)
    planned_t, alone_t, accepted_t = _increases(inp, view, holdings, today_targets)
    planned_a, alone_a, accepted_a = _increases(inp, view, holdings, alt_targets)
    new_targets = {o.exchange_id: o for o in planned_a if o.exchange_id not in alone_t}
    unlocked = {eid: alone_a[eid] for eid in new_targets if eid in alone_a}
    still_blocked: dict[str, int] = {}
    for eid, order in new_targets.items():
        if eid not in unlocked:
            cap = blocking_cap(order, inp, view, holdings)
            still_blocked[cap] = still_blocked.get(cap, 0) + 1
    stake = sum(o.quantity * o.price for o in unlocked.values())
    expected = sum(o.quantity * (o.edge or 0.0) for o in unlocked.values())
    added_stress = sum(_order_stress(o, inp, view) for o in unlocked.values())
    clear = [o for o in unlocked.values() if (o.edge or 0.0) >= CLEAR_EDGE]
    simulation = {
        "targets_today": len(today_targets),
        "targets_alternative": len(alt_targets),
        "alternative_scaling": alt_scaling,
        "increases_with_room_today": len(alone_t),
        "increases_accepted_in_one_cycle_today": len(accepted_t),
        "new_targets_without_room_today": len(new_targets),
        "unlocked": len(unlocked),
        "unlocked_accepted_in_one_cycle": len(set(unlocked) & accepted_a),
        "unlocked_stake": round(stake, 2),
        "unlocked_expected_profit": round(expected, 2),
        "unlocked_stress_s0.5": round(added_stress, 2),
        "still_blocked_by": still_blocked,
        "unlocked_orders": [
            {"exchange_id": o.exchange_id, "race": o.race_key, "side": o.side, "quantity": o.quantity, "price": o.price, "edge": None if o.edge is None else round(o.edge, 4)}
            for o in sorted(unlocked.values(), key=lambda o: -(o.edge or 0.0))
        ],
    }

    # 5. Summary.
    summary_line = {
        "unlocked_markets": len(unlocked),
        "unlocked_with_clear_edge": len(clear),
        "clear_edge_stake": round(sum(o.quantity * o.price for o in clear), 2),
        "clear_edge_expected_profit": round(sum(o.quantity * (o.edge or 0.0) for o in clear), 2),
        "expected_profit_per_stress_dollar": round(expected / added_stress, 3) if added_stress else None,
        "verdict": (
            "frees capital for markets with a clear edge" if clear else "frees no capital for markets with a clear edge"
        ),
    }
    summary = {
        "bankroll": round(bankroll, 2),
        "positions": len(rows),
        "priced_by": {src: sum(1 for r in rows if r.price_source == src) for src in ("fair", "mark", "none")},
        "directional_today": room_today,
        "stress": stress,
        "top_contributions_s0.5": top,
        "focus_races": focus,
        "alternative_capacity": alternative,
        "simulation": simulation,
        "summary": summary_line,
    }
    return rows, summary


def write_outputs(rows: list[PositionRow], summary: dict[str, Any], csv_path: Path = POSITIONS_CSV, json_path: Path = SUMMARY_JSON) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f for f in PositionRow.__dataclass_fields__ if f != "losses"] + [_key(w, s) for w in WAVES for s in WAVE_SIZES]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            writer.writerow({**{k: v for k, v in data.items() if k != "losses"}, **data["losses"]})
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Directional exposure today", extra={"directional": {positions_label(d): v for d, v in summary["directional_today"].items()}})
    for key, value in summary["stress"].items():
        logger.info("Stress loss", extra={"key": key, **value})  # label: "wave X (X wins), s=..."
    for wave, contributions in summary["top_contributions_s0.5"].items():
        for item in contributions:
            logger.info("Stress contribution", extra={"wave": wave_label(wave), "s": ALT_STRESS_S, **item})
    logger.info("Focus races", extra={"races": summary["focus_races"], "keys": "D@s = wave D (D wins), R@s = wave R (R wins)"})
    for direction, value in summary["alternative_capacity"].items():
        logger.info("Alternative directional capacity", extra={"positions": positions_label(direction), "adverse_wave": wave_label(adverse_wave(direction)), **value})
    logger.info("Alternative limit simulation", extra={k: v for k, v in summary["simulation"].items() if k != "unlocked_orders"})
    logger.info("Directional stress preview done", extra=summary["summary"])


async def _amain(args: argparse.Namespace) -> int:
    # Read-only on purpose: DRY_RUN forced on and the kill switch engaged before any request.
    settings = replace(load_settings(), dry_run=True)
    setup_logging(settings.log_level, settings.log_dir)
    market_map = MarketMap.load()
    async with ExchangeClient(settings) as client, PolymarketClient() as polymarket, KalshiClient() as kalshi:
        client.halt("directional_stress_preview.py is read-only")
        reader = SigReader(client, OUT_DIR / "raw", max_reads_per_min=args.max_reads_per_min)
        inp = await gather_inputs(reader, polymarket, kalshi, market_map, settings.tournament_slug, settings.oracle_staleness_threshold_s, with_depth=False)
        rows, summary = build_stress_preview(inp)
        write_outputs(rows, summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: national-wave stress and a stress-based directional limit, simulated.")
    parser.add_argument("--max-reads-per-min", type=float, default=DEFAULT_MAX_READS_PER_MIN, help=f"SIG read spacing (default {DEFAULT_MAX_READS_PER_MIN})")
    args = parser.parse_args(argv)
    if not 0 < args.max_reads_per_min <= 30:
        parser.error("--max-reads-per-min must be in (0, 30]: the live bot uses up to ~80 of the account's 100 reads/min")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
