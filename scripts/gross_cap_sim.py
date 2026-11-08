"""OFFLINE research: what would a looser taker gross cap add to the settlement carry scan?
HYPOTHETICAL throughout (oracle as an independent reference, haircut 0.02 on its
probability, as the scan ranks); zero reads - only files already on disk.

Inputs:
  - the newest data/edge_scan/settlement_carry_scan/*/candidates.csv (or --candidates);
  - current positions, relationships and starting equity from the newest saved
    position_review raw.jsonl (settlement_carry_scan.saved_review; --equity overrides).
For TAKER_GROSS_FRAC in --fractions (default 0.40 baseline, 0.45, 0.50, 0.60) and once with no
gross cap at all (the ceiling under the other caps), the scan's own allocate() is rerun:
greedy in rank order, the bot's caps from risk.py (contract 0.02, race worst loss 0.02,
cluster 0.03) on STARTING equity, on top of what is already held. Only the gross fraction
changes (RiskLimits.max_gross_exposure_fraction).

The CSV keeps no order-book levels, only the depth to best + 0.01 (depth_shares,
depth_cost); each candidate's book is rebuilt as ONE level at the average price
depth_cost / depth_shares. Exact for the full depth; a partial fill is valued at that average,
slightly dearer than walking the levels cheapest first. Candidates without a book read get no
capacity, as in the scan.

Per fraction: gross room, SUSQ allocated, new contracts and races, added expected P&L at
settlement (haircut 0.02) and per SUSQ, the sum over races of the increase in race worst
loss (risk.race_worst_case_loss), the stress (sum of the 5 largest race worst losses) before
and after, the 10 largest additions by cost, and the cap that binds for each candidate
(depth / contract / race / cluster / gross).
-> <scan dir>/gross_cap/: gross_cap_sim.csv (summary rows + one row per candidate and
   fraction), summary.md.

    python scripts/gross_cap_sim.py [--fractions 0.40,0.45,0.50,0.60] [--candidates PATH] [--equity 100000]
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import logging
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, read_jsonl, write_csv  # noqa: E402
from risk import RaceLeg, RiskLimits, build_clusters, race_worst_case_loss  # noqa: E402

_spec = importlib.util.spec_from_file_location("settlement_carry_scan", ROOT / "scripts" / "settlement_carry_scan.py")
scan = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("settlement_carry_scan", scan)
_spec.loader.exec_module(scan)  # reuse the scan's allocate() / capacity() / saved_review()

logger = logging.getLogger("gross_cap_sim")

DEFAULT_FRACTIONS = (0.40, 0.45, 0.50, 0.60)
NO_GROSS_CAP = 1e9  # "unconstrained": no gross cap, every other cap kept
EV_KEY = f"ev_{scan.RANK_HAIRCUT:g}"
STRESS_N = 5
TOP_N = 10
LABEL = "HYPOTHETICAL: oracle probability as an independent reference, not the truth; haircut 0.02; caps on starting equity; no order is placed or simulated"
CAP_ORDER = ("depth", "contract", "race", "cluster", "gross")

SUMMARY_FIELDS = ("level", "fraction", "gross_room", "allocated_cost", "new_contracts", "new_races", "added_ev", "added_ev_per_susq",
                  "added_worst_loss", "stress_before", "stress_after")
CANDIDATE_FIELDS = ("level", "fraction", "rank", "exchange_id", "race", "party", "side", "price", "avg_fill", "depth_shares", "allocated_shares",
                    "allocated_cost", "added_ev", "binding")


# ---------------------------------------------------------------------- inputs


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """The scan's candidate rows with the fields allocate() needs (allocation columns dropped)."""
    keep = ("rank", "exchange_id", "race", "party", "side", "price", "oracle_p_side", "depth_shares", "depth_cost", "book_read")
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {k: r.get(k) for k in keep}
            row["rank"] = int(row["rank"])
            for k in ("price", "oracle_p_side", "depth_shares", "depth_cost"):
                row[k] = None if row[k] in (None, "") else float(row[k])
            row["book_read"] = str(row["book_read"]).strip().lower() == "true"
            out.append(row)
    return sorted(out, key=lambda r: r["rank"])


def levels_from_csv(cands: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """One level per candidate with a book: (depth_cost / depth_shares, depth_shares)."""
    out = {}
    for c in cands:
        if c["book_read"]:
            shares, cost = c.get("depth_shares") or 0.0, c.get("depth_cost") or 0.0
            out[(c["exchange_id"], c["side"])] = [(cost / shares, shares)] if shares > 0 else []
    return out


def saved_equity(root: Path | None = None) -> float | None:
    """initialBalance from the newest saved position_review read."""
    runs = sorted((root or OUT_ROOT / "position_review").glob("*/raw.jsonl"))
    equity = None
    for rec in read_jsonl(runs[-1]) if runs else []:
        if "body" in rec and re.fullmatch(r"/tournaments/[^/]+", str(rec.get("path") or "")):
            equity = float((rec["body"] or {}).get("initialBalance") or 0.0) or equity
    return equity


# ---------------------------------------------------------------------- simulation


def race_legs(positions: Sequence[Mapping[str, Any]], market_map: MarketMap) -> dict[str, list[RaceLeg]]:
    legs: dict[str, list[RaceLeg]] = defaultdict(list)
    for p in positions:
        q = float(p.get("quantity") or 0)
        m = market_map.get(str(p.get("exchangeId")))
        if q and m is not None:
            cost = abs(float(p["costBasis"])) if p.get("costBasis") is not None else abs(q) * float(p.get("avgCost") or 0)
            legs[m.race.key].append(RaceLeg(m.party, "yes" if q > 0 else "no", abs(q), cost))
    return legs


def worst_losses(legs: Mapping[str, Sequence[RaceLeg]], market_map: MarketMap) -> dict[str, float]:
    by_race = market_map.by_race()
    return {race: race_worst_case_loss([m.party for m in by_race.get(race, [])], ls) for race, ls in legs.items()}


def stress(losses: Mapping[str, float], n: int = STRESS_N) -> float:
    """Sum of the n largest race worst losses."""
    return math.fsum(sorted(losses.values(), reverse=True)[:n])


def binding(c: Mapping[str, Any], levels: Sequence[tuple[float, float]]) -> str:
    """Which cap set the allocated size (from the rooms allocate() recorded at that moment)."""
    if not c.get("book_read"):
        return "no book"
    if not levels:
        return "depth"
    max_price = levels[0][0] + scan.DEPTH_SLIPPAGE

    # What each money cap alone would allow - as if the book went on at its last price, so a
    # cap is never "binding" just because the depth ran out first.
    unlimited = [*levels, (levels[-1][0], math.inf)]

    def shares_for(room: float | None) -> float | None:
        return None if room is None else stats.walk_levels(unlimited, max_price, max_cost=room)[0]

    budget = {"depth": c.get("depth_shares"), "contract": shares_for(c.get("contract_room_cost")),
              "cluster": shares_for(c.get("cluster_room_cost")), "gross": shares_for(c.get("gross_room_cost"))}
    # capacity() hands the race cap the shares the other caps already allow (as an int), so
    # race_room_shares never exceeds them: the race cap binds only when it is clearly lower.
    known = [v for v in budget.values() if v is not None]
    race = c.get("race_room_shares")
    if race is not None and known and race < min(known) - 1.0:
        return "race"
    return stats.binding_caps(budget)


def simulate(fraction: float, base: Sequence[Mapping[str, Any]], levels: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
             positions: Sequence[Mapping[str, Any]], market_map: MarketMap, clusters: Mapping[str, str], equity: float,
             limits: RiskLimits | None = None) -> dict[str, Any]:
    """The scan's allocate() with max_gross_exposure_fraction = fraction; added EV and risk."""
    limits = replace(limits or RiskLimits(), max_gross_exposure_fraction=fraction)
    cands = copy.deepcopy(list(base))
    scan.allocate(cands, levels, positions, market_map, clusters, equity, limits)
    current = scan.book_from_positions(positions, market_map, clusters, equity, limits)
    gross_room = max(0.0, fraction * equity - current.gross_cost)
    alloc = [c for c in cands if c.get("allocated_shares")]
    before = race_legs(positions, market_map)
    after = {race: list(legs) for race, legs in before.items()}
    for c in alloc:
        after.setdefault(c["race"], []).append(RaceLeg(c["party"], c["side"], c["allocated_shares"], c["allocated_cost"]))
    loss_before, loss_after = worst_losses(before, market_map), worst_losses(after, market_map)
    cost = math.fsum(c["allocated_cost"] for c in alloc)
    ev = math.fsum(c[EV_KEY] for c in alloc)
    rows = [{"level": "candidate", "fraction": fraction, "rank": c["rank"], "exchange_id": c["exchange_id"], "race": c["race"], "party": c["party"], "side": c["side"],
             "price": c["price"], "avg_fill": c.get("avg_fill"), "depth_shares": c.get("depth_shares"), "allocated_shares": c.get("allocated_shares", 0.0),
             "allocated_cost": c.get("allocated_cost", 0.0), "added_ev": c.get(EV_KEY, 0.0), "binding": binding(c, levels.get((c["exchange_id"], c["side"])) or [])}
            for c in cands]
    return {"level": "summary", "fraction": fraction, "gross_room": round(gross_room, 2), "allocated_cost": round(cost, 2),
            "new_contracts": len({(c["exchange_id"], c["side"]) for c in alloc}), "new_races": len({c["race"] for c in alloc}),
            "added_ev": round(ev, 2), "added_ev_per_susq": round(ev / cost, 4) if cost else None,
            "added_worst_loss": round(math.fsum(loss_after[r] - loss_before.get(r, 0.0) for r in loss_after), 2),
            "stress_before": round(stress(loss_before), 2), "stress_after": round(stress(loss_after), 2),
            "_rows": rows, "_top": sorted((r for r in rows if r["allocated_shares"]), key=lambda r: -r["allocated_cost"])[:TOP_N],
            "_binding": dict(sorted(_count(r["binding"] for r in rows).items()))}


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def summary_markdown(results: Sequence[Mapping[str, Any]], sources: Mapping[str, Any]) -> str:
    def name(f: float) -> str:
        return "no gross cap" if f >= NO_GROSS_CAP else f"{f:.2f}"

    lines = ["# Gross cap sensitivity of the settlement carry scan (HYPOTHETICAL)", "", f"_{LABEL}._", "",
             f"- Candidates: {sources['candidates']}; positions: {sources['positions']}; starting equity {sources['equity']:g}.",
             "- Books rebuilt from the CSV as one level at the average price to best + 0.01 (partial fills valued at that average).", "",
             "| gross frac | gross room | allocated | contracts | races | added EV | EV / SUSQ | added worst loss | stress before -> after |",
             "|---|---|---|---|---|---|---|---|---|",
             *[f"| {name(r['fraction'])} | {r['gross_room'] if r['fraction'] < NO_GROSS_CAP else 'n/a'} | {r['allocated_cost']} | {r['new_contracts']} | {r['new_races']} | "
               f"{r['added_ev']} | {r['added_ev_per_susq']} | {r['added_worst_loss']} | {r['stress_before']} -> {r['stress_after']} |" for r in results], ""]
    for r in results:
        lines += [f"## {name(r['fraction'])}", "", f"Binding cap per candidate: {r['_binding']}", "",
                  "| race | side | shares | avg price | cost | expected P&L |", "|---|---|---|---|---|---|",
                  *[f"| {t['race']} | {t['side']} | {t['allocated_shares']} | {t['avg_fill']} | {t['allocated_cost']} | {t['added_ev']} |" for t in r["_top"]], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: settlement carry scan under looser taker gross caps (HYPOTHETICAL, 0 reads).")
    parser.add_argument("--fractions", type=lambda s: [float(x) for x in s.split(",") if x.strip()], default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--candidates", type=Path, default=None, help="candidates.csv (default: the newest settlement_carry_scan run)")
    parser.add_argument("--equity", type=float, default=None, help="starting equity (default: initialBalance from the saved position_review read)")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    path = args.candidates or (sorted((OUT_ROOT / "settlement_carry_scan").glob("*/candidates.csv")) or [None])[-1]
    if path is None or not path.exists():
        logger.error("No settlement_carry_scan candidates.csv: run settlement_carry_scan.py --yes first, or pass --candidates")
        return 2
    positions, relationships, positions_from = scan.saved_review()
    if positions is None:
        logger.error("No saved position_review read with positions: run position_review.py --yes first")
        return 2
    equity = args.equity or saved_equity()
    if not equity:
        logger.error("Starting equity unknown: pass --equity")
        return 2
    market_map = MarketMap.load()
    base = load_candidates(path)
    levels = levels_from_csv(base)
    clusters = {eid: c.key for c in build_clusters(market_map, relationships) for eid in c.exchange_ids}
    results = [simulate(f, base, levels, positions, market_map, clusters, equity) for f in [*sorted(args.fractions), NO_GROSS_CAP]]

    out = path.parent / "gross_cap"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "gross_cap_sim.csv", [*[{k: r[k] for k in SUMMARY_FIELDS} for r in results], *[row for r in results for row in r["_rows"]]],
              (*SUMMARY_FIELDS, *(f for f in CANDIDATE_FIELDS if f not in SUMMARY_FIELDS)))
    (out / "summary.md").write_text(summary_markdown(results, {"candidates": str(path), "positions": positions_from, "equity": equity}), encoding="utf-8")
    for r in results:
        logger.info("Gross cap simulation (HYPOTHETICAL)", extra={k: r[k] for k in SUMMARY_FIELDS})
    logger.info("Gross cap simulation done", extra={"out_dir": str(out), "label": LABEL})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
