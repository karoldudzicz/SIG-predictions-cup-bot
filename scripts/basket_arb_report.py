"""OFFLINE research: is the race basket (constraint) arbitrage worth an executor?
HYPOTHETICAL throughout; zero SIG reads - only data saved by scripts/relationship_probe.py.

Input: one or more probe run directories (data/edge_scan/relationship_probe/<date>/), each
with raw.jsonl (and samples.jsonl for the sample interval). The probe's samples.jsonl keeps
only per-race SUMS, so per-leg prices are rebuilt from raw.jsonl: every
/relationships/constraints read starts a sample and the /exchanges/prices reads after it
belong to it. Races = mapped races with >= 2 parties (as in the probe).

The trade: when the YES bids of a race sum past 1, buy NO on every leg at 1 - bid. At most
one listed party wins, so at least n - 1 NO legs pay 1: profit >= sum(bids) - 1 per basket
(more if no listed party wins). Per race and in total:
  1) share of observations with sum(bids) > 1, per UTC day and overall;
  2) excess in ticks after one tick of slippage per leg: median, p10, p90;
  3) depth: the probe reads no order books, so shares at the best bid come from order books
     saved by OTHER runs (--books raw.jsonl ...: /exchanges/{id}/orderbook or the books
     embedded in /tournaments/{slug}/markets/{id}), read at another time - labelled so;
     thinnest leg = the basket size available at the quoted prices; "n/a" without books;
  4) capital locked per basket (sum of NO prices), profit per 1 SUSQ locked without and
     with a 0.005 haircut per leg; no fee is assumed - the break-even fee is printed;
  5) persistence: runs of consecutive mispriced observations (the executor needs several
     seconds for all legs); duration = last - first sighting, a lower bound;
  6) expected profit per hour at a realistic size (min(--size, depth)): each run of at
     least --min-obs observations is executed ONCE at its first observation's prices,
     after slippage and haircut, divided by the hours observed. Ranked. HYPOTHETICAL.
-> <last input dir>/analysis/ (or --out): races.csv, summary.md.

    python scripts/basket_arb_report.py data/edge_scan/relationship_probe/2026-10-09 data/edge_scan/relationship_probe/2026-10-12
        [--books data/edge_scan/position_review/2026-10-12/raw.jsonl] [--size 100] [--min-obs 2] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import read_jsonl, write_csv  # noqa: E402

logger = logging.getLogger("basket_arb_report")

TICK = 0.005
SLIPPAGE_TICKS = 1.0
HAIRCUT = 0.005
DEFAULT_SIZE = 100.0
DEFAULT_MIN_OBS = 2
DEFAULT_INTERVAL_S = 60.0
_EPS = 1e-9
LABEL = "HYPOTHETICAL: top-of-book prints of a 60 s sampler, no fills simulated, no fee assumed; profit is a floor (more if no listed party wins)"

FIELDS = ("rank", "race", "legs", "observations", "mispriced", "mispriced_share", "executable_after_slippage", "ticks_median", "ticks_p10", "ticks_p90",
          "capital_median", "profit_per_capital_median", "profit_per_capital_haircut_median", "break_even_fee_per_leg_median",
          "break_even_fee_of_capital_median", "runs", "runs_ge_min_obs", "run_obs_median", "run_obs_max", "run_duration_s_median",
          "depth_shares", "depth_from", "size_used", "profit_per_hour", "per_day_mispriced_share")


def _ts(raw: Any) -> datetime:
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------- inputs


def samples_from_raw(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """[{"ts", "prices": {exchange_id: row}}]: a successful constraints read opens a sample, the
    price reads after it fill it. A sample cut short by an error keeps what it has."""
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for rec in records:
        path = rec.get("path")
        if path == "/relationships/constraints":
            current = None
            if "body" in rec:  # a failed constraints read starts no sample, and adopts no prices
                current = {"ts": rec.get("ts"), "prices": {}}
                out.append(current)
        elif path == "/exchanges/prices" and "body" in rec and current is not None:
            current["prices"].update({str(r["exchangeId"]): r for r in (rec["body"] or {}).get("data") or []})
    return [s for s in out if s["ts"]]


def interval_from(samples_jsonl: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]) -> float:
    saved = [float(s["interval_s"]) for s in samples_jsonl if s.get("interval_s")]
    if saved:
        return max(saved)
    times = sorted(_ts(s["ts"]).timestamp() for s in samples)
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return stats.median_or_none(gaps) or DEFAULT_INTERVAL_S


def race_legs(market_map: MarketMap) -> dict[str, list[str]]:
    return {race: sorted(m.sig_exchange_id for m in group) for race, group in market_map.by_race().items() if len(group) >= 2}


def depth_from_books(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """exchange -> {"quantity", "price", "ts"} at the best YES bid (buying NO hits it), from the
    latest saved order book or market detail of each exchange."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        if "body" not in rec:
            continue
        path, body = str(rec.get("path") or ""), rec["body"] or {}
        books: list[tuple[str, Mapping[str, Any]]] = []
        if path.endswith("/orderbook"):
            books.append((path.split("/")[2], body))
        elif re.fullmatch(r"/tournaments/[^/]+/markets/[^/]+", path):
            books += [(str(ex.get("id")), ex) for ex in body.get("exchanges") or []]
        for eid, book in books:
            bids = sorted(book.get("bids") or [], key=lambda l: -float(l["price"]))
            if bids:
                out[eid] = {"quantity": abs(float(bids[0]["quantity"])), "price": float(bids[0]["price"]), "ts": rec.get("ts")}
    return out


# ---------------------------------------------------------------------- analysis


def race_observations(samples: Sequence[Mapping[str, Any]], legs: Sequence[str]) -> list[dict[str, Any]]:
    """Per sample with every leg bid-quoted: time and basket metrics."""
    obs = []
    for s in samples:
        bids = [(s["prices"].get(e) or {}).get("bestBid") for e in legs]
        if any(b is None for b in bids):
            continue
        m = stats.basket_no_metrics([float(b) for b in bids], TICK, SLIPPAGE_TICKS, HAIRCUT)
        obs.append({"t": _ts(s["ts"]), **m})
    return sorted(obs, key=lambda o: o["t"])


def analyse_race(race: str, legs: Sequence[str], obs: Sequence[Mapping[str, Any]], interval_s: float, hours: float, depth: Mapping[str, Mapping[str, Any]],
                 size: float, min_obs: int) -> dict[str, Any]:
    mis = [o for o in obs if o["excess"] > _EPS]
    runs = stats.episodes([o["t"].timestamp() for o in obs], [o["excess"] > _EPS for o in obs], 1.5 * interval_s)
    by_t = {o["t"].timestamp(): o for o in obs}
    known = [depth.get(e) for e in legs]
    thinnest = None if any(d is None for d in known) else min(d["quantity"] for d in known)
    size_used = size if thinnest is None else min(size, thinnest)
    profit = 0.0
    for run in runs:
        if run["samples"] >= min_obs:
            first = by_t[run["start"]]
            per_share = first["excess_after_slippage"] - first["legs"] * HAIRCUT
            profit += max(0.0, per_share) * size_used
    days = stats.by_utc_day(obs, ts=lambda o: o["t"])
    ticks = sorted(o["excess_ticks_after_slippage"] for o in mis)

    def med(key: str) -> float | None:
        return stats.median_or_none(o[key] for o in mis)

    def pct(q: float) -> float | None:
        return None if not ticks else round(stats.percentile(ticks, q), 2)

    return {"race": race, "legs": len(legs), "observations": len(obs), "mispriced": len(mis),
            "mispriced_share": round(len(mis) / len(obs), 4) if obs else None,
            "executable_after_slippage": sum(1 for o in mis if o["excess_after_slippage"] > _EPS),
            "ticks_median": pct(50), "ticks_p10": pct(10), "ticks_p90": pct(90),
            "capital_median": med("capital"), "profit_per_capital_median": med("profit_per_capital"),
            "profit_per_capital_haircut_median": med("profit_per_capital_haircut"),
            "break_even_fee_per_leg_median": med("break_even_fee_per_leg"), "break_even_fee_of_capital_median": med("break_even_fee_of_capital"),
            "runs": len(runs), "runs_ge_min_obs": sum(1 for r in runs if r["samples"] >= min_obs),
            "run_obs_median": stats.median_or_none(r["samples"] for r in runs), "run_obs_max": max((r["samples"] for r in runs), default=None),
            "run_duration_s_median": stats.median_or_none(r["duration_s"] for r in runs),
            "depth_shares": thinnest, "depth_from": None if thinnest is None else "saved books (not read with the sample): " + ", ".join(sorted({str(d["ts"]) for d in known})),
            "size_used": size_used, "profit_per_hour": round(profit / hours, 4) if hours > 0 else None,
            "per_day_mispriced_share": {d: round(sum(1 for o in os_ if o["excess"] > _EPS) / len(os_), 4) for d, os_ in days.items()}}


def analyse(samples: Sequence[Mapping[str, Any]], races: Mapping[str, Sequence[str]], interval_s: float, depth: Mapping[str, Mapping[str, Any]],
            size: float = DEFAULT_SIZE, min_obs: int = DEFAULT_MIN_OBS) -> dict[str, Any]:
    times = sorted(_ts(s["ts"]).timestamp() for s in samples)
    hours = 0.0 if not times else (times[-1] - times[0] + interval_s) / 3600
    rows = [analyse_race(race, legs, race_observations(samples, legs), interval_s, hours, depth, size, min_obs) for race, legs in sorted(races.items())]
    rows = [r for r in rows if r["observations"]]
    rows.sort(key=lambda r: (-(r["profit_per_hour"] or 0.0), -r["mispriced"], r["race"]))
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    obs_total = sum(r["observations"] for r in rows)
    mis_total = sum(r["mispriced"] for r in rows)
    per_day: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        day = _ts(s["ts"]).strftime("%Y-%m-%d")
        for legs in races.values():
            bids = [(s["prices"].get(e) or {}).get("bestBid") for e in legs]
            if all(b is not None for b in bids):
                per_day[day].append(math.fsum(float(b) for b in bids) - 1.0)
    total = {"samples": len(samples), "hours": round(hours, 2), "interval_s": interval_s, "races": len(rows), "observations": obs_total,
             "mispriced": mis_total, "mispriced_share": round(mis_total / obs_total, 4) if obs_total else None,
             "per_day_mispriced_share": {d: round(sum(1 for x in xs if x > _EPS) / len(xs), 4) for d, xs in sorted(per_day.items())},
             "runs_ge_min_obs": sum(r["runs_ge_min_obs"] for r in rows), "profit_per_hour": round(math.fsum(r["profit_per_hour"] or 0.0 for r in rows), 4),
             "races_with_depth": sum(1 for r in rows if r["depth_shares"] is not None), "size": size, "min_obs": min_obs}
    return {"rows": rows, "total": total}


def summary_markdown(result: Mapping[str, Any], inputs: Sequence[str]) -> str:
    t = result["total"]
    lines = ["# Basket (constraint) arbitrage - probe analysis (HYPOTHETICAL)", "", f"_{LABEL}._", "",
             f"- Inputs: {', '.join(inputs)}; {t['samples']} samples over {t['hours']} h (interval {t['interval_s']:g} s), {t['races']} races.",
             f"- Mispriced (sum of YES bids > 1): {t['mispriced']} of {t['observations']} race observations ({t['mispriced_share']}); per day: {t['per_day_mispriced_share']}.",
             f"- Runs of >= {t['min_obs']} consecutive mispriced observations: {t['runs_ge_min_obs']}; expected profit per hour at size <= {t['size']:g}: "
             f"{t['profit_per_hour']} SUSQ (one execution per run, after {SLIPPAGE_TICKS:g} tick slippage and a {HAIRCUT} haircut per leg).",
             f"- Depth: {t['races_with_depth']} races have saved order books (other runs, not read with the samples); without books the size is the --size assumption.",
             "- No fee is assumed: break-even fee per leg per share = (sum of bids - 1) / legs; as a share of capital = (sum of bids - 1) / sum of NO prices.",
             "", "| rank | race | legs | obs | mispriced | ticks p10 / median / p90 (after slippage) | capital | profit / SUSQ (haircut) | break-even fee / leg | runs >= min (max obs) | depth | profit / h |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|",
             *[f"| {r['rank']} | {r['race']} | {r['legs']} | {r['observations']} | {r['mispriced']} ({r['mispriced_share']}) | {r['ticks_p10']} / {r['ticks_median']} / {r['ticks_p90']} | "
               f"{_f(r['capital_median'])} | {_f(r['profit_per_capital_median'])} ({_f(r['profit_per_capital_haircut_median'])}) | {_f(r['break_even_fee_per_leg_median'])} | "
               f"{r['runs_ge_min_obs']} ({r['run_obs_max']}) | {r['depth_shares'] if r['depth_shares'] is not None else 'n/a'} | {r['profit_per_hour']} |"
               for r in result["rows"] if r["mispriced"]], ""]
    return "\n".join(lines)


def _f(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: basket arbitrage analysis from saved relationship_probe runs (HYPOTHETICAL, 0 reads).")
    parser.add_argument("runs", nargs="+", type=Path, help="relationship_probe run directories (with raw.jsonl)")
    parser.add_argument("--books", nargs="*", type=Path, default=[], help="raw.jsonl files with saved order books / market details (depth)")
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE, help=f"basket size assumed when no depth is known (default {DEFAULT_SIZE:g})")
    parser.add_argument("--min-obs", type=int, default=DEFAULT_MIN_OBS, help=f"consecutive observations a run needs to count (default {DEFAULT_MIN_OBS})")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <last run>/analysis)")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    missing = [str(d) for d in args.runs if not (d / "raw.jsonl").exists()]
    if missing:
        logger.error("No raw.jsonl in", extra={"runs": missing})
        return 2
    samples = sorted((s for d in args.runs for s in samples_from_raw(read_jsonl(d / "raw.jsonl"))), key=lambda s: str(s["ts"]))
    if not samples:
        logger.error("No probe samples in the given runs", extra={"runs": [str(d) for d in args.runs]})
        return 2
    saved = [s for d in args.runs for s in read_jsonl(d / "samples.jsonl")]
    depth = depth_from_books([rec for p in args.books for rec in read_jsonl(p)])
    result = analyse(samples, race_legs(MarketMap.load()), interval_from(saved, samples), depth, args.size, args.min_obs)
    out = args.out or args.runs[-1] / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "races.csv", result["rows"], FIELDS)
    (out / "summary.md").write_text(summary_markdown(result, [str(d) for d in args.runs]), encoding="utf-8")
    logger.info("Basket arbitrage analysis done (HYPOTHETICAL, 0 reads)", extra={"out_dir": str(out), **result["total"], "label": LABEL})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
