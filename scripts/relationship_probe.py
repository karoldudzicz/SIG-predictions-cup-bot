"""READ-ONLY research E: SIG's relationship constraints and race basket sums over time.

Snapshot (always): GET /relationships/constraints (all evaluations, violationsOnly=false)
-> the currently violated constraints, logged per race. A constraint is put on a race when
every exchange it touches (observed prices, direction, suggested trades) maps to that race
in the market map; otherwise "cross:<races>" or "unmapped".

Sampling (--sample-minutes M, --interval-s S, default 60): every S seconds one sample =
the constraints read + GET /exchanges/prices for every mapped race with >= 2 parties.
Each sample is appended to samples.jsonl. Per race and sample:
    sum_bid  = sum of the parties' YES bids   (sell every leg: an edge when > 1)
    sum_ask  = sum of the parties' YES asks   (buy every leg: an edge when < 1 - only a
               sure 1 if a LISTED party wins; kinds are reported separately)
    spread_cost = sum of (ask - bid) over the legs
    engine_violation / engine_status = the engine's own evaluation of the race (max
               violationAmount over its violated constraints), None if it has none.
Reads per sample: 1 constraints + ceil(ids / 100) prices; whole races are packed into one
prices read so a race's legs are always priced together (237 mapped markets -> 3 reads,
so 4 per sample, not 1-2).

Report per race (and per UTC day, and for the first vs second half of the samples):
    persistent premium = median(sum_ask - 1) and median(sum_bid - 1);
    share of samples with engine_violation > 1 tick and > spread_cost;
    share of samples whose excess sits >= 1 / 2 ticks past the race's own premium (FIXED
    thresholds, no z-scores), plus first-half premium applied to the second half;
    executable violations NET of spread (sum_bid > 1 or sum_ask < 1) as episodes of
    consecutive samples; duration = last - first sighting (a lower bound: 0 for one).
Pre-registered rule (basket executor, multi-leg dry-run, is built only if): violations
net of spread lasting > 30 s occur on >= 3 races on each of >= 2 UTC days. Fewer than 2
UTC days of samples -> INSUFFICIENT DATA. Pool several runs with --analyse-only.

Reads only, through research.reader (any write raises; RESEARCH_READS_PER_MIN, default
10). Without --yes only the read estimate is logged. --analyse-only re-analyses every
samples.jsonl under data/edge_scan/relationship_probe/ with no request at all.
-> data/edge_scan/relationship_probe/<UTC date>/: raw.jsonl, samples.jsonl,
   races.csv, summary.json.

    python scripts/relationship_probe.py [--yes] [--sample-minutes 60] [--interval-s 60]
    python scripts/relationship_probe.py --analyse-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from exchange_api import PRICE_TICK, ApiError  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, ResearchAborted, ResearchReader, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min, write_csv, write_json  # noqa: E402

logger = logging.getLogger("relationship_probe")

SCRIPT = "relationship_probe"
PRICES_CHUNK = 100  # /exchanges/prices maximum ids
DEFAULT_SAMPLE_MINUTES = 0.0  # snapshot only
DEFAULT_INTERVAL_S = 60.0
DEVIATION_TICKS = (1, 2)
RULE_MIN_DURATION_S = 30.0
RULE_MIN_RACES = 3
RULE_MIN_DAYS = 2
GAP_FACTOR = 1.5  # consecutive samples further apart than this many intervals break an episode
_EPS = 1e-9
_STATUS_RANK = {"violated": 3, "unevaluable": 2, "satisfied": 1}

RACE_FIELDS = ("race", "legs", "samples", "premium_ask", "premium_bid", "engine_covered_share", "engine_gt_tick_share", "engine_gt_spread_share",
               "ask_cheap_1tick_share", "ask_cheap_2tick_share", "bid_rich_1tick_share", "bid_rich_2tick_share",
               "exec_bid_over_share", "exec_ask_under_share", "episodes", "median_duration_s", "episodes_gt_30s")


# ---------------------------------------------------------------------- races and reads


def race_legs(market_map: MarketMap) -> dict[str, list[str]]:
    """race key -> exchange ids, races with at least two mapped parties only."""
    return {race: sorted(m.sig_exchange_id for m in group) for race, group in market_map.by_race().items() if len(group) >= 2}


def pack_races(races: Mapping[str, Sequence[str]], max_ids: int = PRICES_CHUNK) -> list[list[str]]:
    """Exchange ids in chunks of <= max_ids, never splitting a race across two chunks."""
    chunks: list[list[str]] = [[]]
    for race in sorted(races):
        legs = list(races[race])
        if len(legs) > max_ids:
            raise ValueError(f"race {race} has more legs than one prices read can carry")
        if len(chunks[-1]) + len(legs) > max_ids:
            chunks.append([])
        chunks[-1].extend(legs)
    return [c for c in chunks if c]


def reads_per_sample(races: Mapping[str, Sequence[str]]) -> int:
    return 1 + len(pack_races(races))


def estimate(races: Mapping[str, Sequence[str]], sample_minutes: float, interval_s: float, reads_per_min: float) -> tuple[ReadEstimate, int]:
    samples = 1 + int(sample_minutes * 60 // interval_s) if sample_minutes > 0 else 1
    reads = 1 + samples * reads_per_sample(races)  # + the tournament read
    read_time = (reads - 1) * 60.0 / reads_per_min
    return ReadEstimate(reads, reads_per_min, extra_s=max(0.0, sample_minutes * 60 - read_time)), samples


# ---------------------------------------------------------------------- constraints


def constraint_exchange_ids(constraint: Mapping[str, Any]) -> set[str]:
    ids = {str(p.get("exchangeId")) for p in constraint.get("observedPrices") or [] if p.get("exchangeId") is not None}
    direction = constraint.get("direction") or {}
    ids |= {str(e) for e in (direction.get("fromExchangeIds") or []) + (direction.get("toExchangeIds") or [])}
    ids |= {str(t.get("exchangeId")) for t in constraint.get("suggestedCorrectiveTrades") or [] if t.get("exchangeId") is not None}
    return ids


def constraint_race(constraint: Mapping[str, Any], market_map: MarketMap) -> str:
    races = set()
    for eid in constraint_exchange_ids(constraint):
        mapping = market_map.get(eid)
        if mapping is None:
            return "unmapped"
        races.add(mapping.race.key)
    if not races:
        return "unmapped"
    return next(iter(races)) if len(races) == 1 else "cross:" + "+".join(sorted(races))


def engine_by_race(constraints: Iterable[Mapping[str, Any]], market_map: MarketMap) -> dict[str, dict[str, Any]]:
    """race -> {"status", "violation", "constraints", "types"}; status = the worst one
    (violated > unevaluable > satisfied), violation = max violationAmount among violated."""
    out: dict[str, dict[str, Any]] = {}
    for c in constraints:
        race = constraint_race(c, market_map)
        entry = out.setdefault(race, {"status": None, "violation": 0.0, "constraints": 0, "types": set()})
        status = c.get("evaluationStatus")
        entry["constraints"] += 1
        entry["types"].add(str(c.get("type")))
        if _STATUS_RANK.get(status, 0) > _STATUS_RANK.get(entry["status"], 0):
            entry["status"] = status
        if status == "violated":
            entry["violation"] = max(entry["violation"], float(c.get("violationAmount") or 0.0))
    for entry in out.values():
        entry["types"] = sorted(entry["types"])
    return out


def compact_violation(c: Mapping[str, Any], race: str) -> dict[str, Any]:
    return {"race": race, "relationshipId": c.get("relationshipId"), "type": c.get("type"), "violationAmount": c.get("violationAmount"),
            "currentValue": c.get("currentValue"), "bound": c.get("bound"), "lowerBound": c.get("lowerBound"), "upperBound": c.get("upperBound"),
            "suggested": [(t.get("action"), t.get("outcomeSide"), t.get("exchangeId")) for t in c.get("suggestedCorrectiveTrades") or []]}


# ---------------------------------------------------------------------- one sample


def race_rows(ts: str, races: Mapping[str, Sequence[str]], prices: Mapping[str, Mapping[str, Any]], engine: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per race. A sum is None unless every leg is quoted on that side."""
    rows = []
    for race, legs in sorted(races.items()):
        bids = [(prices.get(e) or {}).get("bestBid") for e in legs]
        asks = [(prices.get(e) or {}).get("bestAsk") for e in legs]
        sum_bid = None if any(b is None for b in bids) else round(math.fsum(float(b) for b in bids), 6)
        sum_ask = None if any(a is None for a in asks) else round(math.fsum(float(a) for a in asks), 6)
        spread = None if sum_bid is None or sum_ask is None else round(sum_ask - sum_bid, 6)
        eng = engine.get(race) or {}
        rows.append({"ts": ts, "race": race, "legs": len(legs), "sum_bid": sum_bid, "sum_ask": sum_ask, "spread_cost": spread,
                     "engine_status": eng.get("status"), "engine_violation": eng.get("violation") if eng else None})
    return rows


# ---------------------------------------------------------------------- analysis


def _epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _share(flags: Sequence[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def analyse_race(rows: Sequence[Mapping[str, Any]], interval_s: float, premiums: Mapping[str, float | None] | None = None) -> dict[str, Any]:
    """Statistics of one race's samples (time order). `premiums`: use these persistent
    premiums instead of this data's own medians (out-of-sample check)."""
    rows = sorted(rows, key=lambda r: r["ts"])
    ask_x = [None if r["sum_ask"] is None else r["sum_ask"] - 1.0 for r in rows]
    bid_x = [None if r["sum_bid"] is None else r["sum_bid"] - 1.0 for r in rows]
    own_ask, own_bid = stats.median_or_none(ask_x), stats.median_or_none(bid_x)
    p_ask = own_ask if premiums is None else premiums.get("ask")
    p_bid = own_bid if premiums is None else premiums.get("bid")
    ticks = [k * PRICE_TICK for k in DEVIATION_TICKS]
    ask_dev = stats.premium_deviation_shares(ask_x, p_ask, ticks, direction=-1)
    bid_dev = stats.premium_deviation_shares(bid_x, p_bid, ticks, direction=+1)
    covered = [r for r in rows if r.get("engine_status") is not None]
    eng_vs_spread = [r for r in covered if r.get("spread_cost") is not None]
    bid_over = [r["sum_bid"] is not None and r["sum_bid"] > 1.0 + _EPS for r in rows]
    ask_under = [r["sum_ask"] is not None and r["sum_ask"] < 1.0 - _EPS for r in rows]
    executable = [a or b for a, b in zip(bid_over, ask_under)]
    times = [_epoch(r["ts"]) for r in rows]
    runs = stats.episodes(times, executable, GAP_FACTOR * interval_s)
    for run in runs:
        kinds = {("bid_over" if bo else "ask_under") for t, bo, au in zip(times, bid_over, ask_under) if run["start"] <= t <= run["end"] and (bo or au)}
        run["kinds"] = sorted(kinds)
        run["day"] = datetime.fromtimestamp(run["start"], tz=timezone.utc).strftime("%Y-%m-%d")
    long_runs = [r for r in runs if r["duration_s"] > RULE_MIN_DURATION_S]
    k1, k2 = (f"{t:g}" for t in ticks)
    return {
        "samples": len(rows),
        "premium_ask": None if own_ask is None else round(own_ask, 4),
        "premium_bid": None if own_bid is None else round(own_bid, 4),
        "engine_covered_share": _share([r.get("engine_status") is not None for r in rows]),
        "engine_gt_tick_share": _share([float(r.get("engine_violation") or 0.0) > PRICE_TICK + _EPS for r in covered]),
        "engine_gt_spread_share": _share([float(r.get("engine_violation") or 0.0) > r["spread_cost"] + _EPS for r in eng_vs_spread]),
        "ask_cheap_1tick_share": ask_dev[k1], "ask_cheap_2tick_share": ask_dev[k2],
        "bid_rich_1tick_share": bid_dev[k1], "bid_rich_2tick_share": bid_dev[k2],
        "exec_bid_over_share": _share(bid_over), "exec_ask_under_share": _share(ask_under),
        "episodes": len(runs),
        "median_duration_s": stats.median_or_none(r["duration_s"] for r in runs),
        "episodes_gt_30s": len(long_runs),
        "long_episodes": long_runs,
    }


def _analyse_subset(rows: Sequence[Mapping[str, Any]], interval_s: float, premiums: Mapping[str, Mapping[str, float | None]] | None = None) -> dict[str, dict[str, Any]]:
    by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_race[row["race"]].append(row)
    return {race: analyse_race(rs, interval_s, None if premiums is None else premiums.get(race, {})) for race, rs in sorted(by_race.items())}


def _qualifying_races(per_race: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(race for race, s in per_race.items() if s["episodes_gt_30s"] > 0)


def analyse(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled, per UTC day, per half, out-of-sample, and the pre-registered verdict.
    `samples`: samples.jsonl records ({"ts", "interval_s", "rows"})."""
    samples = sorted(samples, key=lambda s: s["ts"])
    if not samples:
        return {"samples": 0, "verdict": "INSUFFICIENT DATA", "reason": "no samples"}
    interval_s = max(float(s.get("interval_s") or DEFAULT_INTERVAL_S) for s in samples)
    rows = [r for s in samples for r in s["rows"]]
    pooled = _analyse_subset(rows, interval_s)
    first, second = stats.split_halves(samples, key=lambda s: s["ts"])
    halves = {name: _analyse_subset([r for s in part for r in s["rows"]], interval_s) for name, part in (("first_half", first), ("second_half", second))}
    first_premiums = {race: {"ask": s["premium_ask"], "bid": s["premium_bid"]} for race, s in halves["first_half"].items()}
    oos = _analyse_subset([r for s in second for r in s["rows"]], interval_s, premiums=first_premiums)
    days = stats.by_utc_day(samples, ts=lambda s: datetime.fromisoformat(s["ts"].replace("Z", "+00:00")))
    per_day = {day: _analyse_subset([r for s in part for r in s["rows"]], interval_s) for day, part in days.items()}

    # Rule: on >= RULE_MIN_DAYS UTC days, >= RULE_MIN_RACES races each had an executable
    # violation lasting > 30 s (episodes found per day, so none spans two days).
    qualifying_days = {day: _qualifying_races(per_race) for day, per_race in per_day.items()}
    passing_days = sorted(day for day, races in qualifying_days.items() if len(races) >= RULE_MIN_RACES)
    if len(days) < RULE_MIN_DAYS:
        verdict, reason = "INSUFFICIENT DATA", f"samples cover {len(days)} UTC day(s); the rule needs >= {RULE_MIN_DAYS}"
    elif len(passing_days) >= RULE_MIN_DAYS:
        verdict, reason = "PASS", f"days with >= {RULE_MIN_RACES} races: {passing_days}"
    else:
        verdict, reason = "FAIL", f"days with >= {RULE_MIN_RACES} races: {passing_days or 'none'}"

    def compact(per_race: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        return {"races": len(per_race), "races_with_executable_gt_30s": _qualifying_races(per_race),
                "median_premium_ask": stats.median_or_none(s["premium_ask"] for s in per_race.values()),
                "median_premium_bid": stats.median_or_none(s["premium_bid"] for s in per_race.values()),
                "engine_covered_races": sum(1 for s in per_race.values() if (s["engine_covered_share"] or 0) > 0)}

    return {
        "samples": len(samples), "first_ts": samples[0]["ts"], "last_ts": samples[-1]["ts"], "interval_s": interval_s,
        "utc_days": list(days),
        "pooled": compact(pooled),
        "per_day": {day: compact(per_race) for day, per_race in per_day.items()},
        "halves": {name: compact(per_race) for name, per_race in halves.items()},
        "out_of_sample": {"premium_from": "first_half", "applied_to": "second_half",
                          "ask_cheap_2tick_share_median": stats.median_or_none(s["ask_cheap_2tick_share"] for s in oos.values()),
                          "bid_rich_2tick_share_median": stats.median_or_none(s["bid_rich_2tick_share"] for s in oos.values())},
        "rule": f"executable violations net of spread lasting > {RULE_MIN_DURATION_S:g} s on >= {RULE_MIN_RACES} races on each of >= {RULE_MIN_DAYS} UTC days",
        "qualifying_races_per_day": qualifying_days,
        "verdict": verdict, "reason": reason,
        "hypotheses_tested": {"races": len(pooled), "kinds_per_race": 2, "total": 2 * len(pooled)},
        "_per_race": pooled,
    }


def load_samples(root: Path | None = None) -> list[dict[str, Any]]:
    """Every saved sample of every run (data/edge_scan/relationship_probe/*/samples.jsonl)."""
    root = root or OUT_ROOT / SCRIPT
    return [rec for path in sorted(root.glob("*/samples.jsonl")) for rec in read_jsonl(path)]


def write_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    per_race = report.get("_per_race") or {}
    write_csv(out_dir / "races.csv", ({"race": race, **{k: v for k, v in s.items() if k != "long_episodes"}} for race, s in per_race.items()), RACE_FIELDS)
    write_json(out_dir / "summary.json", {k: v for k, v in report.items() if k != "_per_race"} | {"long_episodes": {r: s["long_episodes"] for r, s in per_race.items() if s["long_episodes"]}})
    logger.info("Relationship probe verdict", extra={k: report.get(k) for k in ("samples", "utc_days", "verdict", "reason", "hypotheses_tested")})
    for name, part in (report.get("halves") or {}).items():
        logger.info("Half", extra={"half": name, **part})
    for day, part in (report.get("per_day") or {}).items():
        logger.info("UTC day", extra={"day": day, **part})


# ---------------------------------------------------------------------- sampling


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def take_sample(reader: ResearchReader, races: Mapping[str, Sequence[str]], chunks: Sequence[Sequence[str]], tournament_id: str | None, market_map: MarketMap,
                      seen_violations: set[str], ts: str) -> dict[str, Any]:
    body = await reader.get("/relationships/constraints", {"violationsOnly": False, "tournamentId": tournament_id})
    constraints = body.get("data") or []
    prices: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        page = await reader.get("/exchanges/prices", {"ids": ",".join(chunk), "tournamentId": tournament_id})
        prices.update({str(r["exchangeId"]): r for r in page.get("data") or []})
    violated = [compact_violation(c, constraint_race(c, market_map)) for c in constraints if c.get("evaluationStatus") == "violated"]
    for v in violated:
        key = str(v["relationshipId"])
        if key not in seen_violations:  # log a violation once, when first seen
            seen_violations.add(key)
            logger.info("Violated constraint", extra=v)
    return {"ts": ts, "constraints": len(constraints), "violated": violated, "computedAt": body.get("computedAt"),
            "rows": race_rows(ts, races, prices, engine_by_race(constraints, market_map))}


async def run(reader: ResearchReader, slug: str, market_map: MarketMap, samples: int, interval_s: float,
              clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
              now_iso: Callable[[], str] = _now_iso) -> dict[str, Any]:
    races = race_legs(market_map)
    chunks = pack_races(races)
    tournament = await reader.get(f"/tournaments/{slug}")
    tournament_id = tournament.get("id")
    samples_path = reader.out_dir / "samples.jsonl"
    seen: set[str] = set()
    taken: list[dict[str, Any]] = []
    for i in range(samples):
        started = clock()
        try:
            sample = await take_sample(reader, races, chunks, tournament_id, market_map, seen, now_iso())
        except ResearchAborted:
            logger.error("Sampling aborted after repeated 429; analysing what was collected", extra={"samples": len(taken)})
            break
        except ApiError as exc:
            logger.warning("Sample skipped", extra={"sample": i, "status": exc.status, "code": exc.code})
        else:
            sample["interval_s"] = interval_s
            with samples_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, separators=(",", ":")) + "\n")
            taken.append(sample)
            by_race = defaultdict(list)
            for v in sample["violated"]:
                by_race[v["race"]].append(v["relationshipId"])
            logger.info("Sample", extra={"sample": i + 1, "of": samples, "constraints": sample["constraints"], "violated": len(sample["violated"]),
                                         "violated_by_race": dict(by_race),
                                         "races_sum_bid_over_1": sum(1 for r in sample["rows"] if r["sum_bid"] is not None and r["sum_bid"] > 1 + _EPS),
                                         "races_sum_ask_under_1": sum(1 for r in sample["rows"] if r["sum_ask"] is not None and r["sum_ask"] < 1 - _EPS)})
        if i < samples - 1:
            await sleep(max(0.0, interval_s - (clock() - started)))
    report = analyse(taken)
    write_report(reader.out_dir, report)
    return report


async def _amain(args: argparse.Namespace, reads_per_min: float, market_map: MarketMap, samples: int) -> int:
    async with research_reader(SCRIPT, reads_per_min) as reader:
        await run(reader, reader.settings.tournament_slug, market_map, samples, args.interval_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: SIG relationship constraints and race basket sums, sampled.")
    add_research_args(parser)
    parser.add_argument("--sample-minutes", type=float, default=DEFAULT_SAMPLE_MINUTES, help="sample for this long (default 0: one snapshot)")
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S, help=f"seconds between samples (default {DEFAULT_INTERVAL_S:g})")
    parser.add_argument("--analyse-only", action="store_true", help="re-analyse every saved samples.jsonl; no API request")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    if args.analyse_only:
        samples = load_samples()
        report = analyse(samples)
        out = OUT_ROOT / SCRIPT / "pooled"
        out.mkdir(parents=True, exist_ok=True)
        write_report(out, report)
        return 0
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    if args.sample_minutes < 0 or args.interval_s <= 0:
        parser.error("--sample-minutes must be >= 0 and --interval-s > 0")
    market_map = MarketMap.load()
    races = race_legs(market_map)
    per_sample = reads_per_sample(races)
    min_interval = per_sample * 60.0 / reads_per_min
    if args.sample_minutes > 0 and args.interval_s < min_interval:
        parser.error(f"--interval-s must be >= {min_interval:g}: one sample is {per_sample} reads at {reads_per_min:g} reads/min")
    read_estimate, samples = estimate(races, args.sample_minutes, args.interval_s, reads_per_min)
    logger.info("Sampling plan", extra={"races": len(races), "reads_per_sample": per_sample, "samples": samples, "interval_s": args.interval_s})
    if not confirm_fetch(read_estimate, args.yes):
        return 0
    return asyncio.run(_amain(args, reads_per_min, market_map, samples))


if __name__ == "__main__":
    raise SystemExit(main())
