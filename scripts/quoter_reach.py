"""OFFLINE research: how often could the quoter's quotes have been reached by real prints on the
quoter-eligible markets, and what was the 300 s maker markout vs the quote mid on those prints?

HYPOTHETICAL / PROXY throughout. NOT modelled: queue position, settlement risk (0 / 1), fees.
A feasibility count, not a test: no pre-registered rule, nothing frozen. Zero SIG and oracle reads.

Inputs: the saved tape and quote snapshots (matching as maker_markout_quotes: Lee-Ready against the
prevailing quote at most 45 s old, tick-rule fallback), the eligible markets of the newest
data/edge_scan/quoter_eligibility run (eligibility.csv), and their fair and u recomputed from that
run's oracle_quotes.json with the bot's rules (research.stats.oracle_fair / oracle_uncertainty, ages
as at the fetch). Fair and u are held CONSTANT over the whole history - they drifted.

Per eligible market and prevailing snapshot (touch bid / ask, spread S ticks), with
edge = max(0.01, u + 0.01) (the quoter's formula, research.stats.quoter_edge):
  - JOIN, S_min in {1, 2, 3}: a YES bid at the touch bid when S >= S_min and bid <= fair - edge; a NO
    bid (= a YES ask at the touch ask) when S >= S_min and ask >= fair + edge;
  - IMPROVE (the current quoter): research.stats.quoter_prices (a copy of quoter.plan_quotes at zero
    race delta: a tick inside the touch, capped at fair - edge, never marketable), S >= S_min.
A matched print is REACHABLE when its maker side (aggressor sell -> our YES bid, buy -> our YES ask)
was quotable in the prevailing snapshot and the print is AT the quote price (JOIN) or at or through
it (IMPROVE). Markout at 300 s = maker_sign * (print price - quote mid at t + 300 s); for IMPROVE the
JOIN markout minus 1 tick - an APPROXIMATION of the fill a tick inside.

Per (fair shift in {0, -0.02, +0.02}, variant, S_min):
  - reachable share of the matched prints, reachable prints per market-day (market-days with a
    matched print), share of snapshots with a quotable YES / NO side;
  - 300 s markout on reachable prints: mean in ticks with the cluster bootstrap (cluster =
    market-day, 1000 resamples, seed 7, 95%) by spread at the print (all / 1 / 2 / >=3) x time half
    (all / first / second, split over that configuration's reachable prints);
  - an optimistic no-queue UPPER BOUND in SUSQ per UTC day = sum of min(100, 0.25 x size) x markout;
  - per market: reachable prints, per day and mean markout; markets with < LOW_REACH flagged.
The 10 best and 10 worst markets (mean 300 s markout, JOIN, S_min 2, fair unshifted) are logged.
Verdict per variant (fair unshifted): INSUFFICIENT when fewer than MIN_MARKETS markets have at least
MIN_REACHABLE reachable prints at S_min 2.

Memory: streamed like maker_markout_quotes, two passes market by market, only aggregates kept;
--max-rss-mb (default 350) stops cleanly (exit 3).
-> data/edge_scan/quoter_reach/<UTC date>/: by_cut.csv, by_day.csv, by_market.csv, summary.json.

    python scripts/quoter_reach.py [--eligibility DIR] [--tape PATH ...] [--snapshots PATH ...] [--max-rss-mb 350]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
from array import array
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.memory import MemoryLimitExceeded, RssGuard  # noqa: E402
from research.reader import OUT_ROOT, run_dir, write_csv, write_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("maker_markout_quotes", ROOT / "scripts" / "maker_markout_quotes.py")
mq = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("maker_markout_quotes", mq)
_spec.loader.exec_module(mq)  # reuse the streaming stores and the per-market matching
maker = mq.maker

logger = logging.getLogger("quoter_reach")

SCRIPT = "quoter_reach"
TICK = 0.005
HORIZON = 300
SHIFTS = (0.0, -0.02, 0.02)
VARIANTS = ("join", "improve")
S_MINS = (1, 2, 3)
CONFIGS = tuple((shift, variant, s_min) for shift in SHIFTS for variant in VARIANTS for s_min in S_MINS)
REFERENCE = (0.0, "join", 2)  # ranks the best / worst markets
SPREADS = ("all", "1", "2", ">=3")
HALVES = ("all", "first_half", "second_half")
FILL_CAP, FILL_SHARE = 100.0, 0.25
MIN_MARKETS, MIN_REACHABLE, LOW_REACH = 6, 100, 50
TOP_N = 10
DEFAULT_MAX_RSS_MB = 350.0
EXIT_MEMORY = 3
LABEL = ("HYPOTHETICAL / PROXY quoter reach on eligible markets: fair and u held constant at the last oracle pass; "
         "IMPROVE markout = JOIN markout - 1 tick (APPROXIMATION); NOT modelled: queue position, settlement risk (0 / 1), fees. "
         "Feasibility count, not a test.")
CUT_FIELDS = ("fair_shift", "variant", "s_min", "spread", "half", "horizon_s", "prints", "qualifying", "drop_rate", "markets", "mean_ticks",
              "ci_low_ticks", "ci_high_ticks", "label")
DAY_FIELDS = ("fair_shift", "variant", "s_min", "day", "matched_prints", "reachable", "markets_reachable", "upper_bound_susq")
MARKET_FIELDS = ("fair_shift", "variant", "s_min", "exchange_id", "fair", "u", "matched_prints", "reachable", "days", "reachable_per_day",
                 "qualifying", "mean_ticks", "low_reach")


# ---------------------------------------------------------------------- inputs


def newest_eligibility(root: Path = OUT_ROOT) -> Path | None:
    runs = [p.parent for p in sorted((root / "quoter_eligibility").glob("*/oracle_quotes.json")) if (p.parent / "eligibility.csv").exists()]
    return runs[-1] if runs else None


def _quote(raw: Mapping[str, Any] | None, at: float) -> Any:
    """A saved oracle quote as the duck type stats.oracle_fair expects, aged as at the fetch."""
    if raw is None:
        return None
    return SimpleNamespace(mid=raw.get("mid"), spread=raw.get("spread"), age_seconds=at - float(raw.get("fetched_at") or 0.0))


def eligible_fairs(rows: Sequence[Mapping[str, str]], oracle: Mapping[str, Any], market_map: Any) -> dict[str, dict[str, float]]:
    """exchange id -> {fair, u} for every eligible row: recomputed from the saved oracle quotes (the CSV
    value when that fails; a WARNING when they differ by more than the CSV's rounding)."""
    at = datetime.fromisoformat(str(oracle["fetched_at"]).replace("Z", "+00:00")).timestamp()
    max_age = float(oracle.get("max_age_s") or 10.0)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("eligible") != "True":
            continue
        eid, m = row["exchange_id"], market_map.get(row["exchange_id"])
        csv_fair, csv_u = float(row["fair"]), float(row["u"])
        quotes = (_quote((oracle.get("polymarket") or {}).get(m.polymarket_token), at) if m is not None and m.polymarket_token else None,
                  _quote((oracle.get("kalshi") or {}).get(m.kalshi_ticker), at) if m is not None and m.kalshi_ticker else None)
        fair, _ = stats.oracle_fair(quotes, max_age)
        if fair is None:
            logger.warning("Fair not recomputable from oracle_quotes.json: the eligibility.csv value is used", extra={"exchange_id": eid})
            out[eid] = {"fair": csv_fair, "u": csv_u}
            continue
        u = stats.oracle_uncertainty(quotes, max_age, TICK)
        if abs(fair - csv_fair) > 1e-4 + 1e-9 or abs(u - csv_u) > 1e-4 + 1e-9:
            logger.warning("Recomputed fair / u differ from eligibility.csv", extra={"exchange_id": eid, "fair": fair, "csv_fair": csv_fair, "u": u, "csv_u": csv_u})
        out[eid] = {"fair": fair, "u": u}
    return out


def load_eligibility(run: Path, market_map: Any) -> tuple[dict[str, dict[str, float]], str]:
    with (run / "eligibility.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    oracle = json.loads((run / "oracle_quotes.json").read_text(encoding="utf-8"))
    return eligible_fairs(rows, oracle, market_map), str(oracle.get("fetched_at"))


def load_market_map() -> MarketMap:
    return MarketMap.load()


# ---------------------------------------------------------------------- per print / snapshot


def config_quotes(bid: float | None, ask: float | None, fair: float, u: float, config: tuple[float, str, int]) -> tuple[float | None, float | None]:
    """(YES bid, YES ask) we would show against this touch under one configuration."""
    shift, variant, s_min = config
    if variant == "join":
        return stats.join_quotes(bid, ask, fair + shift, stats.quoter_edge(u), s_min, TICK)
    return stats.improve_quotes(bid, ask, fair + shift, u, s_min, TICK)


class QuoteCache:
    """config_quotes per (touch, configuration) for one market: touches repeat a lot."""

    def __init__(self, fair: float, u: float) -> None:
        self.fair, self.u = fair, u
        self._cache: dict[tuple[Any, Any], tuple[tuple[float | None, float | None], ...]] = {}

    def get(self, bid: float | None, ask: float | None) -> tuple[tuple[float | None, float | None], ...]:
        key = (bid, ask)
        if key not in self._cache:
            self._cache[key] = tuple(config_quotes(bid, ask, self.fair, self.u, c) for c in CONFIGS)
        return self._cache[key]


def reachable_items(rows: Sequence[Mapping[str, Any]], cache: QuoteCache) -> list[list[Mapping[str, Any]]]:
    """Per configuration (CONFIGS order): the reachable prints of one market, in row order."""
    out: list[list[Mapping[str, Any]]] = [[] for _ in CONFIGS]
    for r in rows:
        if r["call"] is None:
            continue
        for k, (quotes, config) in enumerate(zip(cache.get(r["bid"], r["ask"]), CONFIGS)):
            if stats.reaches(r["call"], r["price"], *quotes, exact=config[1] == "join"):
                out[k].append(r)
    return out


def markout(row: Mapping[str, Any], variant: str) -> float | None:
    """300 s markout of a reachable print in price units; IMPROVE = JOIN - 1 tick (APPROXIMATION)."""
    m = row[f"m{HORIZON}"]
    return None if m is None else (m - TICK if variant == "improve" else m)


def verdict(reachable_by_market: Mapping[str, int]) -> dict[str, Any]:
    enough = sorted(eid for eid, n in reachable_by_market.items() if n >= MIN_REACHABLE)
    return {"markets_with_enough": len(enough), "markets": enough,
            "result": "INSUFFICIENT" if len(enough) < MIN_MARKETS else "enough reachable prints for a pilot (feasibility count only, not a test)"}


# ---------------------------------------------------------------------- aggregation


class ReachAnalysis:
    """pass1(market, rows) for every market, finish_pass1(), then pass2(market, rows, quotes) in the
    SAME order. Only aggregates outlive a market."""

    def __init__(self, fairs: Mapping[str, Mapping[str, float]]) -> None:
        self.fairs = fairs
        self.agg = stats.CutAggregator(TICK)
        self._times = [array("d") for _ in CONFIGS]
        self.halves: list[stats.TimeHalves] = []
        self.matched = 0
        self.market_days: set[tuple[str, str]] = set()
        self.matched_day: dict[str, int] = defaultdict(int)
        self.days: list[dict[str, dict[str, Any]]] = [{} for _ in CONFIGS]
        self.markets: list[dict[str, dict[str, Any]]] = [{} for _ in CONFIGS]
        self.snapshots = 0
        self.quotable = [[0, 0, 0] for _ in CONFIGS]  # YES, NO, either

    def _cache(self, market: str) -> QuoteCache:
        return QuoteCache(self.fairs[market]["fair"], self.fairs[market]["u"])

    def pass1(self, market: str, rows: Sequence[Mapping[str, Any]]) -> None:
        for k, items in enumerate(reachable_items(rows, self._cache(market))):
            self._times[k].extend(r["ts"] for r in items)

    def finish_pass1(self) -> None:
        self.halves = [stats.TimeHalves(times) for times in self._times]
        self._times = []

    def pass2(self, market: str, rows: Sequence[Mapping[str, Any]], quotes: Any) -> None:
        cache = self._cache(market)
        self.matched += len(rows)
        market_days = {r["day"] for r in rows}
        self.market_days |= {(market, d) for d in market_days}
        for r in rows:
            self.matched_day[r["day"]] += 1
        for k, (config, items) in enumerate(zip(CONFIGS, reachable_items(rows, cache))):
            variant = config[1]
            groups: dict[tuple[str, str], list[tuple[str, float | None]]] = defaultdict(list)
            m_partials: list[float] = []
            qualifying = 0
            for r in items:
                m = markout(r, variant)
                half = self.halves[k].label(r["ts"])  # once per reachable print, in row order
                for spread in ("all", r["spread"]) if r["spread"] is not None else ("all",):
                    for h in ("all", half):
                        groups[(spread, h)].append((r["day"], m))
                day = self.days[k].setdefault(r["day"], {"reachable": 0, "markets": set(), "upper": []})
                day["reachable"] += 1
                day["markets"].add(market)
                if m is not None:
                    qualifying += 1
                    stats.msum_partials([m / TICK], m_partials)
                    stats.msum_partials([stats.capped_fill(r["size"], FILL_CAP, FILL_SHARE) * m], day["upper"])
            for (spread, h), its in groups.items():
                self.agg.add_market((config, spread, h), market, its)
            self.markets[k][market] = {"matched_prints": len(rows), "reachable": len(items), "days": len(market_days),
                                       "qualifying": qualifying, "mean_ticks": round(math.fsum(m_partials) / qualifying, 3) if qualifying else None}
        q = quotes  # order does not matter for the counts
        self.snapshots += len(q.t)
        for i in range(len(q.t)):
            for k, (yes, ask) in enumerate(cache.get(mq._opt(q.bid[i]), mq._opt(q.ask[i]))):
                self.quotable[k][0] += yes is not None
                self.quotable[k][1] += ask is not None
                self.quotable[k][2] += yes is not None or ask is not None

    # ------------------------------------------------------------ outputs

    def cut_rows(self) -> list[dict[str, Any]]:
        return [{"fair_shift": c[0], "variant": c[1], "s_min": c[2], "spread": spread, "half": half, "horizon_s": HORIZON,
                 **self.agg.result((c, spread, half)), "label": LABEL}
                for c in CONFIGS for spread in SPREADS for half in HALVES]

    def day_rows(self) -> list[dict[str, Any]]:
        return [{"fair_shift": c[0], "variant": c[1], "s_min": c[2], "day": day, "matched_prints": self.matched_day[day],
                 "reachable": self.days[k].get(day, {}).get("reachable", 0), "markets_reachable": len(self.days[k].get(day, {}).get("markets", ())),
                 "upper_bound_susq": round(math.fsum(self.days[k].get(day, {}).get("upper", [])), 2)}
                for k, c in enumerate(CONFIGS) for day in sorted(self.matched_day)]

    def market_rows(self) -> list[dict[str, Any]]:
        out = []
        for k, c in enumerate(CONFIGS):
            for eid, m in sorted(self.markets[k].items()):
                out.append({"fair_shift": c[0], "variant": c[1], "s_min": c[2], "exchange_id": eid, "fair": round(self.fairs[eid]["fair"], 4),
                            "u": round(self.fairs[eid]["u"], 4), **m, "reachable_per_day": round(m["reachable"] / m["days"], 1) if m["days"] else None,
                            "low_reach": m["reachable"] < LOW_REACH})
        return out

    def config_summary(self, k: int) -> dict[str, Any]:
        c = CONFIGS[k]
        reachable = sum(m["reachable"] for m in self.markets[k].values())
        overall = self.agg.result((c, "all", "all"))
        upper = [math.fsum(d["upper"]) for d in self.days[k].values()]
        return {"fair_shift": c[0], "variant": c[1], "s_min": c[2], "matched_prints": self.matched, "reachable": reachable,
                "reachable_share": round(reachable / self.matched, 4) if self.matched else None,
                "reachable_per_market_day": round(reachable / len(self.market_days), 2) if self.market_days else None,
                "snapshots_quotable_share": {name: round(n / self.snapshots, 4) if self.snapshots else None for name, n in zip(("yes", "no", "either"), self.quotable[k])},
                "mean_ticks": overall["mean_ticks"], "ci_low_ticks": overall["ci_low_ticks"], "ci_high_ticks": overall["ci_high_ticks"],
                "upper_bound_susq_total": round(math.fsum(upper), 2), "upper_bound_susq_per_day": round(math.fsum(upper) / len(self.matched_day), 2) if self.matched_day else None}

    def ranking(self, config: tuple[float, str, int] = REFERENCE, n: int = TOP_N) -> dict[str, list[dict[str, Any]]]:
        k = CONFIGS.index(config)
        ranked = sorted(((eid, m) for eid, m in self.markets[k].items() if m["mean_ticks"] is not None), key=lambda x: (-x[1]["mean_ticks"], x[0]))
        reach = {s: self.markets[CONFIGS.index((config[0], config[1], s))] for s in S_MINS}

        def row(eid: str, m: Mapping[str, Any]) -> dict[str, Any]:
            return {"exchange_id": eid, "mean_ticks": m["mean_ticks"], **{f"reachable_s{s}": reach[s][eid]["reachable"] for s in S_MINS},
                    "low_reach": m["reachable"] < LOW_REACH}

        return {"best": [row(eid, m) for eid, m in ranked[:n]], "worst": [row(eid, m) for eid, m in reversed(ranked[-n:])]}

    def verdicts(self) -> dict[str, Any]:
        return {variant: verdict({eid: m["reachable"] for eid, m in self.markets[CONFIGS.index((0.0, variant, 2))].items()}) for variant in VARIANTS}


def run_analysis(tapes: Any, quotes: Any, fairs: Mapping[str, Mapping[str, float]], guard: RssGuard | None = None) -> ReachAnalysis:
    analysis = ReachAnalysis(fairs)
    markets = sorted(eid for eid in fairs if eid in tapes.markets and eid in quotes.markets)
    for eid in markets:
        analysis.pass1(eid, mq.match_market(eid, tapes.markets[eid], quotes.markets[eid]))
        if guard is not None:
            guard.check(f"pass 1 {eid}")
    analysis.finish_pass1()
    if guard is not None:
        guard.stage("pass 1 (reachable times)")
    for eid in markets:
        analysis.pass2(eid, mq.match_market(eid, tapes.markets[eid], quotes.markets[eid]), quotes.markets[eid])
        if guard is not None:
            guard.check(f"pass 2 {eid}")
    if guard is not None:
        guard.stage("pass 2 (aggregates)")
    return analysis


# ---------------------------------------------------------------------- run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: quoter reach and markout on the eligible markets (HYPOTHETICAL, 0 reads).")
    parser.add_argument("--eligibility", type=Path, default=None, help="a quoter_eligibility run dir (default: the newest with oracle_quotes.json)")
    parser.add_argument("--tape", type=Path, nargs="*", default=None, help="tape raw.jsonl files (default: as maker_markout_quotes)")
    parser.add_argument("--snapshots", type=Path, nargs="*", default=None, help="snapshots.jsonl files (default: every quote_snapshots day)")
    parser.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB, help=f"stop cleanly above this peak RSS (default {DEFAULT_MAX_RSS_MB:g})")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    run = args.eligibility or newest_eligibility()
    tape_paths = args.tape or maker.default_raws()
    snap_paths = args.snapshots or sorted((OUT_ROOT / "quote_snapshots").glob("*/snapshots.jsonl"))
    if run is None or not (run / "eligibility.csv").exists() or not (run / "oracle_quotes.json").exists():
        logger.error("Need a quoter_eligibility run with eligibility.csv and oracle_quotes.json: run quoter_eligibility.py --yes first", extra={"run": str(run)})
        return 2
    if not tape_paths or not snap_paths:
        logger.error("Need both a saved tape and quote snapshots", extra={"tape": [str(p) for p in tape_paths], "snapshots": [str(p) for p in snap_paths]})
        return 2
    fairs, oracle_at = load_eligibility(run, load_market_map())
    logger.info("Eligible markets", extra={"run": str(run), "oracle_fetched_at": oracle_at, "markets": len(fairs), "sig_reads": 0, "oracle_reads": 0})
    guard = RssGuard(args.max_rss_mb)
    try:
        tapes = maker.stream_tapes(tape_paths, guard)
        quotes = mq.stream_snapshots(snap_paths, keep=set(fairs), guard=guard)
        guard.stage("inputs loaded")
        analysis = run_analysis(tapes, quotes, fairs, guard)
        out = run_dir(SCRIPT)
        cuts, days, markets = analysis.cut_rows(), analysis.day_rows(), analysis.market_rows()
        configs = [analysis.config_summary(k) for k in range(len(CONFIGS))]
        ranking, verdicts = analysis.ranking(), analysis.verdicts()
        write_csv(out / "by_cut.csv", cuts, CUT_FIELDS)
        write_csv(out / "by_day.csv", days, DAY_FIELDS)
        write_csv(out / "by_market.csv", markets, MARKET_FIELDS)
        write_json(out / "summary.json", {"label": LABEL, "eligibility_run": str(run), "oracle_fetched_at": oracle_at, "eligible_markets": len(fairs),
                                          "markets_analysed": len(analysis.markets[0]), "configs": configs, "ranking": {"config": REFERENCE, **ranking},
                                          "verdict": verdicts, "cuts_reported": len(cuts),
                                          "thresholds": {"min_markets": MIN_MARKETS, "min_reachable_at_s_min_2": MIN_REACHABLE, "low_reach_flag": LOW_REACH},
                                          "upper_bound": f"sum of min({FILL_CAP:g}, {FILL_SHARE} x size) x markout per UTC day, no queue",
                                          "not_modelled": ["queue position", "settlement risk (0 / 1)", "fees", "fair / u drift over the history"]})
        guard.stage("done")
    except MemoryLimitExceeded as exc:
        logger.error("Memory limit reached: stopped cleanly before the OS would kill the process - raise --max-rss-mb only if the machine has room",
                     extra={"stage": exc.stage, "peak_rss_mb": round(exc.peak_mb, 1), "limit_mb": exc.limit_mb})
        return EXIT_MEMORY
    for c in configs:
        logger.info("Reach (HYPOTHETICAL)" if c["fair_shift"] == 0 else "Reach, fair shifted (sensitivity)", extra={k: v for k, v in c.items() if k != "matched_prints"})
    for name in ("best", "worst"):
        for row in ranking[name]:
            logger.info(f"Market, {name} {TOP_N} (JOIN, S_min 2, mean 300 s markout)", extra=row)
    for variant, v in verdicts.items():
        logger.info("Feasibility", extra={"variant": variant, "markets_with_enough": v["markets_with_enough"], "result": v["result"]})
    logger.info(LABEL)
    logger.info("Quoter reach done", extra={"out_dir": str(out), "cuts_reported": len(cuts), "peak_rss_mb": None if guard.peak_mb is None else round(guard.peak_mb, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
