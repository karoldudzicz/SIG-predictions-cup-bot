"""OFFLINE research: maker markout against the REAL quote mid (saved snapshots), not the tape.
HYPOTHETICAL throughout; zero reads. NOT modelled: settlement risk (0 / 1), queue position, fees.

Inputs: the saved tape (tape_patterns and maker_estimate raw.jsonl, de-duplicated) and the quote
snapshots of scripts/quote_snapshots.py (data/edge_scan/quote_snapshots/*/snapshots.jsonl).
Only the same markets and time window are used: a print is MATCHED when a quote at most
MAX_QUOTE_AGE_S (45 s = 1.5 polls) old prevails at its time.

  - aggressor by Lee-Ready against the prevailing quote: at or above the ask = buy, at or below the
    bid = sell; inside the spread (or one-sided) -> the tick rule, flagged. How often the tick rule
    and the quote rule disagree (where both give a label) is reported;
  - maker markout at h in {60, 300, 1800} s = maker_sign * (trade price - quote mid at t + h), in
    ticks of 0.005; the mid at t + h must be at most 45 s old and inside the snapshot coverage;
  - mean in ticks with a cluster-bootstrap 95% CI (cluster = market-day), for all and the cheap
    bucket (<0.15 or >0.85), per UTC day, first vs second half (time order) and by quoted spread
    width at the trade (1, 2, >=3 ticks). The number of cuts is reported.

Memory (909 MB VPS, ~1M snapshot lines): both inputs are streamed line by line into compact
per-market arrays (trades: time / price / size / id; snapshots: time / bid / ask / mid, kept only
for markets that also have trades). Two passes market by market; only aggregates (exact partial
sums and one (sum, count) per market-day for the bootstrap) outlive a market. --max-rss-mb
(default 350) stops the run cleanly (exit 3) before the OS would kill it; peak RSS logged per stage.

Pre-registered - frozen_rule.json (timestamped, with the data scope and a hash of the inputs) is
written BEFORE any markout is computed: PASS only if, in the cheap bucket, the mean 300 s markout vs
the quote mid is >= +0.25 tick with a CI lower bound > 0 in BOTH time halves, on >= 150 matched
prints in >= 10 markets over >= 2 UTC days. Below those sample thresholds -> INSUFFICIENT DATA;
otherwise FAIL.
-> data/edge_scan/maker_markout_quotes/<UTC date>/: frozen_rule.json, by_cut.csv, summary.json.

    python scripts/maker_markout_quotes.py [--tape PATH ...] [--snapshots PATH ...] [--max-rss-mb 350]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from array import array
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from research import stats  # noqa: E402
from research.memory import MemoryLimitExceeded, RssGuard  # noqa: E402
from research.reader import OUT_ROOT, file_sha256, iter_jsonl, run_dir, write_csv, write_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("maker_estimate", ROOT / "scripts" / "maker_estimate.py")
maker = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("maker_estimate", maker)
_spec.loader.exec_module(maker)  # reuse the streaming tape store and the bucket helpers

logger = logging.getLogger("maker_markout_quotes")

SCRIPT = "maker_markout_quotes"
TICK = 0.005
HORIZONS_S = (60, 300, 1800)
RULE_HORIZON = 300
MAX_QUOTE_AGE_S = 45.0
DEFAULT_MAX_RSS_MB = 350.0
EXIT_MEMORY = 3
LABEL = "HYPOTHETICAL maker markout vs the quote mid; NOT modelled: settlement risk (0 / 1), queue position, fees"
RULE = {
    "rule": "cheap bucket (<0.15 or >0.85): mean 300 s maker markout vs the quote mid >= +0.25 tick with cluster-bootstrap 95% CI lower bound > 0 in BOTH time halves",
    "sample": ">= 150 matched prints in >= 10 markets over >= 2 UTC days",
    "verdict": "PASS if both; below the sample -> INSUFFICIENT DATA; otherwise FAIL",
}
CUT_DIMS = ("all", "day", "half", "spread")
CUT_FIELDS = ("scope", "dimension", "group", "horizon_s", "prints", "qualifying", "drop_rate", "markets", "mean_ticks", "ci_low_ticks", "ci_high_ticks", "label")
_NAN = float("nan")


def _ts(raw: Any) -> datetime:
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _num(x: Any) -> float:
    return _NAN if x is None else float(x)


def _opt(x: float) -> float | None:
    return None if math.isnan(x) else x


# ---------------------------------------------------------------------- streaming snapshot store


@dataclass
class MarketQuotes:
    t: array = field(default_factory=lambda: array("d"))
    bid: array = field(default_factory=lambda: array("d"))
    ask: array = field(default_factory=lambda: array("d"))
    mid: array = field(default_factory=lambda: array("d"))

    def sorted(self) -> MarketQuotes:
        """Time order, stable (file order for equal times)."""
        order = sorted(range(len(self.t)), key=lambda i: self.t[i])
        return MarketQuotes(*(array("d", (col[i] for i in order)) for col in (self.t, self.bid, self.ask, self.mid)))


@dataclass
class SnapshotStore:
    markets: dict[str, MarketQuotes] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)  # every exchange in the snapshots, kept or not

    def add(self, line: Mapping[str, Any], keep: set[str] | None = None) -> None:
        eid = str(line.get("exchange_id") or "")
        if not eid or not line.get("ts"):
            return
        self.seen.add(eid)
        if keep is not None and eid not in keep:
            return
        q = self.markets.setdefault(eid, MarketQuotes())
        q.t.append(_ts(line["ts"]).timestamp())
        q.bid.append(_num(line.get("bid")))
        q.ask.append(_num(line.get("ask")))
        q.mid.append(_num(line.get("mid")))


def stream_snapshots(paths: Iterable[Path], keep: set[str] | None = None, guard: RssGuard | None = None) -> SnapshotStore:
    store = SnapshotStore()
    for p in paths:
        for line in iter_jsonl(p):
            store.add(line, keep)
        if guard is not None:
            guard.check(f"loading {p.name}")
    return store


def snapshots_from_lines(lines: Iterable[Mapping[str, Any]]) -> SnapshotStore:
    """A SnapshotStore from in-memory lines (tests, small inputs)."""
    store = SnapshotStore()
    for line in lines:
        store.add(line)
    return store


# ---------------------------------------------------------------------- one market


def match_market(eid: str, mt: Any, quotes: MarketQuotes, labels: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """One market's matched prints (time order) with the quote-rule aggressor (tick-rule fallback
    flagged) and the markout vs the quote mid per horizon (price units, in m{h}). `labels` (if given)
    accumulates the label agreement counts."""
    q = quotes.sorted()
    q_times = list(q.t)
    index = list(range(len(q_times)))
    order = sorted(range(len(mt)), key=lambda i: (mt.ts[i], mt.ids[i]))
    prices = [mt.price[i] for i in order]
    ticks = stats.tick_rule(prices)
    rows = []
    for k, i in enumerate(order):
        t = mt.ts[i]
        when = datetime.fromtimestamp(t, tz=timezone.utc)
        j = stats.prevailing_quote(q_times, index, t, MAX_QUOTE_AGE_S)
        if j is None:
            continue
        bid, ask = _opt(q.bid[j]), _opt(q.ask[j])
        quote_call = stats.lee_ready(prices[k], bid, ask)
        if labels is not None:
            if quote_call is not None:
                labels["quote_labels"] += 1
                if ticks[k] is not None:
                    labels["agree" if ticks[k] == quote_call else "disagree"] += 1
            labels["tick_fallback"] += quote_call is None and ticks[k] is not None
        call = quote_call if quote_call is not None else ticks[k]
        sign = stats.maker_sign(call)
        row = {"market": eid, "t": when, "ts": t, "day": when.strftime("%Y-%m-%d"), "price": prices[k], "size": mt.size[i],
               "call": call, "tick_fallback": quote_call is None, "cheap": maker.is_cheap(prices[k]), "spread": stats.spread_bucket(bid, ask),
               "bid": bid, "ask": ask}
        for h in HORIZONS_S:
            jh = stats.prevailing_quote(q_times, index, t + h, MAX_QUOTE_AGE_S) if t + h <= q_times[-1] else None
            mid = None if jh is None else _opt(q.mid[jh])
            row[f"m{h}"] = None if sign is None or mid is None else sign * (prices[k] - mid)
        rows.append(row)
    return rows


def label_summary(labels: Mapping[str, int]) -> dict[str, Any]:
    both = labels["agree"] + labels["disagree"]
    return {"quote_labels": labels["quote_labels"], "tick_fallback": labels["tick_fallback"], "labels_compared": both, "disagree": labels["disagree"],
            "disagree_share": round(labels["disagree"] / both, 4) if both else None}


# ---------------------------------------------------------------------- streaming analysis


class QuoteAnalysis:
    """pass1(rows) for every market, finish_pass1(), pass2(market, rows) in the SAME order, result()."""

    def __init__(self) -> None:
        self._called_times = array("d")
        self._rule_times = array("d")
        self.half: stats.TimeHalves | None = None
        self.rule_half: stats.TimeHalves | None = None
        self.agg = stats.CutAggregator(TICK)
        self.touched: dict[tuple[str, str], set[str]] = {(scope, dim): set() for scope in ("all", "cheap") for dim in CUT_DIMS}
        self.rule_n = 0
        self.rule_markets: set[str] = set()
        self.rule_days: set[str] = set()

    def pass1(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for r in rows:
            if r["call"] is not None:
                self._called_times.append(r["ts"])
                if r["cheap"] and r[f"m{RULE_HORIZON}"] is not None:
                    self._rule_times.append(r["ts"])

    def finish_pass1(self) -> None:
        self.half, self.rule_half = stats.TimeHalves(self._called_times), stats.TimeHalves(self._rule_times)
        self._called_times = self._rule_times = array("d")

    def pass2(self, market: str, rows: Sequence[Mapping[str, Any]]) -> None:
        items: dict[tuple[str, str, str], list[tuple[str, Any, Any, Any]]] = defaultdict(list)
        rule_items: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for r in rows:
            called = r["call"] is not None
            groups = {"all": "all", "day": r["day"], "half": self.half.label(r["ts"]) if called else None, "spread": r["spread"]}
            marks = (r["day"], r["m60"], r["m300"], r["m1800"])
            for scope in ("all", "cheap") if r["cheap"] else ("all",):
                for dim, g in groups.items():
                    if g is not None:
                        self.touched[(scope, dim)].add(g)
                        if called:
                            items[(scope, dim, g)].append(marks)
            if called and r["cheap"] and r[f"m{RULE_HORIZON}"] is not None:
                self.rule_n += 1
                self.rule_markets.add(market)
                self.rule_days.add(r["day"])
                rule_items[self.rule_half.label(r["ts"])].append((r["day"], r[f"m{RULE_HORIZON}"], r["ts"]))
        for (scope, dim, g), its in items.items():
            for k, h in enumerate(HORIZONS_S, start=1):
                self.agg.add_market(("cut", scope, dim, g, h), market, [(it[0], it[k]) for it in its])
        for half, its in rule_items.items():  # the halves are time-sorted lists in the old code: clusters in time order
            self.agg.add_market(("rule", half), market, [it[:2] for it in its], [it[2] for it in its])

    def cuts(self) -> list[dict[str, Any]]:
        return [{"scope": scope, "dimension": dim, "group": g, "horizon_s": h, **self.agg.result(("cut", scope, dim, g, h)), "label": LABEL}
                for scope in ("all", "cheap") for dim in CUT_DIMS for g in sorted(self.touched[(scope, dim)]) for h in HORIZONS_S]

    def verdict(self) -> dict[str, Any]:
        halves = {name: self.agg.result(("rule", name)) for name in ("first_half", "second_half")}
        markets, days = len(self.rule_markets), len(self.rule_days)
        sample_ok = self.rule_n >= 150 and markets >= 10 and days >= 2
        holds = all(x["mean_ticks"] is not None and x["mean_ticks"] >= 0.25 and x["ci_low_ticks"] is not None and x["ci_low_ticks"] > 0 for x in halves.values())
        result = "INSUFFICIENT DATA" if not sample_ok else ("PASS" if holds else "FAIL")
        return {"matched_cheap_prints": self.rule_n, "markets": markets, "utc_days": days, "halves": halves, "result": result}


# ---------------------------------------------------------------------- in-memory entry points (tests, small inputs)


def match_prints(tapes: Any, quotes: SnapshotStore) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every market's matched prints. `tapes`: a maker_estimate.TapeStore or {exchange: [trades]}."""
    store = tapes if isinstance(tapes, maker.TapeStore) else maker.store_from_trades(tapes)
    labels = {"quote_labels": 0, "tick_fallback": 0, "agree": 0, "disagree": 0}
    rows: list[dict[str, Any]] = []
    for eid in sorted(store.markets):
        if eid in quotes.markets:
            rows += match_market(eid, store.markets[eid], quotes.markets[eid], labels)
    return rows, label_summary(labels)


def _analysis(rows: Sequence[Mapping[str, Any]]) -> QuoteAnalysis:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["market"], []).append({**r, "ts": r.get("ts", r["t"].timestamp())})
    analysis = QuoteAnalysis()
    for part in grouped.values():
        analysis.pass1(part)
    analysis.finish_pass1()
    for market, part in grouped.items():
        analysis.pass2(market, part)
    return analysis


def cuts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _analysis(rows).cuts()


def verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _analysis(rows).verdict()


# ---------------------------------------------------------------------- run


def freeze_rule(out: Path, paths: Sequence[Path], tapes: Any, quotes: SnapshotStore) -> dict[str, Any]:
    tape_markets = set(tapes.seen)
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "rule": RULE,
              "parameters": {"tick": TICK, "horizon_s": RULE_HORIZON, "max_quote_age_s": MAX_QUOTE_AGE_S, "bootstrap": "cluster = market-day, 1000 resamples, seed 7, 95%"},
              "data": {"files": [str(p) for p in paths], "sha256": file_sha256(paths), "tape_markets": len(tape_markets), "quoted_markets": len(quotes.seen),
                       "markets_in_both": len(tape_markets & quotes.seen)}}
    write_json(out / "frozen_rule.json", frozen)
    return frozen


def run_analysis(tapes: Any, quotes: SnapshotStore, guard: RssGuard | None = None) -> tuple[QuoteAnalysis, dict[str, Any], int]:
    """Two passes market by market; returns the analysis, the label summary and the matched prints."""
    analysis = QuoteAnalysis()
    labels = {"quote_labels": 0, "tick_fallback": 0, "agree": 0, "disagree": 0}
    markets = [eid for eid in sorted(tapes.markets) if eid in quotes.markets]
    matched = 0
    for eid in markets:
        rows = match_market(eid, tapes.markets[eid], quotes.markets[eid], labels)
        matched += len(rows)
        analysis.pass1(rows)
        del rows
        if guard is not None:
            guard.check(f"pass 1 {eid}")
    analysis.finish_pass1()
    if guard is not None:
        guard.stage("pass 1 (match, times)")
    for eid in markets:
        rows = match_market(eid, tapes.markets[eid], quotes.markets[eid])
        analysis.pass2(eid, rows)
        del rows
        if guard is not None:
            guard.check(f"pass 2 {eid}")
    if guard is not None:
        guard.stage("pass 2 (aggregates)")
    return analysis, label_summary(labels), matched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: maker markout vs the saved quote mid, Lee-Ready aggressor (HYPOTHETICAL, 0 reads).")
    parser.add_argument("--tape", type=Path, nargs="*", default=None, help="tape raw.jsonl files (default: tape_patterns + maker_estimate runs)")
    parser.add_argument("--snapshots", type=Path, nargs="*", default=None, help="snapshots.jsonl files (default: every quote_snapshots day)")
    parser.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB, help=f"stop cleanly above this peak RSS (default {DEFAULT_MAX_RSS_MB:g})")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    tape_paths = args.tape or maker.default_raws()
    snap_paths = args.snapshots or sorted((OUT_ROOT / "quote_snapshots").glob("*/snapshots.jsonl"))
    if not tape_paths or not snap_paths:
        logger.error("Need both a saved tape and quote snapshots", extra={"tape": [str(p) for p in tape_paths], "snapshots": [str(p) for p in snap_paths]})
        return 2
    guard = RssGuard(args.max_rss_mb)
    try:
        tapes = maker.stream_tapes(tape_paths, guard)
        guard.stage("tape loaded")
        quotes = stream_snapshots(snap_paths, keep=set(tapes.markets), guard=guard)
        guard.stage("snapshots loaded")
        out = run_dir(SCRIPT)
        frozen = freeze_rule(out, [*tape_paths, *snap_paths], tapes, quotes)  # BEFORE any markout
        logger.info("Rule frozen", extra={"frozen_at": frozen["frozen_at"], "markets_in_both": frozen["data"]["markets_in_both"]})
        analysis, labels, matched = run_analysis(tapes, quotes, guard)
        table = analysis.cuts()
        result = analysis.verdict()
        write_csv(out / "by_cut.csv", table, CUT_FIELDS)
        summary = {"label": LABEL, "matched_prints": matched, "labels": labels, "cuts_reported": len(table), "rule": RULE, "verdict": result,
                   "not_modelled": ["settlement risk (0 / 1)", "queue position", "fees"]}
        write_json(out / "summary.json", summary)
        guard.stage("done")
    except MemoryLimitExceeded as exc:
        logger.error("Memory limit reached: stopped cleanly before the OS would kill the process - raise --max-rss-mb only if the machine has room",
                     extra={"stage": exc.stage, "peak_rss_mb": round(exc.peak_mb, 1), "limit_mb": exc.limit_mb})
        return EXIT_MEMORY
    logger.info("Aggressor labels: quote rule vs tick rule", extra=labels)
    for name, half in result["halves"].items():
        logger.info("Rule half (HYPOTHETICAL)", extra={"half": name, **half})
    logger.info(LABEL)
    logger.info("Maker markout vs quotes done", extra={"out_dir": str(out), "matched_prints": matched, "cuts_reported": len(table),
                                                      "peak_rss_mb": None if guard.peak_mb is None else round(guard.peak_mb, 1),
                                                      **{k: result[k] for k in ("matched_cheap_prints", "markets", "utc_days", "result")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
