"""OFFLINE research: what would a passive market maker have earned on SIG? An estimate from the
anonymous tape only. HYPOTHETICAL throughout; zero SIG reads by default.

Data: every saved tape (data/edge_scan/tape_patterns/*/raw.jsonl, plus top-ups saved by
--fetch-missing under data/edge_scan/maker_estimate/*/raw.jsonl), de-duplicated by trade id. The
fields the tape really has are logged first. Without an aggressor field the aggressor is inferred
with the tick rule; every output then says "PROXY: tick rule".

Model (a hypothetical maker resting at the print price on the other side):
  - aggressor buy -> the maker SOLD YES at p; aggressor sell -> the maker BOUGHT YES at p;
  - markout at h in {60, 300, 1800} s = maker_sign * (p - ref_h) per share (in price units and
    ticks of 0.005), ref_h = the mean of the LAST buy-aggressor and the LAST sell-aggressor print in
    (t, t + h] (damps bid-ask bounce); a print whose window lacks one side, or runs past the data,
    is dropped and counted (drops lean towards liquid markets: their windows fill more often);
  - fills: min(Q, f * size) for f in {0.10, 0.25, 0.50}, Q in {50, 100, 200}; FEE_PER_SHARE
    (--fee, default 0 - SIG fees are UNVERIFIED);
  - inventory simulation per market in time order, |inventory| <= 300 shares (a side stops
    filling at the cap until it reverses), open inventory marked at the last print of each UTC
    day; max inventory and max drawdown reported. NOT captured: the terminal binary settlement
    risk (0 / 1) of inventory held into resolution.
Statistics by price bucket (<0.15, 0.15-0.85, >0.85, plus "cheap" = <0.15 or >0.85), UTC day,
trade-size quartile and first vs second half (time order): mean maker markout in ticks with a
cluster bootstrap 95% CI (cluster = market-day). The number of cuts reported is counted.
Leader benchmark: the turnover needed for +83,000 at the measured net edge per notional, and
that turnover as a share of the observed tape volume (capacity).

Memory (the VPS has 909 MB and no swap): JSONL is streamed line by line; only trade time / price
/ size / id (and any aggressor field as a 1-byte code) are kept, in compact per-market arrays,
de-duplicated with a set of ids. The analysis runs market by market in two passes - pass 1 keeps
only what needs every market at once (print times for the exact time halves, sizes for the
quartile cuts), pass 2 recomputes each market and folds it into per-group, per-market-day
aggregates (exact partial sums, so means equal math.fsum over all prints; the bootstrap resamples
market-day sums). Per-print rows exist for one market at a time. --max-rss-mb (default 350) checks
the peak RSS between markets and stages and stops cleanly (exit 3) before the OS would kill the
process; the peak is logged per stage.

Pre-registered rule - frozen_rule.json is written (timestamped, with the data scope and a hash
of the inputs) BEFORE any markout or per-half result is computed. PASS only if ALL hold, on the
cheap bucket:
  (a) >= 200 qualifying 300 s prints in >= 15 markets;
  (b) mean maker markout at 300 s >= +0.5 tick with bootstrap CI lower bound > 0 in BOTH halves;
  (c) a positive mean at 300 s on >= 2 UTC days;
  (d) with f = 0.25, Q = 100, inventory cap 300 (cheap prints only) the simulated daily P&L is > 0
      on >= 60% of days.
Below the sample thresholds ((a), or fewer than 2 UTC days) -> INSUFFICIENT DATA; otherwise FAIL.

--fetch-missing N tops up the N most-traded saved exchanges with trades newer than their last
saved one (research.reader: any write raises, RESEARCH_READS_PER_MIN, Retry-After / 429 backoff),
after logging the read estimate, and only with --yes.
-> data/edge_scan/maker_estimate/<UTC date>/: summary.json, by_bucket.csv, by_day.csv,
   sim_daily.csv, frozen_rule.json (+ raw.jsonl for a top-up).

    python scripts/maker_estimate.py [--raw PATH ...] [--fee 0.0] [--max-rss-mb 350] [--fetch-missing N --yes]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import math
import sys
from array import array
from collections import Counter, defaultdict
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
from research.reader import OUT_ROOT, ReadEstimate, ResearchReader, add_research_args, confirm_fetch, file_sha256, iter_jsonl, research_reader, resolve_reads_per_min, run_dir, write_csv, write_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("tape_patterns", ROOT / "scripts" / "tape_patterns.py")
tape = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("tape_patterns", tape)
_spec.loader.exec_module(tape)  # reuse the field lists and the trade page size

logger = logging.getLogger("maker_estimate")

SCRIPT = "maker_estimate"
TICK = 0.005
HORIZONS_S = (60, 300, 1800)
RULE_HORIZON = 300
FILL_RATES = (0.10, 0.25, 0.50)
FILL_CAPS = (50, 100, 200)
INV_CAP = 300
RULE_SCENARIO = (0.25, 100)
LEADER_TARGET = 83_000.0
SETTLEMENT = datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc)
MAX_FETCH_PAGES = 10
DEFAULT_MAX_RSS_MB = 350.0
EXIT_MEMORY = 3
AGGRESSOR_FIELDS = ("aggressor", "aggressorSide", "takerSide")
PROXY = "PROXY: tick rule"
LABEL = "HYPOTHETICAL passive maker estimated from the anonymous tape; no order book, no queue position; SIG fees UNVERIFIED (fee per share = {fee})"
CAVEAT = "Terminal binary settlement risk (inventory resolving at 0 or 1) is NOT captured by a tape-only estimate."
DROP_NOTE = "Dropped prints (no buy and sell print in the window, or past the data) lean towards liquid markets: their windows fill more often."
RULE = {
    "a": "cheap bucket (<0.15 or >0.85): >= 200 qualifying 300 s prints in >= 15 markets",
    "b": "cheap bucket: mean maker markout at 300 s >= +0.5 tick and cluster-bootstrap 95% CI lower bound > 0, in BOTH halves (time order)",
    "c": "cheap bucket: positive mean maker markout at 300 s on >= 2 UTC days",
    "d": "cheap prints only, f = 0.25, Q = 100, inventory cap 300: simulated daily P&L > 0 on >= 60% of days",
    "verdict": "PASS only if a-d all hold; (a) failing or fewer than 2 UTC days -> INSUFFICIENT DATA; otherwise FAIL",
}
BUCKET_DIMS = ("all", "cheap", "price_bucket", "day", "size_quartile", "half")

BUCKET_FIELDS = ("dimension", "group", "horizon_s", "prints", "qualifying", "drop_rate", "markets", "mean_ticks", "ci_low_ticks", "ci_high_ticks", "label")
DAY_FIELDS = ("day", "scope", "horizon_s", "prints", "qualifying", "drop_rate", "markets", "mean_ticks", "ci_low_ticks", "ci_high_ticks", "label")
SIM_FIELDS = ("scope", "fill_rate", "max_fill", "inv_cap", "day", "pnl", "markets", "fills", "shares", "label")


def _ts(raw: Any) -> datetime:
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def is_cheap(price: float) -> bool:
    return price < 0.15 or price > 0.85


# ---------------------------------------------------------------------- streaming tape store


def _aggressor_code(value: Any) -> int:
    v = str(value or "").lower()
    return 1 if v == "buy" else 2 if v == "sell" else 0


@dataclass
class MarketTape:
    """One market's priced trades, compact: time (epoch s), price, size, id and aggressor codes."""

    ts: array = field(default_factory=lambda: array("d"))
    price: array = field(default_factory=lambda: array("d"))
    size: array = field(default_factory=lambda: array("d"))
    ids: list[str] = field(default_factory=list)
    aggressor: dict[str, array] = field(default_factory=lambda: {name: array("b") for name in AGGRESSOR_FIELDS})

    def __len__(self) -> int:
        return len(self.ts)


@dataclass
class TapeStore:
    """Every saved trade, de-duplicated per exchange by id (the first copy is kept), with only the
    fields the analysis needs. Also the field inventory and the data scope of frozen_rule.json."""

    markets: dict[str, MarketTape] = field(default_factory=dict)
    seen: dict[str, set[str]] = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    max_created: dict[str, str] = field(default_factory=dict)
    fields: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    timed_trades: int = 0
    first_trade: str | None = None
    last_trade: str | None = None

    def add(self, eid: str, trade: Mapping[str, Any]) -> None:
        tid = str(trade.get("id"))
        seen = self.seen.setdefault(eid, set())
        if tid in seen:
            return
        seen.add(tid)
        self.counts[eid] += 1
        for key, value in trade.items():
            self.fields[key][type(value).__name__] += 1
        created = trade.get("createdAt")
        if created:
            created = str(created)
            self.timed_trades += 1
            self.first_trade = created if self.first_trade is None or created < self.first_trade else self.first_trade
            self.last_trade = created if self.last_trade is None or created > self.last_trade else self.last_trade
            if eid not in self.max_created or created > self.max_created[eid]:
                self.max_created[eid] = created
        if trade.get("price") is None or not created:
            return
        mt = self.markets.setdefault(eid, MarketTape())
        mt.ts.append(_ts(created).timestamp())
        mt.price.append(float(trade["price"]))
        mt.size.append(abs(float(trade.get("size") or 0)))
        mt.ids.append(tid)
        for name in AGGRESSOR_FIELDS:
            mt.aggressor[name].append(_aggressor_code(trade.get(name)))

    def tape_fields(self) -> dict[str, Any]:
        """Same shape as tape_patterns.tape_fields, from the streamed counters."""
        fields = {k: dict(v) for k, v in sorted(self.fields.items())}
        return {"trades_inspected": sum(self.counts.values()), "fields": fields,
                "aggressor_like": [k for k in tape.AGGRESSOR_LIKE if k in fields], "identity_like": [k for k in tape.IDENTITY_LIKE if k in fields]}


def stream_tapes(paths: Iterable[Path], guard: RssGuard | None = None) -> TapeStore:
    """Read every tape file line by line into a TapeStore (raw records are never kept)."""
    store = TapeStore()
    for p in paths:
        for rec in iter_jsonl(p):
            path = str(rec.get("path") or "")
            if "body" not in rec or not path.startswith("/exchanges/") or not path.endswith("/trades"):
                continue
            eid = path.split("/")[2]
            for trade in (rec["body"] or {}).get("data") or []:
                store.add(eid, trade)
        if guard is not None:
            guard.check(f"loading {p.name}")
    return store


def store_from_trades(tapes: Mapping[str, Sequence[Mapping[str, Any]]]) -> TapeStore:
    """A TapeStore from in-memory trade lists (tests, small inputs)."""
    store = TapeStore()
    for eid, trades in tapes.items():
        for trade in trades:
            store.add(eid, trade)
    return store


def default_raws() -> list[Path]:
    return sorted((OUT_ROOT / "tape_patterns").glob("*/raw.jsonl")) + sorted((OUT_ROOT / SCRIPT).glob("*/raw.jsonl"))


def freeze_rule(out: Path, paths: Sequence[Path], store: TapeStore, fee: float) -> dict[str, Any]:
    """frozen_rule.json - written before any markout or per-half result is computed."""
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "rule": RULE, "parameters": {"tick": TICK, "horizon_s": RULE_HORIZON, "fill_rate": RULE_SCENARIO[0],
              "max_fill": RULE_SCENARIO[1], "inv_cap": INV_CAP, "fee_per_share": fee, "bootstrap": "cluster = market-day, 1000 resamples, seed 7, 95%"},
              "data": {"files": [str(p) for p in paths], "sha256": file_sha256(paths), "markets": len(store.seen), "trades": store.timed_trades,
                       "first_trade": store.first_trade, "last_trade": store.last_trade}}
    write_json(out / "frozen_rule.json", frozen)
    return frozen


# ---------------------------------------------------------------------- one market's prints


def aggressor_calls(trades: Sequence[Mapping[str, Any]]) -> tuple[list[str | None], str]:
    """Aggressor per trade (time order): a real field when the tape has one with buy / sell
    values, else the tick rule on the YES price (labelled PROXY)."""
    for name in AGGRESSOR_FIELDS:
        values = [str(t.get(name) or "").lower() for t in trades]
        if trades and all(v in ("buy", "sell") for v in values):
            return list(values), f"aggressor field '{name}'"
    return stats.tick_rule([float(t["price"]) for t in trades]), PROXY


def market_rows(eid: str, mt: MarketTape, fee: float) -> tuple[list[dict[str, Any]], str]:
    """One market's prints in time order (time, then id): market, time, price, size, aggressor,
    bucket and the maker markout (net of fee) and status per horizon."""
    order = sorted(range(len(mt)), key=lambda i: (mt.ts[i], mt.ids[i]))
    prices = [mt.price[i] for i in order]
    secs = [mt.ts[i] for i in order]
    label = PROXY
    calls: list[str | None] = stats.tick_rule(prices)
    for name in AGGRESSOR_FIELDS:
        codes = mt.aggressor[name]
        if order and all(codes[i] in (1, 2) for i in order):
            calls, label = ["buy" if codes[i] == 1 else "sell" for i in order], f"aggressor field '{name}'"
            break
    marks = {h: stats.maker_markouts(secs, prices, calls, h) for h in HORIZONS_S}
    rows = []
    for k, i in enumerate(order):
        when = datetime.fromtimestamp(secs[k], tz=timezone.utc)
        row = {"market": eid, "t": when, "ts": secs[k], "day": when.strftime("%Y-%m-%d"), "price": prices[k], "size": mt.size[i],
               "call": calls[k], "bucket": stats.price_bucket(prices[k]), "cheap": is_cheap(prices[k])}
        for h in HORIZONS_S:
            m, status = marks[h][k]
            row[f"m{h}"] = None if m is None else m - fee
            row[f"s{h}"] = status
        rows.append(row)
    return rows, label


def build_prints(tapes: Mapping[str, Sequence[Mapping[str, Any]]], fee: float) -> tuple[list[dict[str, Any]], str]:
    """Every market's rows from in-memory trades (tests, small inputs only)."""
    store = store_from_trades(tapes)
    rows, labels = [], set()
    for eid in sorted(store.markets):
        part, label = market_rows(eid, store.markets[eid], fee)
        rows += part
        labels.add(label)
    return rows, PROXY if PROXY in labels or not labels else "; ".join(sorted(labels))


# ---------------------------------------------------------------------- streaming accumulators


class SimAccumulator:
    """The per-market inventory simulation, summed over markets as they come."""

    def __init__(self, fill_rate: float, max_fill: float, cheap_only: bool, fee: float) -> None:
        self.fill_rate, self.max_fill, self.cheap_only, self.fee = fill_rate, max_fill, cheap_only, fee
        self.daily: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "markets": 0, "fills": 0, "shares": 0.0})
        self.max_inv = self.max_dd = 0.0

    def add_market(self, rows: Sequence[Mapping[str, Any]]) -> None:
        prints = [(r["t"], r["price"], r["size"], r["call"] if (r["cheap"] or not self.cheap_only) else None) for r in sorted(rows, key=lambda r: r["t"])]
        sim = stats.simulate_maker_inventory(prints, self.fill_rate, self.max_fill, INV_CAP, self.fee)
        self.max_inv, self.max_dd = max(self.max_inv, sim["max_abs_inventory"]), max(self.max_dd, sim["max_drawdown"])
        for day, d in sim["daily"].items():
            if d["fills"] or d["pnl"]:
                self.daily[day]["pnl"] += d["pnl"]
                self.daily[day]["markets"] += 1
                self.daily[day]["fills"] += d["fills"]
                self.daily[day]["shares"] += d["shares"]

    def result(self) -> dict[str, Any]:
        return {"daily": dict(sorted(self.daily.items())), "max_abs_inventory": self.max_inv, "max_drawdown": round(self.max_dd, 2)}


class LeaderAccumulator:
    """Exact sums for the leader benchmark of one scope."""

    def __init__(self) -> None:
        self.notional: list[float] = []
        self.edge: list[float] = []
        self.observed: list[float] = []
        self.days: set[str] = set()

    def add_market(self, rows: Sequence[Mapping[str, Any]], horizon: int = RULE_HORIZON) -> None:
        ok = [r for r in rows if r[f"m{horizon}"] is not None]
        stats.msum_partials((r["size"] * r["price"] for r in ok), self.notional)
        stats.msum_partials((r["size"] * r[f"m{horizon}"] for r in ok), self.edge)
        stats.msum_partials((r["size"] * r["price"] for r in rows), self.observed)
        self.days.update(r["day"] for r in rows)

    def result(self, now: datetime) -> dict[str, Any]:
        notional = math.fsum(self.notional)
        edge = math.fsum(self.edge) / notional if notional else None
        observed = math.fsum(self.observed)
        days = len(self.days) or 1
        days_left = max((SETTLEMENT - now).total_seconds() / 86400, 0.0)
        out = {"edge_per_notional": None if edge is None else round(edge, 5), "observed_notional": round(observed, 2), "observed_days": days,
               "observed_daily_notional": round(observed / days, 2), "days_left": round(days_left, 2)}
        if edge is None or edge <= 0:
            return {**out, "turnover_needed": None, "share_of_observed_volume": None, "note": "no positive edge: the target is unreachable at this edge"}
        turnover = LEADER_TARGET / edge
        return {**out, "turnover_needed": round(turnover, 2), "share_of_observed_volume": round(turnover / observed, 3) if observed else None,
                "daily_turnover_needed": round(turnover / days_left, 2) if days_left else None,
                "share_of_observed_daily_volume": round(turnover / days_left / (observed / days), 3) if days_left and observed else None}


class MakerAnalysis:
    """Two passes over the markets. pass1(rows) for every market, finish_pass1(), pass2(rows) for
    every market in the SAME order (sorted exchange id), then result(). Keeps aggregates only."""

    def __init__(self, fee: float) -> None:
        self.fee = fee
        self._called_times = array("d")
        self._called_sizes = array("d")
        self._rule_times = array("d")
        self.cuts: list[float] = [0, 0, 0]
        self.half: stats.TimeHalves | None = None
        self.rule_half: stats.TimeHalves | None = None
        self.agg = stats.CutAggregator(TICK)
        self.touched: dict[str, set[str]] = {dim: set() for dim in BUCKET_DIMS}
        self.days: set[str] = set()
        self.markets: set[str] = set()
        self.prints = 0
        self.rule_markets: set[str] = set()
        self.rule_days: set[str] = set()
        self.rule_n = 0
        self.sims = {(scope, f, q): SimAccumulator(f, q, scope == "cheap", fee) for scope in ("all", "cheap") for f in FILL_RATES for q in FILL_CAPS}
        self.scenario = {(f, q, scope): [] for f in FILL_RATES for q in FILL_CAPS for scope in ("all", "cheap")}
        self.leader = {"all": LeaderAccumulator(), "cheap": LeaderAccumulator()}

    def pass1(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for r in rows:
            if r["call"] is not None:
                self._called_times.append(r["t"].timestamp())
                self._called_sizes.append(r["size"])
            if r["cheap"] and r[f"m{RULE_HORIZON}"] is not None:
                self._rule_times.append(r["t"].timestamp())

    def finish_pass1(self) -> None:
        sizes = sorted(self._called_sizes)
        self.cuts = [stats.percentile(sizes, q) for q in (25, 50, 75)] if sizes else [0, 0, 0]
        self.half, self.rule_half = stats.TimeHalves(self._called_times), stats.TimeHalves(self._rule_times)
        self._called_times = self._called_sizes = self._rule_times = array("d")  # freed

    def pass2(self, market: str, rows: Sequence[Mapping[str, Any]]) -> None:
        self.markets.add(market)
        self.prints += len(rows)
        bucket_items: dict[tuple[str, str], list[tuple[str, Any, Any, Any]]] = defaultdict(list)
        day_items: dict[tuple[str, str], list[tuple[str, Any, Any, Any]]] = defaultdict(list)
        rule_items: dict[tuple[str, str], list[tuple[Any, ...]]] = defaultdict(list)
        for r in rows:
            day = r["day"]
            self.days.add(day)
            called = r["call"] is not None
            groups = {"all": "all", "cheap": "cheap" if r["cheap"] else None, "price_bucket": r["bucket"], "day": day,
                      "size_quartile": stats.quartile_label(r["size"], self.cuts), "half": self.half.label(r["t"].timestamp()) if called else None}
            marks = (day, r["m60"], r["m300"], r["m1800"])
            for dim, g in groups.items():
                if g is not None:
                    self.touched[dim].add(g)
                    if called:
                        bucket_items[(dim, g)].append(marks)
            if called:
                day_items[(day, "all")].append(marks)
                if r["cheap"]:
                    day_items[(day, "cheap")].append(marks)
            if r["cheap"] and r[f"m{RULE_HORIZON}"] is not None:
                self.rule_n += 1
                self.rule_markets.add(market)
                self.rule_days.add(day)
                rule_items[("half", self.rule_half.label(r["t"].timestamp()))].append((day, r[f"m{RULE_HORIZON}"], r["t"].timestamp()))
                rule_items[("day", day)].append((day, r[f"m{RULE_HORIZON}"]))
        for (dim, g), items in bucket_items.items():
            for k, h in enumerate(HORIZONS_S, start=1):
                self.agg.add_market(("bucket", dim, g, h), market, [(it[0], it[k]) for it in items])
        for (day, scope), items in day_items.items():
            for k, h in enumerate(HORIZONS_S, start=1):
                self.agg.add_market(("day", day, scope, h), market, [(it[0], it[k]) for it in items])
        for key, items in rule_items.items():  # the halves are time-sorted lists in the old code: clusters in time order
            times = [it[2] for it in items] if key[0] == "half" else None
            self.agg.add_market(("rule",) + key, market, [it[:2] for it in items], times)
        for sim in self.sims.values():
            sim.add_market(rows)
        cheap_rows = [r for r in rows if r["cheap"]]
        for (f, q, scope), partials in self.scenario.items():
            sel = rows if scope == "all" else cheap_rows
            stats.msum_partials((min(q, f * r["size"]) * r[f"m{RULE_HORIZON}"] for r in sel if r[f"m{RULE_HORIZON}"] is not None), partials)
        self.leader["all"].add_market(rows)
        self.leader["cheap"].add_market(cheap_rows)

    # ------------------------------------------------------------------ outputs

    def bucket_rows(self, label: str) -> list[dict[str, Any]]:
        return [{"dimension": dim, "group": g, "horizon_s": h, **self.agg.result(("bucket", dim, g, h)), "label": label}
                for dim in BUCKET_DIMS for g in sorted(self.touched[dim]) for h in HORIZONS_S]

    def day_rows(self, label: str) -> list[dict[str, Any]]:
        return [{"day": day, "scope": scope, "horizon_s": h, **self.agg.result(("day", day, scope, h)), "label": label}
                for day in sorted(self.days) for scope in ("all", "cheap") for h in HORIZONS_S]

    def rule_table(self, sim_rule: Mapping[str, Any]) -> list[dict[str, Any]]:
        halves = {name: self.agg.result(("rule", "half", name)) for name in ("first_half", "second_half")}
        days = {d: self.agg.result(("rule", "day", d))["mean_ticks"] for d in sorted(self.rule_days)}
        return rule_table(self.rule_n, len(self.rule_markets), halves, days, sim_rule)

    def result(self, label: str, now: datetime) -> dict[str, Any]:
        buckets, days = self.bucket_rows(label), self.day_rows(label)
        sims = {key: s.result() for key, s in self.sims.items()}
        sim_rows = [{"scope": scope, "fill_rate": f, "max_fill": q, "inv_cap": INV_CAP, "day": day, "pnl": round(d["pnl"], 2), "markets": d["markets"],
                     "fills": d["fills"], "shares": d["shares"], "label": label} for (scope, f, q), s in sims.items() for day, d in s["daily"].items()]
        summary = {
            "label": LABEL.format(fee=self.fee), "aggressor": label, "caveat": CAVEAT, "drop_note": DROP_NOTE,
            "prints": self.prints, "markets": len(self.markets), "days": sorted(self.days),
            "cuts_reported": len(buckets) + len(days), "hypotheses_in_rule": 4,
            "fill_scenarios_no_inventory_limit": {f"f={f},Q={q}": {scope: round(math.fsum(self.scenario[(f, q, scope)]), 2) for scope in ("all", "cheap")}
                                                  for f in FILL_RATES for q in FILL_CAPS},
            "inventory_simulation": {f"{scope} f={f},Q={q},cap={INV_CAP}": {"total_pnl": round(math.fsum(d["pnl"] for d in s["daily"].values()), 2),
                                                                              "max_abs_inventory": s["max_abs_inventory"], "max_drawdown": s["max_drawdown"]}
                                     for (scope, f, q), s in sims.items()},
            "leader_benchmark": {scope: acc.result(now) for scope, acc in self.leader.items()},
            "rule_table": self.rule_table(sims[("cheap", *RULE_SCENARIO)]),
        }
        return {"buckets": buckets, "days": days, "sim_rows": sim_rows, "summary": summary}


def rule_table(n: int, markets: int, halves: Mapping[str, Mapping[str, Any]], days: Mapping[str, float | None], sim_rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The frozen rule, item by item, with per-half results where the rule has halves."""
    a = n >= 200 and markets >= 15
    b = all(x["mean_ticks"] is not None and x["mean_ticks"] >= 0.5 and x["ci_low_ticks"] is not None and x["ci_low_ticks"] > 0 for x in halves.values())
    c = sum(1 for m in days.values() if m is not None and m > 0) >= 2
    pnl_days = [d["pnl"] for d in sim_rule["daily"].values()]
    d_share = sum(1 for p in pnl_days if p > 0) / len(pnl_days) if pnl_days else None
    d = d_share is not None and d_share >= 0.6
    insufficient = not a or len(days) < 2
    verdict = "INSUFFICIENT DATA" if insufficient else ("PASS" if a and b and c and d else "FAIL")

    def half_text(part: Mapping[str, Any]) -> str:
        return f"{part['mean_ticks']} ticks [{part['ci_low_ticks']}, {part['ci_high_ticks']}] n={part['qualifying']}"

    return [
        {"hypothesis": "(a) sample", "rule": RULE["a"], "first_half": f"{halves['first_half']['prints']} prints", "second_half": f"{halves['second_half']['prints']} prints",
         "result": f"{n} prints in {markets} markets", "holds": a},
        {"hypothesis": "(b) markout 300 s", "rule": RULE["b"], "first_half": half_text(halves["first_half"]), "second_half": half_text(halves["second_half"]),
         "result": "both halves" if b else "not both halves", "holds": b},
        {"hypothesis": "(c) days", "rule": RULE["c"], "first_half": "-", "second_half": "-", "result": f"per-day means {dict(days)}", "holds": c},
        {"hypothesis": "(d) simulation", "rule": RULE["d"], "first_half": "-", "second_half": "-",
         "result": f"positive on {sum(1 for p in pnl_days if p > 0)} of {len(pnl_days)} days ({None if d_share is None else round(d_share, 3)})", "holds": d},
        {"hypothesis": "VERDICT", "rule": RULE["verdict"], "first_half": "", "second_half": "", "result": verdict, "holds": verdict == "PASS"},
    ]


# ---------------------------------------------------------------------- in-memory entry points (tests, small inputs)


def _by_market(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["market"], []).append(r)
    return grouped


def _analysis(rows: Sequence[Mapping[str, Any]], fee: float = 0.0) -> MakerAnalysis:
    grouped = _by_market(rows)
    analysis = MakerAnalysis(fee)
    for part in grouped.values():
        analysis.pass1(part)
    analysis.finish_pass1()
    for market, part in grouped.items():
        analysis.pass2(market, part)
    return analysis


def analyse(rows: Sequence[Mapping[str, Any]], label: str, fee: float, now: datetime) -> dict[str, Any]:
    return _analysis(rows, fee).result(label, now)


def evaluate_rule(rows: Sequence[Mapping[str, Any]], sim_rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _analysis(rows).rule_table(sim_rule)


def simulate(rows: Sequence[Mapping[str, Any]], fill_rate: float, max_fill: float, cheap_only: bool, fee: float) -> dict[str, Any]:
    acc = SimAccumulator(fill_rate, max_fill, cheap_only, fee)
    for part in _by_market(rows).values():
        acc.add_market(part)
    return acc.result()


def scenario_pnl(rows: Sequence[Mapping[str, Any]], fill_rate: float, max_fill: float, horizon: int = RULE_HORIZON) -> float:
    """Sum of fill * markout (no inventory limit) over qualifying prints."""
    return round(math.fsum(min(max_fill, fill_rate * r["size"]) * r[f"m{horizon}"] for r in rows if r[f"m{horizon}"] is not None), 2)


def leader_benchmark(rows: Sequence[Mapping[str, Any]], now: datetime, horizon: int = RULE_HORIZON) -> dict[str, Any]:
    acc = LeaderAccumulator()
    for part in _by_market(rows).values():
        acc.add_market(part, horizon)
    return acc.result(now)


# ---------------------------------------------------------------------- streaming run


def run_analysis(store: TapeStore, fee: float, now: datetime, guard: RssGuard | None = None) -> tuple[dict[str, Any], str]:
    """Market by market, two passes; only aggregates survive a market."""
    analysis = MakerAnalysis(fee)
    labels = set()
    markets = sorted(store.markets)
    for eid in markets:
        rows, label = market_rows(eid, store.markets[eid], fee)
        labels.add(label)
        analysis.pass1(rows)
        del rows
        if guard is not None:
            guard.check(f"pass 1 {eid}")
    analysis.finish_pass1()
    if guard is not None:
        guard.stage("pass 1 (times, sizes)")
    label = PROXY if PROXY in labels or not labels else "; ".join(sorted(labels))
    for eid in markets:
        rows, _ = market_rows(eid, store.markets[eid], fee)
        analysis.pass2(eid, rows)
        del rows
        if guard is not None:
            guard.check(f"pass 2 {eid}")
    if guard is not None:
        guard.stage("pass 2 (aggregates)")
    return analysis.result(label, now), label


# ---------------------------------------------------------------------- top-up (optional reads)


async def fetch_missing(reader: ResearchReader, slug: str, store: TapeStore, n: int) -> int:
    """Trades newer than the last saved one for the n most-traded saved exchanges (saved by the reader)."""
    tid = (await reader.get(f"/tournaments/{slug}")).get("id")
    top = sorted((e for e in store.counts if e in store.max_created), key=lambda e: -store.counts[e])[:n]
    fetched = 0
    for eid in top:
        since = store.max_created[eid]
        cursor = None
        for _ in range(MAX_FETCH_PAGES):
            body = await reader.get(f"/exchanges/{eid}/trades", {"limit": tape.TRADE_PAGE, "cursor": cursor, "tournamentId": tid, "from": since})
            fetched += len(body.get("data") or [])
            pagination = body.get("pagination") or {}
            cursor = pagination.get("nextCursor")
            if not pagination.get("hasMore") or not cursor:
                break
    return fetched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: passive market-making estimate from the saved tape (HYPOTHETICAL).")
    add_research_args(parser)
    parser.add_argument("--raw", type=Path, nargs="*", default=None, help="tape raw.jsonl files (default: every tape_patterns and maker_estimate run)")
    parser.add_argument("--fee", type=float, default=0.0, help="FEE_PER_SHARE (default 0; SIG fees are unverified)")
    parser.add_argument("--fetch-missing", type=int, default=0, help="top up the N most-traded saved exchanges (reads; needs --yes)")
    parser.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB, help=f"stop cleanly above this peak RSS (default {DEFAULT_MAX_RSS_MB:g})")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    paths = args.raw or default_raws()
    if not paths or not all(p.exists() for p in paths):
        logger.error("No saved tape: run tape_patterns.py --yes first, or pass --raw", extra={"raw": [str(p) for p in paths]})
        return 2
    guard = RssGuard(args.max_rss_mb)
    try:
        store = stream_tapes(paths, guard)
        guard.stage("loaded")
        if not store.markets:
            logger.error("No trades in the saved tape", extra={"raw": [str(p) for p in paths]})
            return 2
        logger.info("Tape fields", extra=store.tape_fields())

        if args.fetch_missing > 0:
            n = min(args.fetch_missing, len(store.counts))
            if confirm_fetch(ReadEstimate(1 + n * MAX_FETCH_PAGES, reads_per_min), args.yes):
                async def top_up() -> int:
                    async with research_reader(SCRIPT, reads_per_min) as reader:
                        return await fetch_missing(reader, reader.settings.tournament_slug, store, n)

                logger.info("Top-up done", extra={"new_trades": asyncio.run(top_up())})
                paths = args.raw or default_raws()
                del store
                store = stream_tapes(paths, guard)
                guard.stage("reloaded after top-up")
            else:
                logger.info("Top-up skipped (no --yes): analysing the saved tape only")

        out = run_dir(SCRIPT)
        frozen = freeze_rule(out, paths, store, args.fee)  # BEFORE any markout or per-half result
        logger.info("Rule frozen", extra={"frozen_at": frozen["frozen_at"], "sha256": frozen["data"]["sha256"]})
        result, label = run_analysis(store, args.fee, datetime.now(timezone.utc), guard)

        write_csv(out / "by_bucket.csv", result["buckets"], BUCKET_FIELDS)
        write_csv(out / "by_day.csv", result["days"], DAY_FIELDS)
        write_csv(out / "sim_daily.csv", result["sim_rows"], SIM_FIELDS)
        write_json(out / "summary.json", result["summary"])
        guard.stage("done")
    except MemoryLimitExceeded as exc:
        logger.error("Memory limit reached: stopped cleanly before the OS would kill the process - raise --max-rss-mb only if the machine has room",
                     extra={"stage": exc.stage, "peak_rss_mb": round(exc.peak_mb, 1), "limit_mb": exc.limit_mb})
        return EXIT_MEMORY
    logger.info(CAVEAT)
    logger.info("SIG fees are unverified", extra={"fee_per_share": args.fee})
    for row in result["summary"]["rule_table"]:
        logger.info("Rule", extra={k: row[k] for k in ("hypothesis", "rule", "first_half", "second_half", "result")} | {"aggressor": label})
    logger.info("Maker estimate done (HYPOTHETICAL)", extra={"out_dir": str(out), "prints": result["summary"]["prints"], "cuts_reported": result["summary"]["cuts_reported"],
                                                          "verdict": result["summary"]["rule_table"][-1]["result"], "aggressor": label,
                                                          "peak_rss_mb": None if guard.peak_mb is None else round(guard.peak_mb, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
