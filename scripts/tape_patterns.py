"""READ-ONLY research D: do recurring counterparties leave patterns in the anonymous tape?

Selection (two stages):
  1. Universe = every exchange of the tournament market list (/tournaments/{slug}/markets,
     kept in raw.jsonl - threshold_ladder_check reuses it). One read per exchange ranks
     activity: /exchanges/{id}/price-history at 1h resolution, limit 1000 (~41 days, the
     whole tournament) -> trades in total, trades in the last 24h, last close.
  2. Deep paging of /exchanges/{id}/trades (200 per page, at most --max-pages) for 40:
     the 25 most traded overall, the 5 most traded in the last 24h, and the 10 most traded
     CHEAP contracts (last close in 0.02-0.15 or 0.85-0.98), none twice; empty quotas are
     filled from the overall ranking.

The tape's fields are logged first (tape_fields.json); nothing assumes an aggressor field.
The aggressor used below is a PROXY from the tick rule, labelled as such everywhere.

Per exchange and window (full history; last 24h), on the tape:
  (1) size concentration: exact sizes with count >= 20 and share >= 5%;
  (2) notional roundness: notional = API `volume` (size * the executed side's price; size *
      YES price when absent) within 0.5% of a multiple of 500 / 1,000 / 5,000 SUSQ, vs the
      share expected by chance (larger notionals look round more often); size * YES price
      is reported next to it. Per exchange and per UTC day -> roundness.csv;
  (3) timing: inter-arrival gaps (whole seconds, >= 1) with an abnormal spike, and
      minutes of the UTC day hit on more days than chance allows (null follows the
      exchange's own hour-of-day activity; Bonferroni over 1440 minutes);
  (4) extremum: for each concentrated size, share of its trades within 1 tick of the running
      UTC-day min / max vs the other trades, with their YES / NO space and proxy side;
  (5) side skew (proxy buy share, YES-space share) by size bucket;
  (6) signed drift of the tape price (a PROXY for the market price) 60 s and 300 s after the
      trades of each pattern from (1)-(3), signed by the proxy aggressor, tested against the
      same exchange's OTHER trades (Welch t): bid-ask bounce makes every trade's signed
      drift negative on average, so 0 is not the baseline.
A pattern is a CANDIDATE only with >= 2 UTC days, >= 20 events and |t| >= 2 with the same
sign in BOTH halves (time order), per horizon. Candidates are frozen to
frozen_candidates.json (timestamped) and re-tested later on data after the freeze
(--retest); only replicated ones count. Every report states how many hypotheses
(exchange x window x pattern x horizon) were tested. Patterns on one exchange are not
independent: the trades right after a recurring participant's prints show up as their own
gap / minute patterns ("echoes") with the same drift.

Reads only, through research.reader (any write raises; RESEARCH_READS_PER_MIN, default 10).
Without --yes only the read estimate is logged.
-> data/edge_scan/tape_patterns/<UTC date>/ (retest: tape_patterns_retest/<UTC date>/).

    python scripts/tape_patterns.py [--yes] [--max-pages 20]
    python scripts/tape_patterns.py --analyse-only [--raw PATH]
    python scripts/tape_patterns.py --retest PATH/frozen_candidates.json [--yes]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, ResearchReader, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min, write_csv, write_json  # noqa: E402

logger = logging.getLogger("tape_patterns")

SCRIPT = "tape_patterns"
RETEST_SCRIPT = "tape_patterns_retest"
MARKETS_PAGE = 100
HISTORY_RESOLUTION, HISTORY_LIMIT = "1h", 1000
TRADE_PAGE = 200
DEFAULT_MAX_PAGES = 20
UNIVERSE_ESTIMATE = 240  # for the read estimate only (the market map has 237)
TOP_ACTIVE, TOP_24H, TOP_CHEAP = 25, 5, 10
CHEAP_RANGES = ((0.02, 0.15), (0.85, 0.98))
WINDOW_24H = timedelta(hours=24)
HORIZONS_S = (60, 300)
MIN_EVENTS, MIN_DAYS, MIN_ABS_T = 20, 2, 2.0
SIZE_MIN_COUNT, SIZE_MIN_SHARE = 20, 0.05
ROUND_MULTIPLES = (500, 1000, 5000)
ROUND_TOL = 0.005
ROUND_PATTERN_EXCESS = 2.0  # a roundness pattern needs >= this x the chance share
TICK = 0.005
PROXY = "tick-rule PROXY (the tape has no aggressor field)"
# Keys that would change the analysis if the tape ever had them (logged, never assumed).
AGGRESSOR_LIKE = ("aggressor", "aggressorSide", "takerSide", "taker", "makerSide", "isBuyerMaker", "liquidity")
IDENTITY_LIKE = ("accountId", "userId", "profileId", "traderId", "makerId", "takerId", "username")

PATTERN_FIELDS = ("exchange_id", "title", "window", "kind", "key", "horizon_s", "members", "events", "days", "mean_drift", "baseline_drift", "excess_drift",
                  "t_all", "t_first", "t_second", "candidate")
EXCHANGE_FIELDS = ("exchange_id", "title", "option", "trades_total", "trades_24h", "last_close", "cheap", "selected", "reason", "pages", "trades_fetched", "complete", "coverage_from")
ROUND_FIELDS = ("exchange_id", "window", "day", "basis", "multiple", "trades", "round", "share", "expected_share", "excess")
EXTREMUM_FIELDS = ("exchange_id", "window", "size", "trades", "near_extreme_share", "baseline_share", "near_yes_space_share", "near_proxy_buy_share")
SKEW_FIELDS = ("exchange_id", "window", "size_bucket", "trades", "proxy_buy_share", "yes_space_share")
CANDIDATE_FIELDS = ("exchange_id", "title", "window", "kind", "key", "horizon_s", "sign", "events", "days", "mean_drift", "baseline_drift", "excess_drift", "t_first", "t_second")
RETEST_FIELDS = ("exchange_id", "kind", "key", "horizon_s", "frozen_sign", "events", "mean_drift", "baseline_drift", "t_all", "verdict")


def _ts(raw: Any) -> datetime:
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------- selection


def universe_from_markets(pages: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """exchange_id -> {title, option, market_id, latest_price} from market-list pages."""
    out: dict[str, dict[str, Any]] = {}
    for page in pages:
        for market in page.get("data") or []:
            for ex in market.get("exchanges") or []:
                out[str(ex["id"])] = {"title": market.get("title"), "option": ex.get("option"), "market_id": str(market.get("id")), "latest_price": ex.get("latestPrice")}
    return out


def activity_from_candles(candles: Sequence[Mapping[str, Any]], as_of: datetime) -> dict[str, Any]:
    """Trades in total and in the 24 h before `as_of` (1h candles), and the last close."""
    ordered = sorted(candles, key=lambda c: str(c.get("time")))
    since = as_of - WINDOW_24H
    closes = [c.get("close") for c in ordered if c.get("close") is not None]
    return {"trades_total": sum(int(c.get("tradeCount") or 0) for c in ordered),
            "trades_24h": sum(int(c.get("tradeCount") or 0) for c in ordered if _ts(c["time"]) >= since),
            "last_close": None if not closes else float(closes[-1])}


def is_cheap(price: float | None) -> bool:
    return price is not None and any(lo <= price <= hi for lo, hi in CHEAP_RANGES)


def select_exchanges(activity: Mapping[str, Mapping[str, Any]], top_active: int = TOP_ACTIVE, top_24h: int = TOP_24H, top_cheap: int = TOP_CHEAP) -> list[tuple[str, str]]:
    """[(exchange_id, reason)]: most traded overall, most traded in 24 h, most traded cheap;
    no exchange twice; an unfilled quota is filled from the overall ranking. Traded only."""
    traded = {eid: a for eid, a in activity.items() if a["trades_total"] > 0}
    overall = sorted(traded, key=lambda e: (-traded[e]["trades_total"], e))
    recent = sorted((e for e in traded if traded[e]["trades_24h"] > 0), key=lambda e: (-traded[e]["trades_24h"], e))
    cheap = [e for e in overall if is_cheap(traded[e]["last_close"])]
    chosen: dict[str, str] = {}

    def take(ranked: Sequence[str], n: int, reason: str) -> int:
        added = 0
        for eid in ranked:
            if added >= n:
                break
            if eid not in chosen:
                chosen[eid] = reason
                added += 1
        return n - added

    short = take(overall, top_active, "active") + take(recent, top_24h, "active_24h") + take(cheap, top_cheap, "cheap")
    take(overall, short, "fill")
    return list(chosen.items())


def tape_fields(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which fields the tape really carries, their value types, and any aggressor / identity
    field (none is assumed)."""
    seen: dict[str, Counter[str]] = defaultdict(Counter)
    for trade in trades:
        for key, value in trade.items():
            seen[key][type(value).__name__] += 1
    fields = {k: dict(v) for k, v in sorted(seen.items())}
    return {"trades_inspected": len(trades), "fields": fields,
            "aggressor_like": [k for k in AGGRESSOR_LIKE if k in fields], "identity_like": [k for k in IDENTITY_LIKE if k in fields]}


# ---------------------------------------------------------------------- the series


@dataclass
class Series:
    """One exchange's trades in time order (priced trades only)."""

    times: list[datetime]
    secs: list[float]
    prices: list[float]
    sizes: list[float]
    spaces: list[str | None]
    notional_exec: list[float]
    notional_yes: list[float]
    proxy: list[str | None] = field(default_factory=list)

    @classmethod
    def from_trades(cls, trades: Sequence[Mapping[str, Any]]) -> Series:
        rows = sorted((t for t in trades if t.get("price") is not None and t.get("createdAt")), key=lambda t: (str(t["createdAt"]), str(t.get("id"))))
        times = [_ts(t["createdAt"]) for t in rows]
        prices = [float(t["price"]) for t in rows]
        sizes = [float(t.get("size") or 0) for t in rows]
        spaces = [t.get("side") for t in rows]
        exec_ = [float(t["volume"]) if t.get("volume") is not None else s * (p if sp != "NO" else 1 - p) for t, s, p, sp in zip(rows, sizes, prices, spaces)]
        return cls(times, [t.timestamp() for t in times], prices, sizes, spaces, exec_, [s * p for s, p in zip(sizes, prices)], stats.tick_rule(prices))

    def subset(self, start: datetime) -> Series:
        i = next((k for k, t in enumerate(self.times) if t >= start), len(self.times))
        return Series(self.times[i:], self.secs[i:], self.prices[i:], self.sizes[i:], self.spaces[i:], self.notional_exec[i:], self.notional_yes[i:], stats.tick_rule(self.prices[i:]))

    def __len__(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class Pattern:
    kind: str  # size / notional / gap / minute
    key: str
    members: tuple[int, ...]


def gaps_s(series: Series) -> list[int | None]:
    """Gap to the previous trade in whole seconds (None for the first)."""
    return [None] + [round(b - a) for a, b in zip(series.secs, series.secs[1:])]


def members_of(series: Series, kind: str, key: str) -> list[int]:
    """Trade indices matching a pattern definition (used for detection AND for the retest)."""
    if kind == "size":
        return [i for i, s in enumerate(series.sizes) if s == float(key)]
    if kind == "notional":
        return [i for i, n in enumerate(series.notional_exec) if stats.is_round_notional(n, float(key), ROUND_TOL)]
    if kind == "gap":
        return [i for i, g in enumerate(gaps_s(series)) if g is not None and g == int(key)]
    if kind == "minute":
        return [i for i, t in enumerate(series.times) if f"{t.hour:02d}:{t.minute:02d}" == key]
    raise ValueError(f"unknown pattern kind {kind!r}")


def size_key(size: float) -> str:
    """Exact text key of a size (no exponent notation, so a retest matches it again)."""
    return str(int(size)) if float(size).is_integer() else repr(float(size))


def detect_patterns(series: Series) -> list[Pattern]:
    found = [Pattern("size", size_key(c["size"]), tuple(members_of(series, "size", size_key(c["size"]))))
             for c in stats.size_concentration(series.sizes, SIZE_MIN_COUNT, SIZE_MIN_SHARE)]
    for multiple in ROUND_MULTIPLES:
        members = members_of(series, "notional", str(multiple))
        expected = stats.expected_round_share(series.notional_exec, multiple, ROUND_TOL)
        if len(members) >= MIN_EVENTS and expected is not None and len(members) / len(series) >= ROUND_PATTERN_EXCESS * max(expected, 1e-9):
            found.append(Pattern("notional", str(multiple), tuple(members)))
    for spike in stats.gap_spikes([g for g in gaps_s(series) if g is not None]):
        found.append(Pattern("gap", str(spike["gap_s"]), tuple(members_of(series, "gap", str(spike["gap_s"])))))
    for minute in stats.recurring_minutes(series.times)["minutes"]:
        found.append(Pattern("minute", minute["minute"], tuple(members_of(series, "minute", minute["minute"]))))
    return found


def signed_drifts(series: Series, horizon_s: float) -> list[float | None]:
    """Per trade: tape-price drift after `horizon_s`, signed by the proxy aggressor (+ for
    a proxy buy). None without a proxy call or when the data ends before the horizon."""
    out: list[float | None] = []
    for i, call in enumerate(series.proxy):
        direction = {"buy": 1.0, "sell": -1.0}.get(call or "")
        drift = stats.drift_after(series.secs, series.prices, i, horizon_s)
        out.append(None if direction is None or drift is None else direction * drift)
    return out


def evaluate(series: Series, members: Sequence[int], horizon_s: float, signed: Sequence[float | None] | None = None) -> dict[str, Any]:
    """Signed drift after the pattern's trades vs the SAME exchange's other trades in the
    same window (Welch t): signed drift is negative on average for every trade (bid-ask
    bounce), so 0 is not the baseline. Halves split the pattern's events by time; each half
    is compared with the other trades of the same time span."""
    signed = signed_drifts(series, horizon_s) if signed is None else signed
    inside = set(members)
    events = [(series.times[i], signed[i]) for i in members if signed[i] is not None]
    base = [(series.times[i], signed[i]) for i in range(len(series)) if i not in inside and signed[i] is not None]
    first, second = stats.split_halves(events, key=lambda e: e[0])
    cut = second[0][0] if second else None
    base_first = [v for t, v in base if cut is None or t < cut]
    base_second = [v for t, v in base if cut is not None and t >= cut]
    values, base_values = [v for _, v in events], [v for _, v in base]
    mean = None if not values else math.fsum(values) / len(values)
    base_mean = None if not base_values else math.fsum(base_values) / len(base_values)
    by_day = stats.by_utc_day(events, ts=lambda e: e[0])
    base_day = stats.by_utc_day(base, ts=lambda e: e[0])
    return {"members": len(members), "events": len(events), "days": len(by_day),
            "mean_drift": None if mean is None else round(mean, 5), "baseline_drift": None if base_mean is None else round(base_mean, 5),
            "excess_drift": None if mean is None or base_mean is None else round(mean - base_mean, 5),
            "t_all": stats.welch_t(values, base_values),
            "t_first": stats.welch_t((v for _, v in first), base_first), "t_second": stats.welch_t((v for _, v in second), base_second),
            "per_day": {d: {"events": len(e), "mean_drift": round(math.fsum(v for _, v in e) / len(e), 5),
                            "baseline_drift": None if not base_day.get(d) else round(math.fsum(v for _, v in base_day[d]) / len(base_day[d]), 5)}
                        for d, e in by_day.items()}}


def is_candidate(ev: Mapping[str, Any]) -> bool:
    t1, t2 = ev["t_first"], ev["t_second"]
    return (ev["events"] >= MIN_EVENTS and ev["days"] >= MIN_DAYS and t1 is not None and t2 is not None
            and abs(t1) >= MIN_ABS_T and abs(t2) >= MIN_ABS_T and (t1 > 0) == (t2 > 0))


# ---------------------------------------------------------------------- descriptive reports


def roundness_rows(eid: str, window: str, series: Series) -> list[dict[str, Any]]:
    rows = []
    days = stats.by_utc_day(range(len(series)), ts=lambda i: series.times[i])
    for day, idx in [("ALL", list(range(len(series))))] + list(days.items()):
        for basis, values in (("exec_volume", series.notional_exec), ("size_x_yes_price", series.notional_yes)):
            vals = [values[i] for i in idx]
            for multiple in ROUND_MULTIPLES:
                hits = sum(1 for v in vals if stats.is_round_notional(v, multiple, ROUND_TOL))
                expected = stats.expected_round_share(vals, multiple, ROUND_TOL)
                share = round(hits / len(vals), 4) if vals else None
                rows.append({"exchange_id": eid, "window": window, "day": day, "basis": basis, "multiple": multiple, "trades": len(vals), "round": hits,
                             "share": share, "expected_share": None if expected is None else round(expected, 4),
                             "excess": None if not expected or share is None else round(share / expected, 2)})
    return rows


def extremum_rows(eid: str, window: str, series: Series, sizes: Sequence[float]) -> list[dict[str, Any]]:
    near = stats.near_running_day_extreme(series.times, series.prices, TICK)
    rows = []
    for size in sizes:
        mine = [i for i, s in enumerate(series.sizes) if s == size and near[i] is not None]
        other = [i for i, s in enumerate(series.sizes) if s != size and near[i] is not None]
        hit = [i for i in mine if near[i]]
        called = [series.proxy[i] for i in hit if series.proxy[i] is not None]
        rows.append({"exchange_id": eid, "window": window, "size": size, "trades": len(mine),
                     "near_extreme_share": round(len(hit) / len(mine), 4) if mine else None,
                     "baseline_share": round(sum(1 for i in other if near[i]) / len(other), 4) if other else None,
                     "near_yes_space_share": round(sum(1 for i in hit if series.spaces[i] == "YES") / len(hit), 4) if hit else None,
                     "near_proxy_buy_share": round(called.count("buy") / len(called), 4) if called else None})
    return rows


def skew_rows(eid: str, window: str, series: Series) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(series.sizes):
        groups[stats.size_bucket(s)].append(i)
    rows = []
    for _, label in stats.SIZE_BUCKETS:
        idx = groups.get(label, [])
        called = [series.proxy[i] for i in idx if series.proxy[i] is not None]
        rows.append({"exchange_id": eid, "window": window, "size_bucket": label, "trades": len(idx),
                     "proxy_buy_share": round(called.count("buy") / len(called), 4) if called else None,
                     "yes_space_share": round(sum(1 for i in idx if series.spaces[i] == "YES") / len(idx), 4) if idx else None})
    return rows


# ---------------------------------------------------------------------- analysis


def windows_of(series: Series, as_of: datetime) -> dict[str, Series]:
    return {"full": series, "24h": series.subset(as_of - WINDOW_24H)}


def analyse(tapes: Mapping[str, Sequence[Mapping[str, Any]]], titles: Mapping[str, str | None], as_of: datetime) -> dict[str, Any]:
    patterns, candidates, by_day, rounds, extremes, skews = [], [], [], [], [], []
    for eid in sorted(tapes):
        full = Series.from_trades(tapes[eid])
        for window, series in windows_of(full, as_of).items():
            if len(series) < 2:
                continue
            rounds += roundness_rows(eid, window, series)
            skews += skew_rows(eid, window, series)
            found = detect_patterns(series)
            extremes += extremum_rows(eid, window, series, [float(p.key) for p in found if p.kind == "size"])
            signed = {h: signed_drifts(series, h) for h in HORIZONS_S} if found else {}
            for pattern in found:
                for h in HORIZONS_S:
                    ev = evaluate(series, pattern.members, h, signed[h])
                    row = {"exchange_id": eid, "title": titles.get(eid), "window": window, "kind": pattern.kind, "key": pattern.key, "horizon_s": h,
                           **{k: ev[k] for k in ("members", "events", "days", "mean_drift", "baseline_drift", "excess_drift", "t_all", "t_first", "t_second")},
                           "candidate": is_candidate(ev)}
                    patterns.append(row)
                    if row["candidate"]:
                        candidates.append({**row, "sign": 1 if ev["t_first"] > 0 else -1})
                        by_day += [{"exchange_id": eid, "window": window, "kind": pattern.kind, "key": pattern.key, "horizon_s": h, "day": d, **v} for d, v in ev["per_day"].items()]
    return {"patterns": patterns, "candidates": candidates, "candidates_by_day": by_day, "roundness": rounds, "extremum": extremes, "side_skew": skews,
            "hypotheses_tested": len(patterns), "exchanges": len(tapes)}


def freeze(candidates: Sequence[Mapping[str, Any]], data_through: datetime, hypotheses: int) -> dict[str, Any]:
    return {"frozen_at": _iso(datetime.now(timezone.utc)), "data_through": _iso(data_through), "hypotheses_tested": hypotheses,
            "rule": f"events >= {MIN_EVENTS}, UTC days >= {MIN_DAYS}, |t| >= {MIN_ABS_T:g} same sign in both halves; replicate: same sign, |t| >= {MIN_ABS_T:g}, events >= {MIN_EVENTS} on data after data_through",
            "aggressor": PROXY,
            "candidates": [{k: c[k] for k in ("exchange_id", "window", "kind", "key", "horizon_s", "sign", "events", "t_first", "t_second")} for c in candidates]}


def retest(frozen: Mapping[str, Any], tapes: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Each frozen candidate on trades after `data_through` only."""
    since = _ts(frozen["data_through"])
    rows = []
    for c in frozen.get("candidates") or []:
        series = Series.from_trades([t for t in tapes.get(c["exchange_id"], []) if _ts(t["createdAt"]) > since])
        ev = evaluate(series, members_of(series, c["kind"], c["key"]), c["horizon_s"]) if len(series) else {"events": 0, "mean_drift": None, "baseline_drift": None, "t_all": None}
        t = ev["t_all"]
        if ev["events"] < MIN_EVENTS:
            verdict = "INSUFFICIENT DATA"
        elif t is not None and abs(t) >= MIN_ABS_T and (t > 0) == (c["sign"] > 0):
            verdict = "REPLICATED"
        else:
            verdict = "FAILED"
        rows.append({"exchange_id": c["exchange_id"], "kind": c["kind"], "key": c["key"], "horizon_s": c["horizon_s"], "frozen_sign": c["sign"],
                     "events": ev["events"], "mean_drift": ev["mean_drift"], "baseline_drift": ev["baseline_drift"], "t_all": t, "verdict": verdict})
    return rows


# ---------------------------------------------------------------------- raw.jsonl -> inputs


def inputs_from_raw(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Universe, latest candles per exchange, trades per exchange (de-duplicated), as_of."""
    ok = [r for r in records if "body" in r]
    markets = [r["body"] for r in ok if re.fullmatch(r"/tournaments/[^/]+/markets", str(r.get("path")))]
    candles: dict[str, list[dict[str, Any]]] = {}
    trades: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in ok:
        path = str(r.get("path"))
        if path.endswith("/price-history"):
            candles[path.split("/")[2]] = (r["body"] or {}).get("candles") or []  # the latest read wins
        elif path.startswith("/exchanges/") and path.endswith("/trades"):
            for t in (r["body"] or {}).get("data") or []:
                trades[path.split("/")[2]][str(t.get("id"))] = t
    stamps = [_ts(r["ts"]) for r in ok if r.get("ts")]
    return {"universe": universe_from_markets(markets), "candles": candles, "trades": {e: list(t.values()) for e, t in trades.items()},
            "as_of": max(stamps) if stamps else datetime.now(timezone.utc)}


def write_outputs(out: Path, result: Mapping[str, Any], exchange_rows: Sequence[Mapping[str, Any]], fields_seen: Mapping[str, Any], as_of: datetime) -> None:
    write_json(out / "tape_fields.json", fields_seen)
    write_csv(out / "exchanges.csv", exchange_rows, EXCHANGE_FIELDS)
    write_csv(out / "patterns.csv", result["patterns"], PATTERN_FIELDS)
    write_csv(out / "candidates.csv", result["candidates"], CANDIDATE_FIELDS)
    write_csv(out / "candidates_by_day.csv", result["candidates_by_day"], ("exchange_id", "window", "kind", "key", "horizon_s", "day", "events", "mean_drift", "baseline_drift"))
    write_csv(out / "roundness.csv", result["roundness"], ROUND_FIELDS)
    write_csv(out / "extremum.csv", result["extremum"], EXTREMUM_FIELDS)
    write_csv(out / "side_skew.csv", result["side_skew"], SKEW_FIELDS)
    write_json(out / "frozen_candidates.json", freeze(result["candidates"], as_of, result["hypotheses_tested"]))
    summary = {"as_of": _iso(as_of), "exchanges_analysed": result["exchanges"], "hypotheses_tested": result["hypotheses_tested"],
               "patterns_by_kind": dict(Counter(p["kind"] for p in result["patterns"] if p["horizon_s"] == HORIZONS_S[0])),
               "candidates": len(result["candidates"]),
               "echo_note": "patterns on one exchange are not independent: trades right after a recurring participant's prints form their own gap / minute patterns that carry the same drift - read a gap or minute candidate together with the size / notional candidates of the same exchange",
               "expected_false_candidates_note": "each hypothesis passes |t| >= 2 in both halves by chance with roughly 0.05 * 0.05 / 2 ~ 0.1% (independent halves, same sign) - compare with hypotheses_tested",
               "aggressor": PROXY, "drift": "tape price series as a PROXY for the market price; t = Welch t of a pattern's signed drift vs the same exchange's other trades (bid-ask bounce makes every trade's signed drift negative on average)",
               "tape_fields": {k: fields_seen.get(k) for k in ("aggressor_like", "identity_like")}}
    write_json(out / "summary.json", summary)
    logger.info("Tape patterns done", extra=summary)
    for c in result["candidates"]:
        logger.info("Candidate (to be re-tested after freezing)", extra={k: c.get(k) for k in CANDIDATE_FIELDS})


def exchange_table(universe: Mapping[str, Mapping[str, Any]], activity: Mapping[str, Mapping[str, Any]], selected: Sequence[tuple[str, str]],
                   fetched: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    reasons = dict(selected)
    rows = []
    for eid, a in sorted(activity.items(), key=lambda kv: -kv[1]["trades_total"]):
        u = universe.get(eid) or {}
        f = fetched.get(eid) or {}
        rows.append({"exchange_id": eid, "title": u.get("title"), "option": u.get("option"), **a, "cheap": is_cheap(a["last_close"]),
                     "selected": eid in reasons, "reason": reasons.get(eid), **{k: f.get(k) for k in ("pages", "trades_fetched", "complete", "coverage_from")}})
    return rows


# ---------------------------------------------------------------------- reads


async def fetch_trades(reader: ResearchReader, eid: str, tournament_id: str | None, max_pages: int, since: str | None = None) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    cursor, pages, complete = None, 0, False
    while pages < max_pages:
        body = await reader.get(f"/exchanges/{eid}/trades", {"limit": TRADE_PAGE, "cursor": cursor, "tournamentId": tournament_id, "from": since})
        pages += 1
        trades.extend(body.get("data") or [])
        pagination = body.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            complete = True
            break
    oldest = min((str(t.get("createdAt")) for t in trades), default=None)
    return {"trades": trades, "pages": pages, "trades_fetched": len(trades), "complete": complete, "coverage_from": oldest}


async def run(reader: ResearchReader, slug: str, max_pages: int) -> dict[str, Any]:
    tournament = await reader.get(f"/tournaments/{slug}")
    tid = tournament.get("id")
    pages: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = await reader.get(f"/tournaments/{slug}/markets", {"limit": MARKETS_PAGE, "cursor": cursor, "status": "any"})
        pages.append(page)
        pagination = page.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            break
    universe = universe_from_markets(pages)
    logger.info("Universe", extra={"markets": sum(len(p.get("data") or []) for p in pages), "exchanges": len(universe)})
    as_of = datetime.now(timezone.utc)
    activity = {}
    for eid in sorted(universe):
        body = await reader.get(f"/exchanges/{eid}/price-history", {"resolution": HISTORY_RESOLUTION, "limit": HISTORY_LIMIT, "tournamentId": tid})
        activity[eid] = activity_from_candles(body.get("candles") or [], as_of)
    selected = select_exchanges(activity)
    planned = sum(min(max_pages, max(1, math.ceil(activity[e]["trades_total"] / TRADE_PAGE))) for e, _ in selected)
    logger.info("Deep paging plan", extra={"exchanges": len(selected), "pages": planned, "est_min": round(planned * 60 / reader.reads_per_min, 1)})
    fetched: dict[str, dict[str, Any]] = {}
    fields_logged = False
    for eid, reason in selected:
        fetched[eid] = await fetch_trades(reader, eid, tid, max_pages)
        if not fields_logged and fetched[eid]["trades"]:
            seen = tape_fields(fetched[eid]["trades"])
            logger.info("Tape fields (first deep page)", extra=seen)
            if seen["aggressor_like"] or seen["identity_like"]:
                logger.warning("The tape carries aggressor / identity fields - the analysis still uses only the tick-rule proxy", extra=seen)
            fields_logged = True
    tapes = {e: f["trades"] for e, f in fetched.items()}
    result = analyse(tapes, {e: (universe.get(e) or {}).get("title") for e in tapes}, as_of)
    write_outputs(reader.out_dir, result, exchange_table(universe, activity, selected, fetched), tape_fields([t for ts in tapes.values() for t in ts]), as_of)
    return result


async def run_retest(reader: ResearchReader, slug: str, frozen: Mapping[str, Any], max_pages: int) -> list[dict[str, Any]]:
    tid = (await reader.get(f"/tournaments/{slug}")).get("id")
    tapes = {}
    for eid in sorted({c["exchange_id"] for c in frozen.get("candidates") or []}):
        tapes[eid] = (await fetch_trades(reader, eid, tid, max_pages, since=frozen["data_through"]))["trades"]
    rows = retest(frozen, tapes)
    write_csv(reader.out_dir / "retest.csv", rows, RETEST_FIELDS)
    summary = {"frozen_at": frozen.get("frozen_at"), "data_through": frozen.get("data_through"), "candidates": len(rows),
               "verdicts": dict(Counter(r["verdict"] for r in rows)), "hypotheses_tested_at_freeze": frozen.get("hypotheses_tested"), "aggressor": PROXY}
    write_json(reader.out_dir / "retest.json", {**summary, "rows": rows})
    logger.info("Retest done", extra=summary)
    return rows


# ---------------------------------------------------------------------- CLI


def _analyse_raw(raw: Path) -> int:
    inp = inputs_from_raw(read_jsonl(raw))
    activity = {e: activity_from_candles(c, inp["as_of"]) for e, c in inp["candles"].items()}
    selected = select_exchanges(activity)
    tapes = {e: inp["trades"].get(e, []) for e, _ in selected}
    result = analyse(tapes, {e: (inp["universe"].get(e) or {}).get("title") for e in tapes}, inp["as_of"])
    fetched = {e: {"trades_fetched": len(t)} for e, t in tapes.items()}
    out = raw.parent / "reanalysis"
    write_outputs(out, result, exchange_table(inp["universe"], activity, selected, fetched), tape_fields([t for ts in tapes.values() for t in ts]), inp["as_of"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: recurring patterns in the anonymous SIG tape.")
    add_research_args(parser)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help=f"trade pages per selected exchange (default {DEFAULT_MAX_PAGES})")
    parser.add_argument("--analyse-only", action="store_true", help="re-analyse a raw.jsonl; no API request")
    parser.add_argument("--raw", type=Path, default=None, help="raw.jsonl for --analyse-only (default: the newest tape_patterns run)")
    parser.add_argument("--retest", type=Path, default=None, help="frozen_candidates.json to re-test on the trades after its freeze")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    if args.analyse_only:
        raw = args.raw or (sorted((OUT_ROOT / SCRIPT).glob("*/raw.jsonl")) or [None])[-1]
        if raw is None or not Path(raw).exists():
            logger.error("No tape_patterns raw.jsonl found: pass --raw", extra={"raw": str(raw)})
            return 2
        return _analyse_raw(Path(raw))
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.max_pages <= 200:
        parser.error("--max-pages must be in [1, 200]")
    top = TOP_ACTIVE + TOP_24H + TOP_CHEAP
    if args.retest:
        frozen = json.loads(args.retest.read_text(encoding="utf-8"))
        exchanges = len({c["exchange_id"] for c in frozen.get("candidates") or []})
        if not exchanges:
            logger.info("No frozen candidates: nothing to re-test", extra={"file": str(args.retest)})
            return 0
        if not confirm_fetch(ReadEstimate(1 + exchanges * args.max_pages, reads_per_min), args.yes):
            return 0

        async def go() -> None:
            async with research_reader(RETEST_SCRIPT, reads_per_min) as reader:
                await run_retest(reader, reader.settings.tournament_slug, frozen, args.max_pages)

        asyncio.run(go())
        return 0
    # Upper bound: the universe size is only known after the market list.
    reads = 1 + math.ceil(UNIVERSE_ESTIMATE / MARKETS_PAGE) + UNIVERSE_ESTIMATE + top * args.max_pages
    if not confirm_fetch(ReadEstimate(reads, reads_per_min), args.yes):
        return 0

    async def go_run() -> None:
        async with research_reader(SCRIPT, reads_per_min) as reader:
            await run(reader, reader.settings.tournament_slug, args.max_pages)

    asyncio.run(go_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
