"""Regression: the streaming, memory-bounded maker_estimate / maker_markout_quotes reproduce the old
row-based results EXACTLY on a synthetic dataset; plus the peak-RSS guard.

The reference implementation below is a trimmed copy of the pre-streaming code (whole tape and
snapshot files in memory, one dict per print, cut() over lists of rows).

Run from the project root:  python -m pytest tests/test_memory_bounded.py -v
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research import stats
from research.memory import MemoryLimitExceeded, RssGuard, peak_rss_mb
from research.reader import read_jsonl

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("maker_estimate")
MQ = _load("maker_markout_quotes")
TAPE = M.tape
NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
T0 = datetime(2026, 10, 20, 9, tzinfo=timezone.utc)


# ---------------------------------------------------------------------- reference (the old, row-based code)


def ref_ts(raw):
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def ref_build_prints(tapes, fee):
    rows, labels = [], set()
    for eid, trades in sorted(tapes.items()):
        ordered = sorted((t for t in trades if t.get("price") is not None and t.get("createdAt")), key=lambda t: (str(t["createdAt"]), str(t.get("id"))))
        if not ordered:
            continue
        calls, label = M.aggressor_calls(ordered)
        labels.add(label)
        times = [ref_ts(t["createdAt"]) for t in ordered]
        secs = [t.timestamp() for t in times]
        prices = [float(t["price"]) for t in ordered]
        marks = {h: stats.maker_markouts(secs, prices, calls, h) for h in M.HORIZONS_S}
        for i, t in enumerate(ordered):
            row = {"market": eid, "t": times[i], "day": times[i].strftime("%Y-%m-%d"), "price": prices[i], "size": abs(float(t.get("size") or 0)),
                   "call": calls[i], "bucket": stats.price_bucket(prices[i]), "cheap": M.is_cheap(prices[i])}
            for h in M.HORIZONS_S:
                m, status = marks[h][i]
                row[f"m{h}"] = None if m is None else m - fee
                row[f"s{h}"] = status
            rows.append(row)
    return rows, M.PROXY if M.PROXY in labels or not labels else "; ".join(sorted(labels))


def ref_cut(rows, horizon):
    called = [r for r in rows if r["call"] is not None]
    ok = [r for r in called if r[f"m{horizon}"] is not None]
    mean, lo, hi = stats.cluster_bootstrap_mean([r[f"m{horizon}"] / M.TICK for r in ok], [(r["market"], r["day"]) for r in ok])
    return {"prints": len(called), "qualifying": len(ok), "drop_rate": round(1 - len(ok) / len(called), 4) if called else None,
            "markets": len({r["market"] for r in ok}), "mean_ticks": None if mean is None else round(mean, 3),
            "ci_low_ticks": None if lo is None else round(lo, 3), "ci_high_ticks": None if hi is None else round(hi, 3)}


def ref_bucket_rows(rows, label):
    sizes = sorted(r["size"] for r in rows if r["call"] is not None)
    cuts = [stats.percentile(sizes, q) for q in (25, 50, 75)] if sizes else [0, 0, 0]
    first, second = stats.split_halves([r for r in rows if r["call"] is not None], key=lambda r: r["t"])
    half = {id(r): "first_half" for r in first} | {id(r): "second_half" for r in second}
    dims = {"all": lambda r: "all", "cheap": lambda r: "cheap" if r["cheap"] else None, "price_bucket": lambda r: r["bucket"], "day": lambda r: r["day"],
            "size_quartile": lambda r: stats.quartile_label(r["size"], cuts), "half": lambda r: half.get(id(r))}
    out = []
    for dim, key in dims.items():
        groups = defaultdict(list)
        for r in rows:
            g = key(r)
            if g is not None:
                groups[g].append(r)
        for g, members in sorted(groups.items()):
            for h in M.HORIZONS_S:
                out.append({"dimension": dim, "group": g, "horizon_s": h, **ref_cut(members, h), "label": label})
    return out


def ref_day_rows(rows, label):
    return [{"day": day, "scope": scope, "horizon_s": h, **ref_cut(sel, h), "label": label}
            for day, members in stats.by_utc_day(rows, ts=lambda r: r["t"]).items()
            for scope, sel in (("all", members), ("cheap", [r for r in members if r["cheap"]])) for h in M.HORIZONS_S]


def ref_simulate(rows, fill_rate, max_fill, cheap_only, fee):
    by_market = defaultdict(list)
    for r in rows:
        by_market[r["market"]].append(r)
    daily = defaultdict(lambda: {"pnl": 0.0, "markets": 0, "fills": 0, "shares": 0.0})
    max_inv = max_dd = 0.0
    for market, members in by_market.items():
        prints = [(r["t"], r["price"], r["size"], r["call"] if (r["cheap"] or not cheap_only) else None) for r in sorted(members, key=lambda r: r["t"])]
        sim = stats.simulate_maker_inventory(prints, fill_rate, max_fill, M.INV_CAP, fee)
        max_inv, max_dd = max(max_inv, sim["max_abs_inventory"]), max(max_dd, sim["max_drawdown"])
        for day, d in sim["daily"].items():
            if d["fills"] or d["pnl"]:
                daily[day]["pnl"] += d["pnl"]
                daily[day]["markets"] += 1
                daily[day]["fills"] += d["fills"]
                daily[day]["shares"] += d["shares"]
    return {"daily": dict(sorted(daily.items())), "max_abs_inventory": max_inv, "max_drawdown": round(max_dd, 2)}


def ref_leader(rows, now, horizon=300):
    ok = [r for r in rows if r[f"m{horizon}"] is not None]
    notional = math.fsum(r["size"] * r["price"] for r in ok)
    edge = math.fsum(r["size"] * r[f"m{horizon}"] for r in ok) / notional if notional else None
    observed = math.fsum(r["size"] * r["price"] for r in rows)
    days = len({r["day"] for r in rows}) or 1
    days_left = max((M.SETTLEMENT - now).total_seconds() / 86400, 0.0)
    out = {"edge_per_notional": None if edge is None else round(edge, 5), "observed_notional": round(observed, 2), "observed_days": days,
           "observed_daily_notional": round(observed / days, 2), "days_left": round(days_left, 2)}
    if edge is None or edge <= 0:
        return {**out, "turnover_needed": None, "share_of_observed_volume": None, "note": "no positive edge: the target is unreachable at this edge"}
    turnover = M.LEADER_TARGET / edge
    return {**out, "turnover_needed": round(turnover, 2), "share_of_observed_volume": round(turnover / observed, 3) if observed else None,
            "daily_turnover_needed": round(turnover / days_left, 2) if days_left else None,
            "share_of_observed_daily_volume": round(turnover / days_left / (observed / days), 3) if days_left and observed else None}


def ref_rule_table(rows, sim_rule):
    h = 300
    cheap_ok = [r for r in rows if r["cheap"] and r[f"m{h}"] is not None]
    first, second = stats.split_halves(cheap_ok, key=lambda r: r["t"])
    halves = {"first_half": ref_cut(first, h), "second_half": ref_cut(second, h)}
    days = {d: ref_cut(ms, h)["mean_ticks"] for d, ms in stats.by_utc_day(cheap_ok, ts=lambda r: r["t"]).items()}
    return M.rule_table(len(cheap_ok), len({r["market"] for r in cheap_ok}), halves, days, sim_rule)


def ref_analyse(rows, label, fee, now):
    sims = {(scope, f, q): ref_simulate(rows, f, q, scope == "cheap", fee) for scope in ("all", "cheap") for f in M.FILL_RATES for q in M.FILL_CAPS}
    buckets, days = ref_bucket_rows(rows, label), ref_day_rows(rows, label)
    scen = {f"f={f},Q={q}": {"all": round(math.fsum(min(q, f * r["size"]) * r["m300"] for r in rows if r["m300"] is not None), 2),
                             "cheap": round(math.fsum(min(q, f * r["size"]) * r["m300"] for r in rows if r["cheap"] and r["m300"] is not None), 2)}
            for f in M.FILL_RATES for q in M.FILL_CAPS}
    summary = {"prints": len(rows), "markets": len({r["market"] for r in rows}), "days": sorted({r["day"] for r in rows}), "cuts_reported": len(buckets) + len(days),
               "fill_scenarios_no_inventory_limit": scen,
               "inventory_simulation": {f"{scope} f={f},Q={q},cap={M.INV_CAP}": {"total_pnl": round(math.fsum(d["pnl"] for d in s["daily"].values()), 2),
                                                                                "max_abs_inventory": s["max_abs_inventory"], "max_drawdown": s["max_drawdown"]}
                                        for (scope, f, q), s in sims.items()},
               "leader_benchmark": {"all": ref_leader(rows, now), "cheap": ref_leader([r for r in rows if r["cheap"]], now)},
               "rule_table": ref_rule_table(rows, sims[("cheap", *M.RULE_SCENARIO)])}
    sim_rows = [{"scope": scope, "fill_rate": f, "max_fill": q, "inv_cap": M.INV_CAP, "day": day, "pnl": round(d["pnl"], 2), "markets": d["markets"],
                 "fills": d["fills"], "shares": d["shares"], "label": label} for (scope, f, q), s in sims.items() for day, d in s["daily"].items()]
    return {"buckets": buckets, "days": days, "sim_rows": sim_rows, "summary": summary}


def ref_quotes(paths):
    by_ex = defaultdict(list)
    for p in paths:
        for line in read_jsonl(p):
            if line.get("exchange_id") and line.get("ts"):
                by_ex[str(line["exchange_id"])].append({**line, "_t": ref_ts(line["ts"]).timestamp()})
    out = {}
    for eid, lines in by_ex.items():
        lines.sort(key=lambda l: l["_t"])
        out[eid] = ([l["_t"] for l in lines], lines)
    return out


def ref_match(tapes, quotes):
    rows = []
    agree = disagree = fallback = quote_labels = 0
    for eid, trades in sorted(tapes.items()):
        if eid not in quotes:
            continue
        q_times, q_lines = quotes[eid]
        ordered = sorted((t for t in trades if t.get("price") is not None and t.get("createdAt")), key=lambda t: (str(t["createdAt"]), str(t.get("id"))))
        prices = [float(t["price"]) for t in ordered]
        ticks = stats.tick_rule(prices)
        for i, trade in enumerate(ordered):
            when = ref_ts(trade["createdAt"])
            t = when.timestamp()
            q0 = stats.prevailing_quote(q_times, q_lines, t, MQ.MAX_QUOTE_AGE_S)
            if q0 is None:
                continue
            quote_call = stats.lee_ready(prices[i], q0.get("bid"), q0.get("ask"))
            if quote_call is not None:
                quote_labels += 1
                if ticks[i] is not None:
                    agree, disagree = agree + (ticks[i] == quote_call), disagree + (ticks[i] != quote_call)
            call = quote_call if quote_call is not None else ticks[i]
            fallback += quote_call is None and ticks[i] is not None
            sign = stats.maker_sign(call)
            row = {"market": eid, "t": when, "day": when.strftime("%Y-%m-%d"), "price": prices[i], "call": call, "cheap": M.is_cheap(prices[i]),
                   "spread": stats.spread_bucket(q0.get("bid"), q0.get("ask"))}
            for h in MQ.HORIZONS_S:
                qh = stats.prevailing_quote(q_times, q_lines, t + h, MQ.MAX_QUOTE_AGE_S) if t + h <= q_times[-1] else None
                row[f"m{h}"] = None if sign is None or qh is None or qh.get("mid") is None else sign * (prices[i] - qh["mid"])
            rows.append(row)
    both = agree + disagree
    return rows, {"quote_labels": quote_labels, "tick_fallback": fallback, "labels_compared": both, "disagree": disagree,
                  "disagree_share": round(disagree / both, 4) if both else None}


def ref_cuts(rows):
    first, second = stats.split_halves([r for r in rows if r["call"] is not None], key=lambda r: r["t"])
    half = {id(r): "first_half" for r in first} | {id(r): "second_half" for r in second}
    dims = {"all": lambda r: "all", "day": lambda r: r["day"], "half": lambda r: half.get(id(r)), "spread": lambda r: r["spread"]}
    out = []
    for scope, sel in (("all", list(rows)), ("cheap", [r for r in rows if r["cheap"]])):
        for dim, key in dims.items():
            groups = defaultdict(list)
            for r in sel:
                g = key(r)
                if g is not None:
                    groups[g].append(r)
            for g, members in sorted(groups.items()):
                for h in MQ.HORIZONS_S:
                    out.append({"scope": scope, "dimension": dim, "group": g, "horizon_s": h, **ref_cut(members, h), "label": MQ.LABEL})
    return out


def ref_verdict(rows):
    cheap = [r for r in rows if r["cheap"] and r["call"] is not None and r["m300"] is not None]
    markets, days = len({r["market"] for r in cheap}), len({r["day"] for r in cheap})
    first, second = stats.split_halves(cheap, key=lambda r: r["t"])
    halves = {"first_half": ref_cut(first, 300), "second_half": ref_cut(second, 300)}
    sample_ok = len(cheap) >= 150 and markets >= 10 and days >= 2
    holds = all(x["mean_ticks"] is not None and x["mean_ticks"] >= 0.25 and x["ci_low_ticks"] is not None and x["ci_low_ticks"] > 0 for x in halves.values())
    return {"matched_cheap_prints": len(cheap), "markets": markets, "utc_days": days, "halves": halves,
            "result": "INSUFFICIENT DATA" if not sample_ok else ("PASS" if holds else "FAIL")}


# ---------------------------------------------------------------------- synthetic data (written to files)


def _iso(t):
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def synthetic(tmp_path):
    """6 markets x 3 days of trades (cheap and mid prices, ties within and across markets, a price-less
    trade, duplicates across two files) and 30 s quote snapshots with gaps and one-sided quotes."""
    rng = random.Random(11)
    trades = defaultdict(list)
    for m in range(6):
        eid = f"{100 + m}"
        base = [0.06, 0.92, 0.45, 0.10, 0.88, 0.55][m]
        price = base
        n = 0
        for day in range(3):
            t = T0 + timedelta(days=day)
            for _ in range(120):
                t += timedelta(seconds=rng.choice([0, 10, 25, 40, 90, 200]))  # 0 s gaps: ties inside a market
                price = min(0.99, max(0.01, round(price + rng.choice([-0.005, 0, 0.005, 0.01, -0.01]), 3)))
                n += 1
                trades[eid].append({"id": f"{eid}-{n:05d}", "createdAt": _iso(t), "price": price, "size": rng.choice([5, 20, 50, 100, 400, 1500]), "side": "YES"})
        trades[eid].append({"id": f"{eid}-nop", "createdAt": _iso(T0), "price": None, "size": 1})
    for eid in ("101", "102"):  # identical timestamps across markets: ties at the half cut
        trades[eid].append({"id": f"{eid}-tie", "createdAt": _iso(T0 + timedelta(days=1, hours=4)), "price": 0.9 if eid == "101" else 0.45, "size": 30})
    tape_a, tape_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    records = [{"ts": "2026-10-23T00:00:00Z", "path": f"/exchanges/{eid}/trades", "body": {"data": ts[k:k + 150], "pagination": {"hasMore": False}}}
               for eid, ts in sorted(trades.items()) for k in range(0, len(ts), 150)]
    tape_a.write_text("\n".join(json.dumps(r) for r in records[: len(records) // 2 + 1]) + "\n", encoding="utf-8")
    tape_b.write_text("\n".join(json.dumps(r) for r in records[len(records) // 2:]) + "\n", encoding="utf-8")  # one page in both files
    snaps = tmp_path / "snapshots.jsonl"
    with snaps.open("w", encoding="utf-8") as fh:
        for eid, ts in sorted(trades.items()):
            if eid == "105":
                continue  # a traded market without quotes
            first, last = ref_ts(ts[0]["createdAt"]), max(ref_ts(t["createdAt"]) for t in ts)
            t = first - timedelta(seconds=30)
            mid = float(ts[0]["price"])
            while t <= last + timedelta(seconds=2000):
                if rng.random() > 0.05:  # ~5% skipped polls: gaps longer than 45 s
                    mid = min(0.985, max(0.015, mid + rng.choice([-0.005, 0, 0.005])))
                    bid, ask = round(mid - 0.005, 3), round(mid + rng.choice([0.0, 0.005, 0.01]), 3)
                    line = {"ts": _iso(t), "exchange_id": eid, "bid": None if rng.random() < 0.03 else bid, "ask": ask, "mid": None, "source": "bulk"}
                    line["mid"] = None if line["bid"] is None else round((line["bid"] + ask) / 2, 6)
                    fh.write(json.dumps(line) + "\n")
                    if rng.random() < 0.02:  # a second line at the same time (a book read)
                        fh.write(json.dumps({**line, "source": "book"}) + "\n")
                t += timedelta(seconds=30)
    return [tape_a, tape_b], snaps


def _old_tapes(paths):
    return TAPE.inputs_from_raw([rec for p in paths for rec in read_jsonl(p)])["trades"]


# ---------------------------------------------------------------------- regression


def test_maker_estimate_streaming_equals_the_old_code(tmp_path):
    paths, _ = synthetic(tmp_path)
    old_rows, old_label = ref_build_prints(_old_tapes(paths), fee=0.0005)
    old = ref_analyse(old_rows, old_label, 0.0005, NOW)
    new, label = M.run_analysis(M.stream_tapes(paths), 0.0005, NOW)
    assert label == old_label
    assert new["buckets"] == old["buckets"]
    assert new["days"] == old["days"]
    assert new["sim_rows"] == old["sim_rows"]
    for key, value in old["summary"].items():
        assert new["summary"][key] == value, key
    assert any(r["dimension"] == "half" for r in new["buckets"]) and new["summary"]["prints"] > 2000


def test_maker_estimate_frozen_scope_equals_the_old_code(tmp_path):
    paths, _ = synthetic(tmp_path)
    tapes = _old_tapes(paths)
    times = [str(t["createdAt"]) for ts in tapes.values() for t in ts if t.get("createdAt")]
    store = M.stream_tapes(paths)
    frozen = M.freeze_rule(tmp_path, paths, store, 0.0)
    assert frozen["data"]["markets"] == len(tapes) and frozen["data"]["trades"] == len(times)
    assert (frozen["data"]["first_trade"], frozen["data"]["last_trade"]) == (min(times), max(times))
    import hashlib
    assert frozen["data"]["sha256"] == hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
    assert store.tape_fields() == TAPE.tape_fields([t for ts in tapes.values() for t in ts])


def test_maker_markout_quotes_streaming_equals_the_old_code(tmp_path):
    paths, snaps = synthetic(tmp_path)
    old_rows, old_labels = ref_match(_old_tapes(paths), ref_quotes([snaps]))
    tapes = M.stream_tapes(paths)
    quotes = MQ.stream_snapshots([snaps], keep=set(tapes.markets))
    analysis, labels, matched = MQ.run_analysis(tapes, quotes)
    assert matched == len(old_rows) and labels == old_labels
    assert analysis.cuts() == ref_cuts(old_rows)
    assert analysis.verdict() == ref_verdict(old_rows)
    assert len(quotes.seen) == 5 and set(quotes.markets) <= set(tapes.markets)


# ---------------------------------------------------------------------- the RSS guard


def test_rss_guard_raises_above_the_limit_and_logs_stages(caplog):
    readings = iter([50.0, 120.0])
    guard = RssGuard(100.0, measure=lambda: next(readings))
    with caplog.at_level("INFO"):
        assert guard.stage("loaded") == 50.0
    assert any(r.getMessage() == "Peak RSS" and r.stage == "loaded" for r in caplog.records)
    with pytest.raises(MemoryLimitExceeded) as exc:
        guard.check("pass 1 101")
    assert exc.value.stage == "pass 1 101" and exc.value.peak_mb == 120.0 and guard.peak_mb == 120.0
    assert RssGuard(1.0, measure=lambda: None).check("x") is None  # unreadable: disabled, never raises


def _quiet(monkeypatch, tmp_path, module):
    import config

    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(module, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(module, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    (tmp_path / "out").mkdir(exist_ok=True)


def test_rss_guard_stops_both_scripts_cleanly_with_a_tiny_limit(monkeypatch, tmp_path, caplog):
    paths, snaps = synthetic(tmp_path)
    _quiet(monkeypatch, tmp_path, M)
    _quiet(monkeypatch, tmp_path, MQ)
    monkeypatch.setattr(M, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reads")))
    if peak_rss_mb() is None:
        pytest.skip("peak RSS unreadable on this platform")
    with caplog.at_level("ERROR"):
        assert M.main(["--raw", *map(str, paths), "--max-rss-mb", "1"]) == M.EXIT_MEMORY
        assert MQ.main(["--tape", *map(str, paths), "--snapshots", str(snaps), "--max-rss-mb", "1"]) == MQ.EXIT_MEMORY
    stops = [r for r in caplog.records if r.getMessage().startswith("Memory limit reached")]
    assert len(stops) == 2 and all(r.limit_mb == 1.0 for r in stops)


def test_both_scripts_run_end_to_end_under_the_default_limit(monkeypatch, tmp_path):
    paths, snaps = synthetic(tmp_path)
    _quiet(monkeypatch, tmp_path, M)
    _quiet(monkeypatch, tmp_path, MQ)
    monkeypatch.setattr(M, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reads")))
    assert M.main(["--raw", *map(str, paths)]) == 0
    assert MQ.main(["--tape", *map(str, paths), "--snapshots", str(snaps)]) == 0
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["matched_prints"] > 0
