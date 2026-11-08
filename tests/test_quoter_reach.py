"""Tests for scripts/quoter_reach.py (offline: no SIG or oracle reads).

Run from the project root:  python -m pytest tests/test_quoter_reach.py -v
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race
from research import stats

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "quoter_reach.py"
_spec = importlib.util.spec_from_file_location("quoter_reach", SCRIPT)
QR = importlib.util.module_from_spec(_spec)
sys.modules["quoter_reach"] = QR
_spec.loader.exec_module(QR)
MQ = QR.mq
T0 = datetime(2026, 10, 20, 9, tzinfo=timezone.utc)
FAIRS = {"10": {"fair": 0.10, "u": 0.005}, "11": {"fair": 0.50, "u": 0.01}, "12": {"fair": 0.88, "u": 0.005}}


def _iso(t):
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------- pure helpers


def _row(call, price, bid, ask, m300=0.005, ts=0.0, spread=">=3"):
    return {"call": call, "price": price, "bid": bid, "ask": ask, "m300": m300, "ts": ts, "spread": spread, "day": "2026-10-20", "size": 40}


def test_reachable_items_join_is_exact_improve_is_at_or_through():
    cache = QR.QuoteCache(0.10, 0.005)  # edge 0.015 -> JOIN bid when bid <= 0.085, ask when ask >= 0.115
    rows = [_row("sell", 0.07, 0.07, 0.085), _row("sell", 0.065, 0.07, 0.085), _row("buy", 0.085, 0.07, 0.085), _row(None, 0.07, 0.07, 0.085),
            _row("buy", 0.12, 0.07, 0.12), _row("sell", 0.07, 0.07, 0.075)]
    items = dict(zip(QR.CONFIGS, QR.reachable_items(rows, cache)))
    assert items[(0.0, "join", 2)] == items[(0.0, "join", 3)] == [rows[0], rows[4]]  # at the bid; the ask 0.12 >= 0.115
    assert items[(0.0, "join", 1)] == [rows[0], rows[4], rows[5]]  # the 1-tick touch only at S_min 1
    assert items[(0.0, "improve", 2)] == [rows[0], rows[1], rows[4]]  # our bid 0.075: 0.07 and 0.065 are through it
    assert items[(0.02, "join", 2)] == [rows[0]]  # fair 0.12: the ask needs >= 0.135
    assert QR.markout(rows[0], "improve") == pytest.approx(0.0) and QR.markout(rows[0], "join") == 0.005
    assert QR.markout({"m300": None}, "improve") is None


def test_verdict_thresholds():
    assert QR.verdict({str(i): 100 for i in range(6)})["result"].startswith("enough")
    v = QR.verdict({**{str(i): 100 for i in range(5)}, "x": 99})
    assert v["result"] == "INSUFFICIENT" and v["markets_with_enough"] == 5


def _mapping(eid, token=None, ticker=None):
    return MarketMapping(eid, f"m{eid}", f"race {eid}", Race("house", "NH", eid[-2:]), "D", token, ticker)


def test_eligible_fairs_recompute_and_fallback(caplog):
    at = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    oracle = {"fetched_at": at.isoformat(), "max_age_s": 10.0,
              "polymarket": {"t10": {"mid": 0.10, "spread": 0.02, "fetched_at": at.timestamp() - 1}},
              "kalshi": {"K11": {"mid": 0.50, "spread": 0.01, "fetched_at": at.timestamp() - 30}}}  # stale at the fetch
    rows = [{"exchange_id": "10", "eligible": "True", "fair": "0.1", "u": "0.01"}, {"exchange_id": "11", "eligible": "True", "fair": "0.5", "u": "0.005"},
            {"exchange_id": "12", "eligible": "False", "fair": "0.3", "u": "0.01"}]
    market_map = MarketMap({"10": _mapping("10", token="t10"), "11": _mapping("11", ticker="K11"), "12": _mapping("12")})
    with caplog.at_level("WARNING"):
        fairs = QR.eligible_fairs(rows, oracle, market_map)
    assert fairs == {"10": {"fair": 0.10, "u": 0.01}, "11": {"fair": 0.5, "u": 0.005}}
    assert any(r.getMessage().startswith("Fair not recomputable") for r in caplog.records)


# ---------------------------------------------------------------------- synthetic tape + snapshots


def _synthetic(days=2, seed=3):
    rng = random.Random(seed)
    trades, snaps = defaultdict(list), []
    for eid, f in FAIRS.items():
        mid = f["fair"]
        for d in range(days):
            t = T0 + timedelta(days=d)
            for k in range(0, 5400, 30):
                spread = rng.choice([0.005, 0.01, 0.015, 0.025])
                mid = min(0.96, max(0.04, mid + rng.choice([-0.005, 0, 0.005])))
                bid = round(mid - 0.005, 3)
                ask = round(bid + spread, 3)
                snaps.append({"ts": _iso(t + timedelta(seconds=k)), "exchange_id": eid, "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4)})
                if rng.random() < 0.5:
                    price = rng.choice([bid, ask, round(bid - 0.005, 3), round(ask + 0.005, 3), round((bid + ask) / 2, 4)])
                    trades[eid].append({"id": f"{eid}-{d}-{k}", "createdAt": _iso(t + timedelta(seconds=k + 5)), "price": price, "size": rng.choice([10, 100, 1000])})
    return dict(trades), snaps


def _ref_cut(rows, variant):
    ms = [QR.markout(r, variant) for r in rows]
    ok = [(r, m) for r, m in zip(rows, ms) if m is not None]
    mean, lo, hi = stats.cluster_bootstrap_mean([m / QR.TICK for _, m in ok], [(r["market"], r["day"]) for r, _ in ok])
    rnd = lambda x: None if x is None else round(x, 3)  # noqa: E731
    return {"prints": len(rows), "qualifying": len(ok), "mean_ticks": rnd(mean), "ci_low_ticks": rnd(lo), "ci_high_ticks": rnd(hi)}


def test_analysis_equals_a_row_based_reference():
    trades, snaps = _synthetic()
    tapes, quotes = MQ.maker.store_from_trades(trades), MQ.snapshots_from_lines(snaps)
    analysis = QR.run_analysis(tapes, quotes, FAIRS)
    cuts = {(c["fair_shift"], c["variant"], c["s_min"], c["spread"], c["half"]): c for c in analysis.cut_rows()}
    assert len(cuts) == len(QR.CONFIGS) * 4 * 3
    all_rows = {eid: MQ.match_market(eid, tapes.markets[eid], quotes.markets[eid]) for eid in sorted(FAIRS)}
    matched = sum(len(r) for r in all_rows.values())
    for k, config in enumerate(QR.CONFIGS):
        reach = [r for eid in sorted(FAIRS) for r in QR.reachable_items(all_rows[eid], QR.QuoteCache(**FAIRS[eid]))[k]]
        first, second = stats.split_halves(reach, key=lambda r: r["ts"])
        half = {id(r): "first_half" for r in first} | {id(r): "second_half" for r in second}
        for spread in QR.SPREADS:
            for h in QR.HALVES:
                part = [r for r in reach if (spread == "all" or r["spread"] == spread) and (h == "all" or half[id(r)] == h)]
                got = cuts[(*config, spread, h)]
                assert {k2: got[k2] for k2 in ("prints", "qualifying", "mean_ticks", "ci_low_ticks", "ci_high_ticks")} == _ref_cut(part, config[1]), (config, spread, h)
        summary = analysis.config_summary(k)
        assert summary["reachable"] == len(reach) and summary["matched_prints"] == matched
        upper = sum(stats.capped_fill(r["size"]) * QR.markout(r, config[1]) for r in reach if r["m300"] is not None)
        assert summary["upper_bound_susq_total"] == pytest.approx(round(upper, 2), abs=0.011)
    join2 = analysis.config_summary(QR.CONFIGS.index((0.0, "join", 2)))
    assert 0 < join2["reachable"] < matched and join2["snapshots_quotable_share"]["either"] > 0
    ranking = analysis.ranking()
    assert len(ranking["best"]) == 3 and ranking["best"][0]["mean_ticks"] >= ranking["worst"][0]["mean_ticks"]
    assert set(ranking["best"][0]) == {"exchange_id", "mean_ticks", "reachable_s1", "reachable_s2", "reachable_s3", "low_reach"}


# ---------------------------------------------------------------------- end to end


@pytest.fixture
def env(monkeypatch, tmp_path):
    import config

    trades, snaps = _synthetic()
    tape, snap_path = tmp_path / "raw.jsonl", tmp_path / "snapshots.jsonl"
    tape.write_text("\n".join(json.dumps({"ts": "x", "path": f"/exchanges/{eid}/trades", "body": {"data": ts}}) for eid, ts in trades.items()) + "\n", encoding="utf-8")
    snap_path.write_text("\n".join(json.dumps(s) for s in snaps) + "\n", encoding="utf-8")
    run = tmp_path / "elig"
    run.mkdir()
    at = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    with (run / "eligibility.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, ["exchange_id", "eligible", "fair", "u"])
        w.writeheader()
        for eid, f in FAIRS.items():
            w.writerow({"exchange_id": eid, "eligible": "True", "fair": f["fair"], "u": f["u"]})
    pm = {f"t{eid}": {"mid": f["fair"], "spread": 2 * f["u"], "fetched_at": at.timestamp() - 1} for eid, f in FAIRS.items()}
    (run / "oracle_quotes.json").write_text(json.dumps({"fetched_at": at.isoformat(), "max_age_s": 10.0, "polymarket": pm, "kalshi": {}}), encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(QR, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(QR, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.setattr(QR, "load_market_map", lambda: MarketMap({eid: _mapping(eid, token=f"t{eid}") for eid in FAIRS}))
    (tmp_path / "out").mkdir()
    return tmp_path, ["--eligibility", str(run), "--tape", str(tape), "--snapshots", str(snap_path)]


def test_main_end_to_end(env):
    tmp_path, args = env
    assert QR.main(args) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["eligible_markets"] == 3 and len(summary["configs"]) == 18 and summary["verdict"]["join"]["result"] == "INSUFFICIENT"
    assert {c["fair_shift"] for c in summary["configs"]} == {0.0, -0.02, 0.02}
    with (tmp_path / "out" / "by_market.csv").open(encoding="utf-8") as fh:
        markets = list(csv.DictReader(fh))
    assert len(markets) == 3 * 18 and all(m["low_reach"] in ("True", "False") for m in markets)
    with (tmp_path / "out" / "by_day.csv").open(encoding="utf-8") as fh:
        assert {r["day"] for r in csv.DictReader(fh)} == {"2026-10-20", "2026-10-21"}
    assert not (tmp_path / "out" / "frozen_rule.json").exists()


def test_main_missing_inputs_and_memory_guard(env):
    from research.memory import peak_rss_mb

    tmp_path, args = env
    assert QR.main(["--eligibility", str(tmp_path / "nowhere"), *args[2:]]) == 2
    assert QR.main([*args[:4], "--snapshots"]) == 2
    if peak_rss_mb() is None:
        pytest.skip("peak RSS unreadable on this platform")
    assert QR.main([*args, "--max-rss-mb", "1"]) == QR.EXIT_MEMORY


def test_script_makes_no_reads():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "oracle_api", "edge_scan", "aiohttp"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            assert "research_reader" not in [a.name for a in node.names]
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in forbidden for a in node.names)
