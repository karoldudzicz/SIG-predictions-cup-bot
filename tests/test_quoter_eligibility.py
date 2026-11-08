"""Tests for scripts/quoter_eligibility.py (read-only feasibility count, no network: the oracle pass is stubbed).

Run from the project root:  python -m pytest tests/test_quoter_eligibility.py -v
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote
from research import stats

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "quoter_eligibility.py"
_spec = importlib.util.spec_from_file_location("quoter_eligibility", SCRIPT)
QE = importlib.util.module_from_spec(_spec)
sys.modules["quoter_eligibility"] = QE
_spec.loader.exec_module(QE)
MQ = QE.mq
T0 = datetime(2026, 10, 20, 9, tzinfo=timezone.utc)


def _iso(t):
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------- pure helpers


def test_block_reason_follows_the_quoter_order():
    assert QE.block_reason(None, False, 0.5, "ok", 0.01) == "not mapped"
    assert QE.block_reason("I", False, 0.5, "ok", 0.01) == "not D / R"
    assert QE.block_reason("D", True, None, "stale", None) == "ladder market"
    assert QE.block_reason("R", False, None, "stale", 0.005) == "no oracle fair (stale)"
    assert QE.block_reason("D", False, 0.02, "ok", 0.05) == "u > 0.04"  # u before the tail, as the quoter
    assert QE.block_reason("D", False, 0.02, "ok", 0.01) == "fair outside [0.05, 0.95]"
    assert QE.block_reason("D", False, 0.05, "ok", 0.04) is None and QE.block_reason("R", False, 0.95, "ok", 0.005) is None


def test_held_positions_and_ladder_replay():
    held = QE.held_from_positions([{"exchangeId": "1", "quantity": 120}, {"exchangeId": "2", "quantity": -40}, {"exchangeId": "3", "quantity": 5, "settled": True},
                                   {"exchangeId": "4", "quantity": 0}])
    assert held == {"1": ("yes", 120), "2": ("no", 40)}
    journal = [{"event": "fill", "exchange_id": "7", "quantity": 300}, {"event": "exit_fill", "exchange_id": "7", "quantity": 100},
               {"event": "fill", "exchange_id": "8", "quantity": 50}, {"event": "exit_fill", "exchange_id": "8", "quantity": 50},
               {"event": "fill", "exchange_id": "9", "quantity": 10}, {"event": "round_closed", "exchange_id": "9"}, {"event": "placement", "exchange_id": "5"}]
    assert QE.ladder_markets(journal) == {"7"}


def test_feasibility_thresholds():
    assert QE.feasibility(8, 300).startswith("enough")
    assert QE.feasibility(7, 10_000) == "INSUFFICIENT for a pilot"
    assert QE.feasibility(40, 299) == "INSUFFICIENT for a pilot"


def test_last_quote_is_the_newest_snapshot():
    q = MQ.snapshots_from_lines([{"ts": _iso(T0 + timedelta(seconds=60)), "exchange_id": "1", "bid": 0.10, "ask": 0.115, "mid": 0.1075},
                                 {"ts": _iso(T0), "exchange_id": "1", "bid": 0.10, "ask": 0.12, "mid": 0.11}]).markets["1"]
    out = QE.last_quote(q, T0.timestamp() + 660)
    assert out == {"last_bid": 0.10, "last_ask": 0.115, "last_spread_ticks": 3, "last_quote_age_min": 10.0}


# ---------------------------------------------------------------------- synthetic inputs


def _synthetic(n_markets=4, days=2, seed=5):
    rng = random.Random(seed)
    trades, snaps = defaultdict(list), []
    for m in range(n_markets):
        eid, mid = str(10 + m), [0.08, 0.50, 0.90, 0.30][m % 4]
        for d in range(days):
            t = T0 + timedelta(days=d)
            for k in range(0, 3600, 30):
                spread = rng.choice([0.005, 0.01, 0.015, 0.02])
                mid = min(0.97, max(0.03, mid + rng.choice([-0.005, 0, 0.005])))
                bid = round(mid - 0.005, 3)
                snaps.append({"ts": _iso(t + timedelta(seconds=k)), "exchange_id": eid, "bid": bid, "ask": round(bid + spread, 3), "mid": round(bid + spread / 2, 4)})
                if rng.random() < 0.4:
                    price = rng.choice([bid, round(bid + spread, 3), round(bid + spread / 2, 4)])
                    trades[eid].append({"id": f"{eid}-{d}-{k}", "createdAt": _iso(t + timedelta(seconds=k + 5)), "price": price, "size": 10})
    return dict(trades), snaps


def _reference_cut(rows):
    called = [r for r in rows if r["call"] is not None]
    ok = [r for r in called if r["m300"] is not None]
    mean, lo, hi = stats.cluster_bootstrap_mean([r["m300"] / QE.TICK for r in ok], [(r["market"], r["day"]) for r in ok])
    return {"prints": len(called), "qualifying": len(ok), "mean_ticks": None if mean is None else round(mean, 3),
            "ci_low_ticks": None if lo is None else round(lo, 3), "ci_high_ticks": None if hi is None else round(hi, 3)}


def test_markouts_equal_a_row_based_reference():
    trades, snaps = _synthetic()
    tapes, quotes = MQ.maker.store_from_trades(trades), MQ.snapshots_from_lines(snaps)
    eligible, no_carry = ["10", "11", "12"], {"10", "12"}
    result = QE.run_markouts(tapes, quotes, eligible, no_carry)
    rows = [r for eid in sorted(eligible) for r in MQ.match_market(eid, tapes.markets[eid], quotes.markets[eid])]
    cuts = {(c["scope"], c["price"], c["spread"], c["half"]): c for c in result.cut_rows(QE.SCOPES)}
    assert len(cuts) == 2 * 2 * 4 * 3
    for scope, members in (("eligible", rows), ("no_carry", [r for r in rows if r["market"] in no_carry])):
        for price, sel in (("all", members), ("cheap", [r for r in members if r["cheap"]])):
            first, second = stats.split_halves([r for r in sel if r["call"] is not None], key=lambda r: r["ts"])
            half = {id(r): "first_half" for r in first} | {id(r): "second_half" for r in second}
            for spread in QE.SPREADS:
                for h in QE.HALVES:
                    part = [r for r in sel if (spread == "all" or r["spread"] == spread) and (h == "all" or half.get(id(r)) == h)]
                    got = cuts[(scope, price, spread, h)]
                    assert {k: got[k] for k in ("prints", "qualifying", "mean_ticks", "ci_low_ticks", "ci_high_ticks")} == _reference_cut(part), (scope, price, spread, h)
    assert result.totals["eligible"]["matched"] == len(rows)
    assert result.totals["eligible"]["spread_ge2"] == sum(r["spread"] in ("2", ">=3") for r in rows)
    days = result.day_rows(QE.SCOPES)
    assert sum(d["matched_prints"] for d in days if d["scope"] == "eligible") == len(rows) and {d["day"] for d in days} == {"2026-10-20", "2026-10-21"}
    assert result.per_market["11"]["matched_prints"] == sum(r["market"] == "11" for r in rows)


# ---------------------------------------------------------------------- end to end (oracle stubbed)


def _mapping(eid, party, token=None, ticker=None):
    return MarketMapping(eid, f"m{eid}", f"Will the X Party win race {eid}?", Race("house", "NH", eid[-2:].zfill(2)), party, token, ticker)


def _oq(source, ref, bid, ask, age=1.0):
    return OracleQuote(source, ref, bid, ask, 100.0, 100.0, None, time.time() - age)


@pytest.fixture
def env(monkeypatch, tmp_path):
    import config

    trades, snaps = _synthetic(n_markets=4)
    tape, snap_path = tmp_path / "raw.jsonl", tmp_path / "snapshots.jsonl"
    tape.write_text("\n".join(json.dumps({"ts": "x", "path": f"/exchanges/{eid}/trades", "body": {"data": ts}}) for eid, ts in trades.items()) + "\n", encoding="utf-8")
    snap_path.write_text("\n".join(json.dumps(s) for s in snaps) + "\n", encoding="utf-8")
    positions = tmp_path / "review.jsonl"
    positions.write_text(json.dumps({"ts": "2026-10-22T10:00:00Z", "path": "/portfolio/positions", "body": {"positions": [{"exchangeId": "11", "quantity": -200}]}}) + "\n",
                         encoding="utf-8")
    ladder = tmp_path / "ladder_journal.jsonl"
    ladder.write_text(json.dumps({"event": "fill", "exchange_id": "13", "quantity": 100}) + "\n", encoding="utf-8")
    mapping = MarketMap({"10": _mapping("10", "D", token="t10"), "11": _mapping("11", "R", ticker="K11"), "12": _mapping("12", "I", token="t12"),
                         "13": _mapping("13", "D", token="t13")})
    fetched = []

    async def fake_fetch(tokens, tickers):
        fetched.append((list(tokens), list(tickers)))
        return {"t10": _oq("polymarket", "t10", 0.07, 0.09)}, {"K11": _oq("kalshi", "K11", 0.48, 0.50)}

    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(QE, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(QE, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.setattr(QE, "load_market_map", lambda: mapping)
    monkeypatch.setattr(QE, "fetch_oracles", fake_fetch)
    monkeypatch.setattr(QE, "LADDER_JOURNAL", ladder)
    (tmp_path / "out").mkdir()
    return tmp_path, ["--tape", str(tape), "--snapshots", str(snap_path), "--positions", str(positions)], fetched


def test_main_needs_yes_before_the_oracle_pass(env):
    tmp_path, args, fetched = env
    assert QE.main(args) == 0
    assert fetched == [] and not (tmp_path / "out" / "summary.json").exists()


def test_main_end_to_end(env):
    tmp_path, args, fetched = env
    assert QE.main(["--yes", *args]) == 0
    assert fetched == [(["t10"], ["K11"])]  # only mapped D / R markets outside the ladder
    import csv

    with (tmp_path / "out" / "eligibility.csv").open(encoding="utf-8") as fh:
        table = {r["exchange_id"]: r for r in csv.DictReader(fh)}
    assert table["10"]["eligible"] == "True" and table["10"]["fair"] == "0.08" and table["10"]["carry"] == ""
    assert table["11"]["eligible"] == "True" and table["11"]["carry"] == "no 200"
    assert table["12"]["reason"] == "not D / R" and table["13"]["reason"] == "ladder market" and table["12"]["matched_prints"] == ""
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["eligible"] == 2 and summary["counts"]["eligible_no_carry"] == 1 and summary["counts"]["ladder"] == 1
    assert summary["feasibility"]["eligible"]["result"] == "INSUFFICIENT for a pilot" and summary["feasibility"]["no_carry"]["eligible_markets"] == 1
    assert summary["feasibility"]["eligible"]["matched_prints"] == int(table["10"]["matched_prints"]) + int(table["11"]["matched_prints"])
    assert (tmp_path / "out" / "by_cut.csv").exists() and (tmp_path / "out" / "by_day.csv").exists() and (tmp_path / "out" / "oracle_quotes.json").exists()
    assert not (tmp_path / "out" / "frozen_rule.json").exists()  # a feasibility count, not a pre-registered test


def test_main_memory_guard_and_missing_inputs(env, monkeypatch):
    from research.memory import peak_rss_mb

    tmp_path, args, _ = env
    assert QE.main(["--yes", "--tape", str(tmp_path / "raw.jsonl"), "--snapshots"]) == 2
    if peak_rss_mb() is None:
        pytest.skip("peak RSS unreadable on this platform")
    assert QE.main(["--yes", *args, "--max-rss-mb", "1"]) == QE.EXIT_MEMORY


def test_script_is_read_only():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "edge_scan"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            assert "research_reader" not in [a.name for a in node.names]
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in forbidden for a in node.names)
