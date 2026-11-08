"""scripts/maker_estimate.py on synthetic tapes and hand-built rows - offline, no network.

Run from the project root:  python -m pytest tests/test_maker_estimate.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "maker_estimate.py"
_spec = importlib.util.spec_from_file_location("maker_estimate", SCRIPT)
M = importlib.util.module_from_spec(_spec)
sys.modules["maker_estimate"] = M
_spec.loader.exec_module(M)

T0 = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)


def _row(market, minutes, m300, price=0.10, size=100.0, call="buy"):
    t = T0 + timedelta(minutes=minutes)
    return {"market": market, "t": t, "day": t.strftime("%Y-%m-%d"), "price": price, "size": size, "call": call,
            "bucket": M.stats.price_bucket(price), "cheap": M.is_cheap(price), "m60": m300, "m300": m300, "m1800": m300,
            "s60": "ok", "s300": "ok", "s1800": "ok"}


def _rows(markets=20, per_market_day=6, m300=0.01, days=2):
    rows = []
    for d in range(days):
        for k in range(markets):
            for j in range(per_market_day):
                noise = 0.001 * ((k + j) % 3 - 1)
                rows.append(_row(f"x{k}", d * 1440 + k * 10 + j, m300 + noise))
    return rows


def _sim(daily_pnls):
    return {"daily": {f"2026-10-{10 + i:02d}": {"pnl": p, "markets": 1, "fills": 1, "shares": 1} for i, p in enumerate(daily_pnls)}}


# ---------------------------------------------------------------------- the frozen rule


def test_rule_pass_when_every_item_holds():
    table = M.evaluate_rule(_rows(), _sim([5.0, 3.0]))
    holds = {r["hypothesis"]: r["holds"] for r in table}
    assert holds["(a) sample"] and holds["(b) markout 300 s"] and holds["(c) days"] and holds["(d) simulation"]
    assert table[-1]["result"] == "PASS"
    assert "ticks" in table[1]["first_half"] and "ticks" in table[1]["second_half"]  # per-half results reported
    assert table[3]["result"].startswith("positive on 2 of 2 days")  # K of N, not a share


def test_rule_fail_and_insufficient_data():
    assert M.evaluate_rule(_rows(m300=-0.01), _sim([5.0, 3.0]))[-1]["result"] == "FAIL"  # negative markout
    assert M.evaluate_rule(_rows(), _sim([-5.0, -3.0]))[-1]["result"] == "FAIL"  # simulation loses
    assert M.evaluate_rule(_rows(markets=10), _sim([5.0, 3.0]))[-1]["result"] == "INSUFFICIENT DATA"  # < 15 markets
    assert M.evaluate_rule(_rows(days=1, per_market_day=12), _sim([5.0]))[-1]["result"] == "INSUFFICIENT DATA"  # one UTC day
    small = M.evaluate_rule(_rows(m300=0.002), _sim([5.0, 3.0]))  # 0.4 tick: below +0.5
    assert small[1]["holds"] is False and small[-1]["result"] == "FAIL"


# ---------------------------------------------------------------------- prints and markouts


def _trade(tid, seconds, price, size=100, **extra):
    return {"id": str(tid), "createdAt": (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z"), "price": price, "size": size, **extra}


def test_aggressor_calls_use_a_real_field_else_the_tick_rule():
    with_field = [_trade(1, 0, 0.5, takerSide="buy"), _trade(2, 1, 0.5, takerSide="sell")]
    assert M.aggressor_calls(with_field) == (["buy", "sell"], "aggressor field 'takerSide'")
    calls, label = M.aggressor_calls([_trade(1, 0, 0.50), _trade(2, 1, 0.51), _trade(3, 2, 0.50)])
    assert calls == [None, "buy", "sell"] and label == M.PROXY


def test_build_prints_markouts_net_of_fee_and_statuses():
    trades = [_trade(1, 0, 0.10), _trade(2, 10, 0.12), _trade(3, 20, 0.10), _trade(4, 30, 0.11), _trade(5, 4000, 0.10)]
    rows, label = M.build_prints({"x": trades}, fee=0.001)
    assert label == M.PROXY and [r["call"] for r in rows] == [None, "buy", "sell", "buy", "sell"]
    second = rows[1]  # maker SOLD 0.12; in (10, 310]: last buy 0.11, last sell 0.10 -> ref 0.105 -> +0.015, minus fee 0.001
    assert second["s300"] == "ok" and second["m300"] == pytest.approx(0.014) and second["cheap"] is True
    assert rows[0]["s300"] == "no aggressor" and rows[4]["s300"] == "beyond data"


def test_simulation_cheap_only_still_marks_with_every_print():
    rows = [_row("x", 0, 0.0, price=0.10, size=1000, call="sell"),  # cheap: maker buys 100 @ 0.10
            _row("x", 60, 0.0, price=0.50, size=1000, call="sell")]  # not cheap: no fill, but marks the day at 0.50
    sim = M.simulate(rows, 0.25, 100, cheap_only=True, fee=0.0)
    day = sim["daily"]["2026-10-10"]
    assert day["fills"] == 1 and day["pnl"] == pytest.approx(100 * (0.50 - 0.10)) and sim["max_abs_inventory"] == 100


def test_leader_benchmark_turnover_and_capacity():
    rows = [_row("x", 0, 0.005, price=0.10, size=100), _row("x", 1, 0.005, price=0.10, size=100)]
    bench = M.leader_benchmark(rows, now=datetime(2026, 10, 25, 17, tzinfo=timezone.utc))
    assert bench["edge_per_notional"] == pytest.approx(0.05) and bench["turnover_needed"] == pytest.approx(83_000 / 0.05)
    assert bench["observed_notional"] == 20.0 and bench["share_of_observed_volume"] == pytest.approx(83_000 / 0.05 / 20.0)
    assert bench["days_left"] == 10.0
    assert M.leader_benchmark([_row("x", 0, -0.01)], now=T0)["turnover_needed"] is None


# ---------------------------------------------------------------------- the run


def _tape_records():
    trades = []
    n = 0
    for day in range(2):
        for k in range(60):
            n += 1
            trades.append(_trade(n, day * 86400 + k * 30, [0.10, 0.11, 0.10, 0.09][k % 4]))
    return [{"ts": "2026-10-12T00:00:00Z", "path": "/exchanges/7/trades", "body": {"data": trades, "pagination": {"hasMore": False}}}]


def test_main_freezes_the_rule_before_any_markout_and_writes_outputs(monkeypatch, tmp_path):
    import config

    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in _tape_records()) + "\n", encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(M, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(M, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.setattr(M, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reads by default")))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    (tmp_path / "out").mkdir()
    real_build = M.build_prints

    def build_after_freeze(*a, **k):
        assert (tmp_path / "out" / "frozen_rule.json").exists(), "the rule must be frozen before markouts are computed"
        return real_build(*a, **k)

    monkeypatch.setattr(M, "build_prints", build_after_freeze)
    assert M.main(["--raw", str(raw), "--fetch-missing", "3"]) == 0  # no --yes: the top-up is skipped, no read
    out = tmp_path / "out"
    for name in ("summary.json", "by_bucket.csv", "by_day.csv", "sim_daily.csv", "frozen_rule.json"):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggressor"] == M.PROXY and summary["cuts_reported"] > 0 and "NOT captured" in summary["caveat"]
    assert summary["rule_table"][-1]["result"] == "INSUFFICIENT DATA"  # one market: below the sample thresholds
    frozen = json.loads((out / "frozen_rule.json").read_text(encoding="utf-8"))
    assert frozen["data"]["trades"] == 120 and len(frozen["data"]["sha256"]) == 64
    assert "PROXY: tick rule" in (out / "by_bucket.csv").read_text(encoding="utf-8")


def test_script_imports_no_strategy_quoter_or_ladder():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
