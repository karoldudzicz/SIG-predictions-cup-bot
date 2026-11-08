"""scripts/maker_markout_quotes.py on synthetic tapes and snapshots - offline, hand-computed cases.

Run from the project root:  python -m pytest tests/test_maker_markout_quotes.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "maker_markout_quotes.py"
_spec = importlib.util.spec_from_file_location("maker_markout_quotes", SCRIPT)
MQ = importlib.util.module_from_spec(_spec)
sys.modules["maker_markout_quotes"] = MQ
_spec.loader.exec_module(MQ)

T0 = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)


def _iso(seconds):
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _snap(seconds, bid, ask):
    return {"ts": _iso(seconds), "exchange_id": "1", "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "source": "bulk"}


# Quote 0.10 / 0.12 (mid 0.11) until 90 s, then 0.10 / 0.11 (mid 0.105), every 30 s up to 600 s.
SNAPS = [_snap(s, 0.10, 0.12 if s < 100 else 0.11) for s in range(0, 601, 30)]
TRADES = [
    {"id": "a", "createdAt": _iso(10), "price": 0.12, "size": 100},  # at the ask -> buy (maker sold 0.12)
    {"id": "b", "createdAt": _iso(20), "price": 0.11, "size": 100},  # inside -> tick rule: down -> sell (flagged)
    {"id": "c", "createdAt": _iso(70), "price": 0.09, "size": 100},  # below the bid -> sell; tick: down -> sell (agree)
    {"id": "d", "createdAt": _iso(80), "price": 0.10, "size": 100},  # at the bid -> sell; tick: up -> buy (disagree)
    {"id": "e", "createdAt": _iso(700), "price": 0.10, "size": 100},  # last quote 100 s old -> unmatched
]


def _quotes():
    return MQ.snapshots_from_lines(SNAPS)


def test_match_prints_lee_ready_fallback_and_markout_vs_mid():
    rows, labels = MQ.match_prints({"1": TRADES}, _quotes())
    by_id = {r["t"].second + 60 * r["t"].minute: r for r in rows}
    assert len(rows) == 4  # the 700 s print has no fresh quote
    a = by_id[10]  # maker sold 0.12; mid at 310 s = 0.105 -> +0.015 = 3 ticks
    assert a["call"] == "buy" and not a["tick_fallback"] and a["m300"] == pytest.approx(0.015) and a["spread"] == ">=3"  # 0.02 = 4 ticks
    b = by_id[20]  # maker bought 0.11 (tick fallback); mid at 320 s 0.105 -> -0.005
    assert b["call"] == "sell" and b["tick_fallback"] and b["m300"] == pytest.approx(-0.005)
    assert by_id[10]["m1800"] is None  # 1810 s is past the snapshots
    # "a" is the first print: no tick-rule label, so only "c" and "d" are compared (and "d" disagrees)
    assert labels == {"quote_labels": 3, "tick_fallback": 1, "labels_compared": 2, "disagree": 1, "disagree_share": 0.5}


def _row(market, minutes, m300, price=0.10):
    t = T0 + timedelta(minutes=minutes)
    return {"market": market, "t": t, "day": t.strftime("%Y-%m-%d"), "price": price, "size": 100.0, "call": "buy", "tick_fallback": False,
            "cheap": MQ.maker.is_cheap(price), "spread": "1", "m60": m300, "m300": m300, "m1800": None}


def _rows(markets=10, per=8, m300=0.0025, days=2):
    return [_row(f"x{k}", d * 1440 + k * 10 + j, m300 + 0.0005 * ((k + j) % 3 - 1)) for d in range(days) for k in range(markets) for j in range(per)]


def test_verdict_pass_fail_insufficient():
    assert MQ.verdict(_rows())["result"] == "PASS"  # 160 prints, 10 markets, 2 days, ~0.5 tick
    assert MQ.verdict(_rows(m300=-0.0025))["result"] == "FAIL"
    assert MQ.verdict(_rows(m300=0.0005))["result"] == "FAIL"  # 0.1 tick: below +0.25
    assert MQ.verdict(_rows(markets=9, per=9))["result"] == "INSUFFICIENT DATA"  # 9 markets
    assert MQ.verdict(_rows(days=1, per=16))["result"] == "INSUFFICIENT DATA"  # one UTC day
    assert MQ.verdict(_rows(per=7))["result"] == "INSUFFICIENT DATA"  # 140 prints


def test_cuts_cover_scopes_dimensions_and_spread():
    table = MQ.cuts(_rows())
    dims = {(r["scope"], r["dimension"]) for r in table}
    assert {("all", "all"), ("cheap", "half"), ("cheap", "day"), ("all", "spread")} <= dims
    assert all(r["label"] == MQ.LABEL for r in table)


def test_main_freezes_before_markouts_and_writes_outputs(monkeypatch, tmp_path):
    import config

    tape = tmp_path / "raw.jsonl"
    tape.write_text(json.dumps({"ts": "2026-10-20T13:00:00Z", "path": "/exchanges/1/trades", "body": {"data": TRADES, "pagination": {"hasMore": False}}}) + "\n",
                    encoding="utf-8")
    snaps = tmp_path / "snapshots.jsonl"
    snaps.write_text("\n".join(json.dumps(s) for s in SNAPS) + "\n", encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(MQ, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(MQ, "run_dir", lambda script: tmp_path / "out")
    (tmp_path / "out").mkdir()
    real = MQ.match_prints

    def after_freeze(*a, **k):
        assert (tmp_path / "out" / "frozen_rule.json").exists(), "the rule must be frozen before markouts are computed"
        return real(*a, **k)

    monkeypatch.setattr(MQ, "match_prints", after_freeze)
    assert MQ.main(["--tape", str(tape), "--snapshots", str(snaps)]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"]["result"] == "INSUFFICIENT DATA" and summary["matched_prints"] == 4 and "fees" in summary["not_modelled"]
    assert (tmp_path / "out" / "by_cut.csv").exists()
    assert MQ.main(["--tape", str(tape), "--snapshots"]) == 2  # no snapshots


def test_script_makes_no_reads():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "oracle_api", "aiohttp"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            assert "research_reader" not in [a.name for a in node.names]
