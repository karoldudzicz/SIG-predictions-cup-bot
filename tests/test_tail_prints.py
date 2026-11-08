"""scripts/tail_prints.py on a synthetic saved tape - offline, no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tail_prints.py"
_spec = importlib.util.spec_from_file_location("tail_prints", SCRIPT)
T = importlib.util.module_from_spec(_spec)
sys.modules["tail_prints"] = T
_spec.loader.exec_module(T)

MAP = MarketMap({"1": MarketMapping("1", "m1", "D RI", Race("senate", "RI"), "D", None, None)})


def _trade(tid, when, price, size=100):
    return {"id": str(tid), "createdAt": when.isoformat().replace("+00:00", "Z"), "price": price, "size": size, "side": "YES"}


def _day(start, n=10, step_min=1, price=0.50, tail_at=None, tail_price=0.02, prefix="x"):
    """n trades a minute apart at `price`; trade number `tail_at` printed at `tail_price`."""
    return [_trade(f"{prefix}{k}", start + timedelta(minutes=step_min * k), tail_price if k == tail_at else price) for k in range(n)]


OPEN = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
TAPE = (_day(OPEN, tail_at=3, prefix="a")  # tail at 16:03 - inside the excluded 30 minutes
        + _day(OPEN + timedelta(hours=2), tail_at=5, tail_price=0.97, prefix="b")  # tail at 18:05
        + _day(OPEN + timedelta(days=1), prefix="c"))  # 10-02: no tails


def test_tapes_from_raw_dedupes_and_marks_complete_histories():
    records = [{"ts": "x", "path": "/tournaments/t/markets", "body": {"data": [{"id": "m", "title": "RI Senate", "exchanges": [{"id": "1"}]}]}},
               {"ts": "x", "path": "/exchanges/1/trades", "params": {"limit": 200}, "body": {"data": TAPE[:5], "pagination": {"hasMore": True}}},
               {"ts": "x", "path": "/exchanges/1/trades", "params": {"limit": 200, "cursor": "c"}, "body": {"data": TAPE, "pagination": {"hasMore": False}}},
               {"ts": "x", "path": "/exchanges/2/trades", "params": {"limit": 200}, "body": {"data": TAPE[:3], "pagination": {"hasMore": True}}},
               {"ts": "x", "path": "/exchanges/3/trades", "params": {"from": "2026-10-05T00:00:00Z"}, "body": {"data": [], "pagination": {"hasMore": False}}}]
    tapes, complete, titles = T.tapes_from_raw(records)
    assert len(tapes["1"]) == 30 and complete == {"1"} and titles == {"1": "RI Senate"}  # a `from` page never proves completeness


def test_covers_day():
    assert T.covers_day("2026-10-01", None, True)
    assert T.covers_day("2026-10-03", datetime(2026, 10, 2, 9, tzinfo=timezone.utc), False)
    assert not T.covers_day("2026-10-02", datetime(2026, 10, 2, 9, tzinfo=timezone.utc), False)  # starts mid-day: partial


def test_exchange_day_rows_tails_levels_and_proxy():
    rows = {r["day"]: r for r in T.exchange_day_rows("1", TAPE, True, MAP, None)}
    d1 = rows["2026-10-01"]
    assert (d1["trades"], d1["tails"], d1["tails_<0.05"], d1["tails_>0.95"]) == (20, 2, 1, 1)
    assert d1["proxy_earn"] == pytest.approx(100 * 0.48 + 100 * 0.47)
    assert d1["proxy_earn_excl_open30"] == pytest.approx(100 * 0.47)  # the 16:03 tail is excluded
    assert d1["tail_share_of_trades"] == 0.1 and d1["race"] == "senate:RI"
    assert rows["2026-10-02"]["tails"] == 0 and rows["2026-10-02"]["proxy_earn"] == 0


def test_daily_totals_only_count_covered_exchanges_and_compare():
    covered = T.exchange_day_rows("1", TAPE, True, MAP, None)
    partial = [{**r, "exchange_id": "2", "covered": False} for r in covered]
    daily = {d["day"]: d for d in T.daily_totals(covered + partial)}
    assert daily["2026-10-01"]["exchanges_covered"] == 1 and daily["2026-10-01"]["exchanges_partial"] == 1
    assert daily["2026-10-01"]["tails"] == 2 and daily["2026-10-01"]["proxy_earn_per_1000_trades"] == pytest.approx(1000 * 95.0 / 20)
    cmp = T.opening_vs_later(list(daily.values()))
    assert cmp["opening"]["day"] == "2026-10-01" and cmp["later_days"] == 1 and cmp["later_median_proxy_earn"] == 0.0
    md = T.summary_markdown(list(daily.values()), cmp, 2, "raw")
    assert md.startswith("# Tail prints (PROXY)") and "not P&L" in md and "| 2026-10-02 |" in md


def test_main_offline(monkeypatch, tmp_path):
    import config

    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"ts": "x", "path": "/exchanges/1/trades", "params": {"limit": 200}, "body": {"data": TAPE, "pagination": {"hasMore": False}}}) + "\n",
                   encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(T, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(T.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.setattr(T, "run_dir", lambda script: tmp_path / "out")
    (tmp_path / "out").mkdir()
    assert T.main(["--raw", str(raw)]) == 0
    for name in ("tail_prints_by_exchange_day.csv", "tail_prints_daily.csv", "summary.md"):
        assert (tmp_path / "out" / name).exists(), name
    monkeypatch.setattr(T, "OUT_ROOT", tmp_path / "nothing")
    assert T.main([]) == 2


def test_script_makes_no_reads():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "oracle_api", "aiohttp"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            assert "research_reader" not in [a.name for a in node.names]
