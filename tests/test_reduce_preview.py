"""Tests for scripts/reduce_preview.py (read-only; SIG reads faked).

Run from the project root:  python -m pytest tests/test_reduce_preview.py -v
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "reduce_preview.py"
_spec = importlib.util.spec_from_file_location("reduce_preview", SCRIPT)
RP = importlib.util.module_from_spec(_spec)
sys.modules["reduce_preview"] = RP
_spec.loader.exec_module(RP)

POSITIONS = [{"exchangeId": "1076", "quantity": 1000, "costBasis": 900.0}, {"exchangeId": "1077", "quantity": -400, "costBasis": -360.0},
             {"exchangeId": "1065", "quantity": -200, "avgCost": 0.80}, {"exchangeId": "9", "quantity": 5, "settled": True}]
BOOKS = {
    "1076": {"bids": [{"price": 0.93, "quantity": 300}, {"price": 0.92, "quantity": 400}], "asks": [{"price": 0.95, "quantity": 100}]},
    "1077": {"bids": [{"price": 0.05, "quantity": 500}], "asks": [{"price": 0.07, "quantity": 100}, {"price": 0.08, "quantity": 1000}]},
    "1065": {"bids": [{"price": 0.15, "quantity": 50}], "asks": [{"price": 0.18, "quantity": 30}, {"price": 0.20, "quantity": 100}]},
}


def _mapping(eid, race, party, token):
    return MarketMapping(eid, f"m{eid}", f"{race} {party}", race, party, token, None)


RI, NH = Race("senate", "RI"), Race("house", "NH", "01")
MAP = MarketMap({"1076": _mapping("1076", RI, "D", "tD"), "1077": _mapping("1077", RI, "R", "tR"), "1065": _mapping("1065", NH, "R", "tN"),
                 "1066": _mapping("1066", NH, "D", "tN2")})


def test_held_position_uses_cost_basis_then_avg_cost():
    assert RP.held_position(POSITIONS, "1076") == {"side": "yes", "shares": 1000.0, "avg": 0.9}
    assert RP.held_position(POSITIONS, "1077") == {"side": "no", "shares": 400.0, "avg": 0.9}
    assert RP.held_position(POSITIONS, "1065") == {"side": "no", "shares": 200.0, "avg": 0.8}
    assert RP.held_position(POSITIONS, "9") is None and RP.held_position(POSITIONS, "404") is None


def test_saved_fair_ages_as_at_the_fetch_and_overrides():
    at = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    oracle = {"fetched_at": at.isoformat(), "max_age_s": 10.0, "polymarket": {"tD": {"mid": 0.95, "spread": 0.01, "fetched_at": at.timestamp() - 2}}, "kalshi": {}}
    assert RP.saved_fair(oracle, MAP.get("1076")) == (0.95, "ok")
    assert RP.saved_fair(oracle, MAP.get("1077")) == (None, "not in the saved oracle pass")
    assert RP.saved_fair(oracle, None) == (None, "not mapped")
    assert RP.parse_fair_overrides(["1076=0.9", " 1077 =0.1"]) == {"1076": 0.9, "1077": 0.1}
    with pytest.raises(ValueError):
        RP.parse_fair_overrides(["1076=1.2"])


def test_market_rows_numbers():
    held = RP.held_position(POSITIONS, "1076")  # 1000 YES @ 0.90; bids 300 @ 0.93, 400 @ 0.92; mid 0.94
    rows = {r["fraction"]: r for r in RP.market_rows("1076", "senate:RI", "D", ["D", "R"], held, BOOKS["1076"], 0.95)}
    half = rows[0.5]  # 500 shares: 300 @ 0.93 + 200 @ 0.92
    assert half["filled"] == 500 and half["depth_enough"] and half["levels_used"] == 2 and half["proceeds"] == pytest.approx(463.0)
    assert half["vwap"] == pytest.approx(0.926) and half["slip_vs_best"] == pytest.approx(2.0) and half["slip_vs_mid"] == pytest.approx(7.0)
    assert half["realized_pnl"] == pytest.approx(13.0) and half["lost_vs_oracle"] == pytest.approx(12.0)
    assert half["ev_hold"] == pytest.approx(50.0) and half["ev_after"] == pytest.approx(38.0)  # 463 + 500 x 0.95 - 900
    assert half["ev_after_shrunk"] == pytest.approx(463 + 500 * 0.93 - 900) and half["ev_hold_shrunk"] == pytest.approx(30.0)
    assert half["worst_before"] == pytest.approx(-900.0) and half["worst_after"] == pytest.approx(-437.0)
    assert half["break_even_yes_p"] == pytest.approx(0.926)
    full = rows[1.0]
    assert full["filled"] == 700 and not full["depth_enough"] and full["remaining_shares"] == 300
    no = {r["fraction"]: r for r in RP.market_rows("1077", "senate:RI", "R", ["D", "R"], RP.held_position(POSITIONS, "1077"), BOOKS["1077"], 0.04)}
    q = no[0.25]  # 100 NO sold at 1 - 0.07 = 0.93; p_side 0.96; break-even YES p = 0.07
    assert q["proceeds"] == pytest.approx(93.0) and q["p_side"] == 0.96 and q["break_even_yes_p"] == pytest.approx(0.07) and q["mid"] == pytest.approx(0.94)
    unpriced = RP.market_rows("1076", "senate:RI", "D", ["D", "R"], held, BOOKS["1076"], None)[0]
    assert unpriced["lost_vs_oracle"] is None and unpriced["ev_after"] is None and unpriced["worst_after"] is not None


def test_race_rows_combine_the_same_event():
    rows = (RP.market_rows("1076", "senate:RI", "D", ["D", "R"], RP.held_position(POSITIONS, "1076"), BOOKS["1076"], 0.95)
            + RP.market_rows("1077", "senate:RI", "R", ["D", "R"], RP.held_position(POSITIONS, "1077"), BOOKS["1077"], 0.04))
    combined = {r["fraction"]: r for r in RP.race_rows("senate:RI", ["D", "R"], rows)}
    zero_sale_worst = -(900.0 + 360.0)  # R wins: D YES and R NO both pay 0
    assert combined[0.25]["worst_before"] == pytest.approx(zero_sale_worst) and combined[0.25]["worst_winner"] == "R"
    by = {(r["exchange_id"], r["fraction"]): r for r in rows}
    assert combined[0.5]["proceeds"] == pytest.approx(by[("1076", 0.5)]["proceeds"] + by[("1077", 0.5)]["proceeds"])
    assert combined[0.5]["worst_after"] == pytest.approx(zero_sale_worst + combined[0.5]["proceeds"])
    totals = {t["fraction"]: t for t in RP.spread_totals(rows)}
    assert totals[1.0]["depth_short"] == ["1076"]
    assert totals[0.5]["slip_vs_mid"] == pytest.approx(by[("1076", 0.5)]["slip_vs_mid"] + by[("1077", 0.5)]["slip_vs_mid"])


class FakeReader:
    def __init__(self, out_dir):
        self.out_dir, self.calls = out_dir, []

    async def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/portfolio/positions":
            return {"positions": POSITIONS}
        return BOOKS[path.split("/")[2]]


@pytest.fixture
def env(monkeypatch, tmp_path):
    import config

    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(RP, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(RP, "load_market_map", lambda: MAP)
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    reader = FakeReader(tmp_path / "out")
    reader.out_dir.mkdir()

    @asynccontextmanager
    async def fake_reader(script, reads_per_min):
        assert script == "reduce_preview" and reads_per_min == 10.0
        yield reader

    monkeypatch.setattr(RP, "research_reader", fake_reader)
    at = datetime.now(timezone.utc) - timedelta(hours=5)
    oracle = tmp_path / "oracle_quotes.json"
    oracle.write_text(json.dumps({"fetched_at": at.isoformat(), "max_age_s": 10.0, "kalshi": {},
                                  "polymarket": {"tD": {"mid": 0.95, "spread": 0.01, "fetched_at": at.timestamp() - 1},
                                                 "tN": {"mid": 0.15, "spread": 0.02, "fetched_at": at.timestamp() - 1}}}), encoding="utf-8")
    return reader, oracle


def test_main_needs_yes(env):
    reader, oracle = env
    assert RP.main(["--oracle", str(oracle)]) == 0
    assert reader.calls == []


def test_main_reads_positions_and_one_book_per_contract(env):
    reader, oracle = env
    assert RP.main(["--yes", "--oracle", str(oracle), "--fair", "1077=0.04"]) == 0
    assert reader.calls == [("/portfolio/positions", None), *((f"/exchanges/{eid}/orderbook", {"depth": 50}) for eid in ("1076", "1077", "1065"))]
    with (reader.out_dir / "by_market.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12 and {r["fair_yes"] for r in rows if r["exchange_id"] == "1077"} == {"0.04"}
    with (reader.out_dir / "by_race.csv").open(encoding="utf-8") as fh:
        races = list(csv.DictReader(fh))
    assert {r["race"] for r in races} == {"senate:RI"} and len(races) == 4  # NH-01: only one held market
    summary = json.loads((reader.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "MANUAL" in summary["note"] and "h old" in summary["oracle"] and len(summary["spread_cost"]) == 4
    report = (reader.out_dir / "report.md").read_text(encoding="utf-8")
    assert "## senate:RI combined" in report and "## 1065 house:NH-01 R NO" in report


def test_script_is_read_only():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "edge_scan"}
    source = SCRIPT.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in forbidden for a in node.names)
    assert "place_" not in source and "cancel" not in source
