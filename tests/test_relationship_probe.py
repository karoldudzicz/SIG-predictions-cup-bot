"""scripts/relationship_probe.py on synthetic data with a fake GET-only reader - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "relationship_probe.py"
_spec = importlib.util.spec_from_file_location("relationship_probe", SCRIPT)
P = importlib.util.module_from_spec(_spec)
sys.modules["relationship_probe"] = P
_spec.loader.exec_module(P)


def _m(eid, state, party):
    return MarketMapping(eid, eid, f"{party} {state}", Race("senate", state), party, None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [
    _m("1", "AZ", "D"), _m("2", "AZ", "R"),
    _m("3", "GA", "D"), _m("4", "GA", "R"), _m("5", "GA", "I"),
    _m("6", "NV", "D"),  # single party: not a basket
]})
RACES = P.race_legs(MAP)


def _constraint(rid, status, amount, ids, rtype="mutually_exclusive"):
    return {"relationshipId": rid, "type": rtype, "evaluationStatus": status, "violationAmount": amount,
            "observedPrices": [{"exchangeId": e, "price": 0.5} for e in ids], "suggestedCorrectiveTrades": [{"exchangeId": ids[0], "action": "sell", "outcomeSide": "yes"}]}


# ---------------------------------------------------------------------- pure parts


def test_race_legs_and_packing_never_split_a_race():
    assert RACES == {"senate:AZ": ["1", "2"], "senate:GA": ["3", "4", "5"]}
    chunks = P.pack_races({"a": ["1", "2"], "b": ["3", "4", "5"], "c": ["6", "7"]}, max_ids=4)
    assert chunks == [["1", "2"], ["3", "4", "5"], ["6", "7"]]
    assert P.pack_races({"a": ["1", "2"], "c": ["6", "7"]}, max_ids=4) == [["1", "2", "6", "7"]]
    with pytest.raises(ValueError):
        P.pack_races({"big": ["1", "2", "3"]}, max_ids=2)


def test_estimate_counts_samples_and_reads():
    est, samples = P.estimate(RACES, 60, 60, 10)
    assert samples == 61 and est.reads == 1 + 61 * 2  # constraints + one prices chunk per sample
    assert est.runtime_s == pytest.approx(60 * 60)  # wall time dominated by the sampling window
    snap, n = P.estimate(RACES, 0, 60, 10)
    assert n == 1 and snap.reads == 3


def test_constraint_race_mapping():
    assert P.constraint_race(_constraint("r", "violated", 0.02, ["1", "2"]), MAP) == "senate:AZ"
    assert P.constraint_race(_constraint("r", "violated", 0.02, ["1", "3"]), MAP) == "cross:senate:AZ+senate:GA"
    assert P.constraint_race(_constraint("r", "violated", 0.02, ["1", "999"]), MAP) == "unmapped"
    assert P.constraint_race({"relationshipId": "x"}, MAP) == "unmapped"


def test_engine_by_race_worst_status_and_max_violation():
    eng = P.engine_by_race([_constraint("a", "satisfied", 0.0, ["1", "2"]), _constraint("b", "violated", 0.03, ["1", "2"]),
                            _constraint("c", "violated", 0.01, ["2", "1"], "complementary"), _constraint("d", "unevaluable", 0.0, ["3", "4"])], MAP)
    assert eng["senate:AZ"] == {"status": "violated", "violation": 0.03, "constraints": 3, "types": ["complementary", "mutually_exclusive"]}
    assert eng["senate:GA"]["status"] == "unevaluable" and eng["senate:GA"]["violation"] == 0.0


def test_race_rows_need_every_leg_quoted():
    prices = {"1": {"bestBid": 0.50, "bestAsk": 0.53}, "2": {"bestBid": 0.52, "bestAsk": 0.55},
              "3": {"bestBid": 0.60, "bestAsk": None}, "4": {"bestBid": 0.30, "bestAsk": 0.33}, "5": {"bestBid": 0.02, "bestAsk": 0.04}}
    rows = {r["race"]: r for r in P.race_rows("2026-10-09T12:00:00Z", RACES, prices, {})}
    assert rows["senate:AZ"]["sum_bid"] == pytest.approx(1.02) and rows["senate:AZ"]["sum_ask"] == pytest.approx(1.08)
    assert rows["senate:AZ"]["spread_cost"] == pytest.approx(0.06) and rows["senate:AZ"]["engine_status"] is None
    assert rows["senate:GA"]["sum_bid"] == pytest.approx(0.92) and rows["senate:GA"]["sum_ask"] is None and rows["senate:GA"]["spread_cost"] is None


# ---------------------------------------------------------------------- analysis


def _row(ts, race, sum_bid, sum_ask, engine=None):
    return {"ts": ts, "race": race, "legs": 2, "sum_bid": sum_bid, "sum_ask": sum_ask,
            "spread_cost": None if sum_bid is None or sum_ask is None else sum_ask - sum_bid,
            "engine_status": None if engine is None else ("violated" if engine > 0 else "satisfied"), "engine_violation": engine}


def _iso(t):
    return t.isoformat().replace("+00:00", "Z")


def test_analyse_race_premium_deviation_and_episodes():
    t0 = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    # sum_ask - 1 sits at +0.03; minutes 3-4 dip to -0.005 (executable buy basket for 60 s).
    asks = [1.03, 1.03, 1.03, 0.995, 0.995, 1.03, 1.03]
    rows = [_row(_iso(t0 + timedelta(minutes=i)), "r", 0.97, a, engine=0.0) for i, a in enumerate(asks)]
    out = P.analyse_race(rows, 60)
    assert out["premium_ask"] == pytest.approx(0.03) and out["premium_bid"] == pytest.approx(-0.03)
    assert out["ask_cheap_1tick_share"] == pytest.approx(2 / 7, abs=1e-4)  # 0.035 below the premium
    assert out["exec_ask_under_share"] == pytest.approx(2 / 7, abs=1e-4) and out["exec_bid_over_share"] == 0
    assert out["episodes"] == 1 and out["median_duration_s"] == 60 and out["episodes_gt_30s"] == 1
    assert out["long_episodes"][0]["kinds"] == ["ask_under"] and out["long_episodes"][0]["day"] == "2026-10-09"
    assert out["engine_covered_share"] == 1.0 and out["engine_gt_tick_share"] == 0.0


def test_single_sighting_never_counts_as_longer_than_30s():
    t0 = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    rows = [_row(_iso(t0 + timedelta(minutes=i)), "r", b, 1.05) for i, b in enumerate([0.99, 1.01, 0.99])]
    out = P.analyse_race(rows, 60)
    assert out["episodes"] == 1 and out["median_duration_s"] == 0 and out["episodes_gt_30s"] == 0


def _samples_for_day(day, races_with_edge, n=5):
    """n samples a minute apart; the listed races have sum_bid 1.01 for the first 3."""
    t0 = datetime(2026, 10, day, 12, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        ts = _iso(t0 + timedelta(minutes=i))
        rows = [_row(ts, race, 1.01 if race in races_with_edge and i < 3 else 0.97, 1.04) for race in ("a", "b", "c", "d")]
        out.append({"ts": ts, "interval_s": 60, "rows": rows})
    return out


def test_verdict_pass_fail_and_insufficient():
    one_day = P.analyse(_samples_for_day(9, {"a", "b", "c"}))
    assert one_day["verdict"] == "INSUFFICIENT DATA" and one_day["utc_days"] == ["2026-10-09"]
    passing = P.analyse(_samples_for_day(9, {"a", "b", "c"}) + _samples_for_day(10, {"a", "b", "d"}))
    assert passing["verdict"] == "PASS"
    assert passing["qualifying_races_per_day"] == {"2026-10-09": ["a", "b", "c"], "2026-10-10": ["a", "b", "d"]}
    assert passing["halves"]["first_half"]["races_with_executable_gt_30s"] == ["a", "b", "c"]
    assert passing["hypotheses_tested"]["total"] == 8
    failing = P.analyse(_samples_for_day(9, {"a", "b", "c"}) + _samples_for_day(10, {"a", "b"}))
    assert failing["verdict"] == "FAIL"
    assert P.analyse([])["verdict"] == "INSUFFICIENT DATA"


# ---------------------------------------------------------------------- full run


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeReader:
    def __init__(self, tmp_path, price_seq):
        self.out_dir = tmp_path
        self.calls = []
        self.price_seq = list(price_seq)

    def stats(self):
        return {"requests": len(self.calls)}

    async def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if path == "/tournaments/t":
            return {"id": "tid"}
        if path == "/relationships/constraints":
            assert params["violationsOnly"] is False and params["tournamentId"] == "tid"
            return {"data": [_constraint("rel-1", "violated", 0.02, ["1", "2"])], "violationsCount": 1, "computedAt": "x"}
        if path == "/exchanges/prices":
            prices = self.price_seq.pop(0)
            return {"data": [{"exchangeId": e, **prices[e]} for e in params["ids"].split(",") if e in prices]}
        raise AssertionError(f"unexpected GET {path}")


def test_run_samples_writes_jsonl_and_report(tmp_path, caplog):
    quote = {"1": {"bestBid": 0.50, "bestAsk": 0.53}, "2": {"bestBid": 0.52, "bestAsk": 0.55}, "3": {"bestBid": 0.6, "bestAsk": 0.66},  # GA asks sum 1.01: no edge
             "4": {"bestBid": 0.3, "bestAsk": 0.32}, "5": {"bestBid": 0.02, "bestAsk": 0.03}}
    reader = FakeReader(tmp_path, [quote, quote, quote])
    clock = FakeClock()
    stamps = iter(["2026-10-09T12:00:00Z", "2026-10-09T12:01:00Z", "2026-10-09T12:02:00Z"])
    with caplog.at_level("INFO"):
        report = asyncio.run(P.run(reader, "t", MAP, samples=3, interval_s=60, clock=clock, sleep=clock.sleep, now_iso=lambda: next(stamps)))
    assert [c[0] for c in reader.calls] == ["/tournaments/t"] + ["/relationships/constraints", "/exchanges/prices"] * 3
    assert clock.sleeps == [60, 60]
    lines = [json.loads(l) for l in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and lines[0]["violated"][0]["race"] == "senate:AZ"
    az = next(r for r in lines[0]["rows"] if r["race"] == "senate:AZ")
    assert az["sum_bid"] == pytest.approx(1.02) and az["engine_violation"] == 0.02
    # The violated constraint is logged once, not every sample.
    assert sum(1 for r in caplog.records if r.getMessage() == "Violated constraint") == 1
    assert report["verdict"] == "INSUFFICIENT DATA" and report["pooled"]["races_with_executable_gt_30s"] == ["senate:AZ"]
    assert (tmp_path / "races.csv").exists() and (tmp_path / "summary.json").exists()


def test_run_keeps_going_after_a_failed_sample(tmp_path):
    from exchange_api import ApiError

    class Flaky(FakeReader):
        async def get(self, path, params=None):
            if path == "/exchanges/prices" and len([c for c in self.calls if c[0] == "/exchanges/prices"]) == 0:
                self.calls.append((path, params))
                raise ApiError(503, "UNAVAILABLE", "down")
            return await super().get(path, params)

    quote = {e: {"bestBid": 0.4, "bestAsk": 0.45} for e in "12345"}
    reader = Flaky(tmp_path, [quote])
    clock = FakeClock()
    report = asyncio.run(P.run(reader, "t", MAP, samples=2, interval_s=60, clock=clock, sleep=clock.sleep, now_iso=lambda: "2026-10-09T12:00:00Z"))
    assert report["samples"] == 1


# ---------------------------------------------------------------------- CLI


@pytest.fixture
def quiet_main(monkeypatch, tmp_path):
    import config

    settings = config.Settings(sig_api_key="test-key", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(P, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(P.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)


def test_without_yes_nothing_is_fetched(quiet_main, monkeypatch, caplog):
    monkeypatch.setattr(P, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reader without --yes")))
    with caplog.at_level("INFO"):
        assert P.main(["--sample-minutes", "60"]) == 0
    planned = next(r for r in caplog.records if r.getMessage() == "Planned SIG reads")
    assert planned.reads == 1 + 61 * 2


def test_interval_too_short_for_the_read_budget_is_refused(quiet_main):
    with pytest.raises(SystemExit):
        P.main(["--sample-minutes", "10", "--interval-s", "5"])  # 2 reads per sample need >= 12 s at 10/min


def test_analyse_only_makes_no_request(quiet_main, monkeypatch, tmp_path):
    root = tmp_path / "relationship_probe"
    (root / "2026-10-09").mkdir(parents=True)
    (root / "2026-10-09" / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in _samples_for_day(9, {"a"})) + "\n", encoding="utf-8")
    monkeypatch.setattr(P, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(P, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reader in --analyse-only")))
    assert P.main(["--analyse-only"]) == 0
    assert json.loads((root / "pooled" / "summary.json").read_text(encoding="utf-8"))["samples"] == 5


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
