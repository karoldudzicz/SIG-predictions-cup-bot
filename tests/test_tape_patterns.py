"""scripts/tape_patterns.py on a synthetic tape with a planted pattern - no network.

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

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tape_patterns.py"
_spec = importlib.util.spec_from_file_location("tape_patterns", SCRIPT)
T = importlib.util.module_from_spec(_spec)
sys.modules["tape_patterns"] = T
_spec.loader.exec_module(T)

DAY0 = datetime(2026, 10, 2, tzinfo=timezone.utc)


def _iso(t):
    return t.isoformat().replace("+00:00", "Z")


def synthetic_tape(days=4, planted=True, start=DAY0):
    """Noise every 90 s (sizes cycle over 40 values, price wobbles around a level) from
    10:00 to 18:00 UTC; if `planted`, a size-137 buy at :05 / :25 / :45 that lifts the level."""
    trades, n = [], 0
    level = 0.20
    for d in range(days):
        base = start + timedelta(days=d, hours=10)
        events = []
        for k in range(320):
            events.append((base + timedelta(seconds=90 * k), "noise", k))
        if planted:
            for k in range(24):
                events.append((base + timedelta(minutes=5 + 20 * k, seconds=7), "bot", k))
        for t, kind, k in sorted(events):
            n += 1
            if kind == "bot":
                price, size = level + 0.004, 137
                level += 0.006
            else:
                price, size = level + (0.001 if k % 2 else 0.0), 10 * (1 + k % 40)
            trades.append({"id": str(n), "createdAt": _iso(t), "price": round(price, 6), "size": size, "side": "YES" if k % 3 else "NO", "volume": size * price})
    return trades


# ---------------------------------------------------------------------- selection


def test_universe_activity_and_cheap():
    pages = [{"data": [{"id": "m1", "title": "Seats > 50", "exchanges": [{"id": "1", "option": "Yes", "latestPrice": 0.4}]},
                       {"id": "m2", "title": "AZ", "exchanges": [{"id": "2", "option": None}, {"id": "3"}]}]}]
    assert T.universe_from_markets(pages)["1"] == {"title": "Seats > 50", "option": "Yes", "market_id": "m1", "latest_price": 0.4}
    as_of = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    candles = [{"time": "2026-10-10T10:00:00Z", "close": 0.5, "tradeCount": 7}, {"time": "2026-10-12T11:00:00Z", "close": 0.1, "tradeCount": 3},
               {"time": "2026-10-11T12:00:00Z", "close": None, "tradeCount": 1}]
    assert T.activity_from_candles(candles, as_of) == {"trades_total": 11, "trades_24h": 4, "last_close": 0.1}
    assert T.is_cheap(0.02) and T.is_cheap(0.98) and T.is_cheap(0.90) and not T.is_cheap(0.5) and not T.is_cheap(0.01) and not T.is_cheap(None)


def test_select_exchanges_quotas_no_duplicates_and_fill():
    activity = {str(i): {"trades_total": 1000 - i, "trades_24h": i, "last_close": 0.5} for i in range(1, 60)}
    activity["58"].update(last_close=0.05, trades_24h=0)  # cheap, barely traded overall, not in the 24 h top
    activity["99"] = {"trades_total": 0, "trades_24h": 0, "last_close": 0.05}  # never traded: never selected
    chosen = T.select_exchanges(activity)
    ids = [e for e, _ in chosen]
    assert len(ids) == 40 and len(set(ids)) == 40 and "99" not in ids
    reasons = dict(chosen)
    assert reasons["1"] == "active" and reasons["59"] == "active_24h" and reasons["58"] == "cheap"
    assert sum(1 for r in reasons.values() if r == "fill") == 9  # 9 cheap slots unfilled -> overall ranking


def test_tape_fields_reports_what_is_there():
    out = T.tape_fields([{"id": "1", "price": 0.4, "size": 3, "side": "YES"}, {"id": "2", "price": None, "size": 1, "side": "NO", "takerSide": "buy"}])
    assert out["fields"]["price"] == {"float": 1, "NoneType": 1}
    assert out["aggressor_like"] == ["takerSide"] and out["identity_like"] == []


# ---------------------------------------------------------------------- patterns


def test_size_key_is_exact():
    assert T.size_key(1234567.0) == "1234567" and T.size_key(137) == "137" and T.size_key(2.5) == "2.5"


def test_planted_bot_is_a_candidate_and_noise_is_not():
    as_of = DAY0 + timedelta(days=4)
    result = T.analyse({"bot": synthetic_tape(), "noise": synthetic_tape(planted=False)}, {"bot": "B", "noise": "N"}, as_of)
    cands = {(c["exchange_id"], c["window"], c["kind"], c["key"], c["horizon_s"]) for c in result["candidates"]}
    assert ("bot", "full", "size", "137", 300) in cands and ("bot", "full", "size", "137", 60) in cands
    assert not any(c[0] == "noise" for c in cands)
    size_row = next(p for p in result["patterns"] if p["exchange_id"] == "bot" and p["kind"] == "size" and p["key"] == "137" and p["horizon_s"] == 300)
    assert size_row["events"] == 96 and size_row["days"] == 4 and size_row["mean_drift"] > 0 > size_row["baseline_drift"]
    assert result["hypotheses_tested"] == len(result["patterns"]) > 0
    # The regular 90 s noise is detected as a gap spike, but its drift equals its own baseline.
    kinds = {(p["exchange_id"], p["kind"]) for p in result["patterns"]}
    assert ("noise", "gap") in kinds
    by_day = [r for r in result["candidates_by_day"] if r["exchange_id"] == "bot" and r["key"] == "137" and r["horizon_s"] == 300 and r["window"] == "full"]
    assert len(by_day) == 4 and all(r["events"] == 24 for r in by_day)


def test_roundness_rows_per_day_and_expected_share():
    series = T.Series.from_trades([{"id": str(i), "createdAt": _iso(DAY0 + timedelta(hours=i)), "price": 0.5, "size": 2000 + i, "side": "YES", "volume": 1000.0 + i}
                                   for i in range(30)])
    rows = T.roundness_rows("x", "full", series)
    all_exec_1000 = next(r for r in rows if r["day"] == "ALL" and r["basis"] == "exec_volume" and r["multiple"] == 1000)
    assert all_exec_1000["trades"] == 30 and all_exec_1000["round"] == 6  # 1000..1005 within 0.5%
    assert all_exec_1000["expected_share"] == pytest.approx(0.01) and all_exec_1000["excess"] == pytest.approx(20.0)
    assert {r["day"] for r in rows} == {"ALL", "2026-10-02", "2026-10-03"}


def test_extremum_and_skew_rows_label_proxy_shares():
    series = T.Series.from_trades(synthetic_tape(days=2))
    ext = T.extremum_rows("bot", "full", series, [137.0])
    assert ext[0]["trades"] == 48 and ext[0]["near_extreme_share"] == 1.0  # each bot print is a new day high
    skew = {r["size_bucket"]: r for r in T.skew_rows("bot", "full", series)}
    assert skew["100-499"]["trades"] > 0 and skew["3000+"]["trades"] == 0 and skew["3000+"]["proxy_buy_share"] is None


def test_is_candidate_needs_same_sign_in_both_halves():
    base = {"events": 40, "days": 3}
    assert T.is_candidate({**base, "t_first": 2.5, "t_second": 2.1})
    assert not T.is_candidate({**base, "t_first": 2.5, "t_second": -2.1})
    assert not T.is_candidate({**base, "t_first": 2.5, "t_second": 1.9})
    assert not T.is_candidate({"events": 19, "days": 3, "t_first": 5, "t_second": 5})
    assert not T.is_candidate({"events": 40, "days": 1, "t_first": 5, "t_second": 5})
    assert not T.is_candidate({**base, "t_first": None, "t_second": 3})


# ---------------------------------------------------------------------- freeze and retest


def test_freeze_and_retest_only_uses_data_after_the_freeze():
    as_of = DAY0 + timedelta(days=4)
    result = T.analyse({"bot": synthetic_tape()}, {}, as_of)
    frozen = T.freeze(result["candidates"], as_of, result["hypotheses_tested"])
    assert frozen["data_through"] == "2026-10-06T00:00:00Z" and frozen["candidates"] and "PROXY" in frozen["aggressor"]
    later_bot = synthetic_tape(days=2, start=as_of)
    rows = T.retest(frozen, {"bot": synthetic_tape() + later_bot})  # old trades are ignored
    size_rows = [r for r in rows if r["kind"] == "size" and r["key"] == "137"]
    assert size_rows and all(r["verdict"] == "REPLICATED" and r["events"] == 48 for r in size_rows)
    gone = T.retest(frozen, {"bot": synthetic_tape(days=2, planted=False, start=as_of)})
    assert all(r["verdict"] in ("INSUFFICIENT DATA", "FAILED") for r in gone)
    assert T.retest(frozen, {})[0]["verdict"] == "INSUFFICIENT DATA"


# ---------------------------------------------------------------------- full run (fake reader)


class FakeReader:
    reads_per_min = 10.0

    def __init__(self, tmp_path):
        self.out_dir = tmp_path
        self.calls = []
        self.bot = synthetic_tape()

    def stats(self):
        return {"requests": len(self.calls)}

    async def get(self, path, params=None):
        params = dict(params or {})
        self.calls.append((path, params))
        if path == "/tournaments/t":
            return {"id": "tid"}
        if path == "/tournaments/t/markets":
            return {"data": [{"id": "m", "title": "Senate AZ", "exchanges": [{"id": "1"}, {"id": "2"}, {"id": "3"}]}], "pagination": {"hasMore": False}}
        if path.endswith("/price-history"):
            assert params["resolution"] == "1h" and params["limit"] == 1000
            eid = path.split("/")[2]
            return {"candles": [] if eid == "3" else [{"time": "2026-10-05T12:00:00Z", "close": 0.5 if eid == "1" else 0.05, "tradeCount": 1000}]}
        if path.endswith("/trades"):
            eid = path.split("/")[2]
            data = self.bot if eid == "1" else synthetic_tape(days=1, planted=False)
            start = int(params["cursor"] or 0)
            page = data[::-1][start : start + 200]  # newest first, like the API
            more = start + 200 < len(data)
            return {"data": page, "pagination": {"hasMore": more, "nextCursor": str(start + 200) if more else None}}
        raise AssertionError(f"unexpected GET {path}")


def test_run_end_to_end(tmp_path, caplog):
    reader = FakeReader(tmp_path)
    with caplog.at_level("INFO"):
        result = asyncio.run(T.run(reader, "t", max_pages=3))
    trade_calls = [c for c in reader.calls if c[0].endswith("/trades")]
    assert {c[0] for c in trade_calls} == {"/exchanges/1/trades", "/exchanges/2/trades"}  # 3 never traded
    assert sum(1 for c in trade_calls if c[0] == "/exchanges/1/trades") == 3  # capped by --max-pages
    assert any(r.getMessage() == "Tape fields (first deep page)" for r in caplog.records)
    for name in ("tape_fields.json", "exchanges.csv", "patterns.csv", "candidates.csv", "candidates_by_day.csv", "roundness.csv",
                 "extremum.csv", "side_skew.csv", "frozen_candidates.json", "summary.json"):
        assert (tmp_path / name).exists(), name
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["hypotheses_tested"] == result["hypotheses_tested"] and "PROXY" in summary["aggressor"]
    exchanges = (tmp_path / "exchanges.csv").read_text(encoding="utf-8")
    assert "cheap" in exchanges.splitlines()[0]


def test_inputs_from_raw_rebuilds_the_run(tmp_path):
    records = [
        {"ts": "2026-10-06T00:00:00Z", "path": "/tournaments/t/markets", "body": {"data": [{"id": "m", "title": "X", "exchanges": [{"id": "1"}]}]}},
        {"ts": "2026-10-06T00:00:01Z", "path": "/exchanges/1/price-history", "body": {"candles": [{"time": "2026-10-05T12:00:00Z", "close": 0.4, "tradeCount": 5}]}},
        {"ts": "2026-10-06T00:00:02Z", "path": "/exchanges/1/trades", "body": {"data": [{"id": "a", "createdAt": "2026-10-05T12:00:00Z", "price": 0.4, "size": 1}]}},
        {"ts": "2026-10-06T00:00:03Z", "path": "/exchanges/1/trades", "body": {"data": [{"id": "a", "createdAt": "2026-10-05T12:00:00Z", "price": 0.4, "size": 1}]}},
        {"ts": "2026-10-06T00:00:04Z", "path": "/exchanges/2/trades", "error": {"status": 500}},
    ]
    inp = T.inputs_from_raw(records)
    assert inp["universe"]["1"]["title"] == "X" and len(inp["trades"]["1"]) == 1 and inp["candles"]["1"][0]["tradeCount"] == 5
    assert inp["as_of"] == datetime(2026, 10, 6, 0, 0, 3, tzinfo=timezone.utc)  # the last SUCCESSFUL read


# ---------------------------------------------------------------------- CLI


@pytest.fixture
def quiet_main(monkeypatch, tmp_path):
    import config

    settings = config.Settings(sig_api_key="test-key", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(T, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(T, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reader without --yes")))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)


def test_without_yes_nothing_is_fetched(quiet_main, caplog):
    with caplog.at_level("INFO"):
        assert T.main([]) == 0
    planned = next(r for r in caplog.records if r.getMessage() == "Planned SIG reads")
    assert planned.reads == 1 + 3 + 240 + 40 * 20


def test_analyse_only_from_raw_makes_no_request(quiet_main, tmp_path):
    raw = tmp_path / "raw.jsonl"
    bot = synthetic_tape()
    records = [{"ts": "2026-10-06T00:00:00Z", "path": "/tournaments/t/markets", "body": {"data": [{"id": "m", "title": "X", "exchanges": [{"id": "1"}]}]}},
               {"ts": "2026-10-06T00:00:01Z", "path": "/exchanges/1/price-history", "body": {"candles": [{"time": "2026-10-05T12:00:00Z", "close": 0.5, "tradeCount": 1300}]}},
               {"ts": "2026-10-06T00:00:02Z", "path": "/exchanges/1/trades", "body": {"data": bot}}]
    raw.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    assert T.main(["--analyse-only", "--raw", str(raw)]) == 0
    frozen = json.loads((tmp_path / "reanalysis" / "frozen_candidates.json").read_text(encoding="utf-8"))
    assert any(c["kind"] == "size" and c["key"] == "137" for c in frozen["candidates"])


def test_retest_without_candidates_stops(quiet_main, tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"data_through": "2026-10-06T00:00:00Z", "candidates": []}), encoding="utf-8")
    assert T.main(["--retest", str(path)]) == 0


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
