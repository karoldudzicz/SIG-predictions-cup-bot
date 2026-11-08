"""scripts/position_review.py with fake reads, saved raw and fake oracles - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote
from risk import RiskLimits

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "position_review.py"
_spec = importlib.util.spec_from_file_location("position_review", SCRIPT)
P = importlib.util.module_from_spec(_spec)
sys.modules["position_review"] = P
_spec.loader.exec_module(P)


def _m(eid, office, state, party, mid):
    return MarketMapping(eid, mid, f"{party} {state}", Race(office, state), party, f"tok-{eid}", None)


MAP = MarketMap({m.sig_exchange_id: m for m in [
    _m("1", "senate", "RI", "D", "m1"), _m("2", "senate", "RI", "R", "m2"),
    _m("3", "senate", "AZ", "D", "m3"), _m("4", "senate", "AZ", "R", "m4"),
    _m("5", "house_control", None, "D", "m5"),
]})
RAW_MAP = {"entries": {
    "1": {"polymarket": {"event_slug": "rhode-island-senate-election-winner", "label": "Jack Reed (D)", "token_id": "tok-1"},
          "kalshi": {"event_ticker": "SENATERI-26", "label": "Jack Reed", "ticker": "SENATERI-26-D"}, "check_prices": {"polymarket": 0.96}},
    "2": {"polymarket": {"event_slug": "rhode-island-senate-election-winner", "label": "Allan Fung (D)", "token_id": "tok-2"}},  # wrong party suffix
    "3": {"kalshi": {"event_ticker": "SENATEAZ-26", "label": "Ruben Gallego", "ticker": "SENATEAZ-26-D"}},
    "4": {"polymarket": {"event_slug": "arizona-senate-election-winner", "label": "Kari Lake", "token_id": "tok-4"}},  # no party in the label
    "5": {"polymarket": {"event_slug": "which-party-will-win-the-house-in-2026", "label": "Democratic Party", "token_id": "tok-5"}},
}}
POSITIONS = [
    {"exchangeId": "1", "marketId": "m1", "quantity": 1000, "avgCost": 0.95, "costBasis": 950.0},  # YES D RI
    {"exchangeId": "2", "marketId": "m2", "quantity": -1000, "avgCost": 0.06, "costBasis": 60.0},  # NO R RI: the same bet as YES D
    {"exchangeId": "3", "marketId": "m3", "quantity": -500, "avgCost": 0.40, "costBasis": 200.0},  # NO D AZ = long R
    {"exchangeId": "4", "marketId": "m4", "quantity": 0, "avgCost": 0.0, "costBasis": 0.0},  # flat: ignored
]


def _detail(mid, eid, title, bid, ask, bid_q=500, ask_q=700):
    return {"id": mid, "title": title, "exchanges": [{"id": eid, "bestBid": bid, "bestAsk": ask, "bids": [{"price": bid, "quantity": bid_q}], "asks": [{"price": ask, "quantity": ask_q}]}]}


def _party_node(title, party):
    return {"market_id": "x", "root": {"node_type": "contract", "title": title, "contract_type": "Election Outcome", "settlement_date": "2026-11-04T00:00:00Z",
                                       "settled_with": None, "contract_details": {"resolutionType": "Party Winner", "party": party}}}


DETAILS = {
    "m1": _detail("m1", "1", "Will the Democratic candidate win the 2026 Rhode Island Senate race?", 0.92, 0.94),
    "m2": _detail("m2", "2", "Will the Republican candidate win the 2026 Rhode Island Senate race?", 0.05, 0.07),
    "m3": _detail("m3", "3", "Will the Democratic candidate win the 2026 Arizona Senate race?", 0.55, 0.59),
    "m4": _detail("m4", "4", "Will the Republican candidate win the 2026 Arizona Senate race?", 0.40, 0.44),
}
NODES = {"m1": _party_node("Democrat wins RI Senate", "Democratic"), "m2": _party_node("Republican wins RI Senate", "Republican"),
         "m3": _party_node("Democrat wins AZ Senate", "Democratic"), "m4": _party_node("Republican wins AZ Senate", "Republican")}
JOURNAL = [
    {"key": "a", "status": "accepted", "payload": {"exchangeId": "1"}, "result": {"side": "yes", "action": "buy", "price": 0.94, "quantityTraded": 600}},
    {"key": "b", "status": "accepted", "payload": {"exchangeId": "1"}, "result": {"side": "yes", "action": "buy", "price": 0.96, "quantityTraded": 400}},
    {"key": "c", "status": "accepted", "payload": {"exchangeId": "1"}, "result": {"side": "no", "action": "buy", "price": 0.05, "quantityTraded": 10}},
    {"key": "d", "status": "dry_run", "payload": {"exchangeId": "1"}},
]
ORACLE_MIDS = {"1": 0.97, "2": 0.03, "3": 0.50, "4": 0.50}


def _inputs(mids=ORACLE_MIDS):
    return P.Inputs(10_000.0, POSITIONS, [], DETAILS, NODES, dict(mids), "test")


def _q(ref, bid, ask):
    return OracleQuote("polymarket", ref, bid, ask, 100, 100, None, 0.0)


# ---------------------------------------------------------------------- pure parts


def test_scope_races_plus_held():
    s = P.scope(MAP, ["senate:RI"], POSITIONS)
    assert s == {"1": {"market_id": "m1", "why": "replay race + held"}, "2": {"market_id": "m2", "why": "replay race + held"}, "3": {"market_id": "m3", "why": "held"}}


def test_races_from_replay(tmp_path):
    path = tmp_path / "by_race.csv"
    path.write_text("source,group,fills,shares,cost,pnl,unvalued,pnl_ticks_per_share\n"
                    "dry_run,senate:RI,3,100,50,-80,0,-1\ndry_run,senate:FL,3,100,50,-20,0,-1\nlive,senate:XX,1,1,1,-999,0,-1\n"
                    "dry_run,senate:AZ,3,100,50,15,0,1\ndry_run,senate:GA,1,1,1,,1,\n", encoding="utf-8")
    assert P.races_from_replay(path) == ["senate:RI", "senate:FL"]


def test_rules_text_and_resolution_type():
    assert P.resolution_type(NODES["m1"]) == "Party Winner" and P.is_party_winner("Party Winner") and P.is_party_winner("party_winner")
    nested = {"root": {"node_type": "operator", "operator": "NOT", "children": [{"node_type": "contract", "title": "x", "contract_details": {"resolutionType": "Chamber Control"}}]}}
    assert P.resolution_type(nested) == "Chamber Control" and not P.is_party_winner("Chamber Control")
    assert P.rules_text(nested).startswith("NOT | [None] x") and P.resolution_type(None) is None


def test_party_flag_compares_party_only():
    def flag(eid, party, title="Will the Democratic candidate win the 2026 Rhode Island Senate race?"):
        return P.party_flag(title, P.oracle_entry(RAW_MAP, eid), party)

    one = flag("1", "D")  # "Jack Reed (D)" and "-D"
    assert (one["sig_party"], one["party_check"], one["reasons"]) == ("D", "match", [])
    two = flag("2", "R", "Will the Republican candidate win the 2026 Rhode Island Senate race?")
    assert two["party_check"] == "mismatch" and two["reasons"] == ["party: SIG R vs polymarket D"]
    assert flag("9", "D")["party_check"] == "no oracle"


def test_party_flag_regression_candidate_name_is_not_a_mismatch():
    """Before: a SIG title naming the party and an oracle label naming only the candidate
    ("Kari Lake") were flagged. Now the label has no party: "unknown", not a flag."""
    out = P.party_flag("Will the Republican candidate win the 2026 Arizona Senate race?", P.oracle_entry(RAW_MAP, "4"), "R")
    assert out["party_check"] == "unknown" and out["reasons"] == []
    rows = {r["exchange_id"]: r for r in P.build_rows(_inputs(), MAP, RAW_MAP, ["senate:AZ"], JOURNAL)}
    assert rows["4"]["flag"] is False and rows["4"]["party_check"] == "unknown"


def test_book_and_position_values():
    assert P.book_of(DETAILS["m1"], "1") == {"bid": 0.92, "ask": 0.94, "mid": 0.93, "bid_size": 500.0, "ask_size": 700.0, "spread": 0.02}
    no = P.position_values(-500, 200.0, 0.40, {"mid": 0.57, "bid": 0.55, "ask": 0.59})
    assert no["side"] == "no" and no["unrealized_at_mid"] == pytest.approx(500 * (0.43 - 0.40)) and no["unrealized_at_bid"] == pytest.approx(500 * (0.41 - 0.40))
    assert P.journal_avg_entry(JOURNAL, "1", "yes") == (pytest.approx(0.948), 1000.0)


def test_build_rows_direction_flags_and_caps():
    rows = {r["exchange_id"]: r for r in P.build_rows(_inputs(), MAP, RAW_MAP, ["senate:RI"], JOURNAL, RiskLimits())}
    assert rows["1"]["direction"] == "D" and rows["2"]["direction"] == "D" and rows["3"]["direction"] == "R"  # YES-D and NO-R: same bet
    assert rows["1"]["flag"] is False and rows["2"]["flag"] is True and rows["1"]["resolution_type"] == "Party Winner"
    # RI: D YES 1000 @ 0.95 + R NO 1000 @ 0.06 -> D loses: -950 - 60 = -1010 -> 0.101 of 10,000
    assert rows["1"]["race_worst_loss_share"] == 0.101 and rows["1"]["cluster"] == "senate:RI" and rows["1"]["oracle_mid_now"] == 0.97


# ---------------------------------------------------------------------- settlement carry


def test_expected_pnl_yes_no_haircut_and_break_even():
    rows = P.build_rows(_inputs(), MAP, RAW_MAP, ["senate:RI"], JOURNAL)
    e = {x["exchange_id"]: x for x in P.expected_pnl_rows(rows)}
    one = e["1"]  # YES 1000 @ 0.95, p = 0.97: 1000 * (0.97 - 0.95) = 20; haircut: 1000 * (0.95 - 0.95) = 0
    assert one["expected_pnl"] == pytest.approx(20.0) and one["expected_pnl_haircut"] == pytest.approx(0.0)
    assert one["break_even_p"] == 0.95 and one["break_even_refers_to"].startswith("YES") and one["break_even_yes_p"] == 0.95
    two = e["2"]  # NO 1000 @ 0.06, p_yes 0.03 -> p_side 0.97: 1000 * 0.91 = 910; haircut 1000 * 0.89 = 890
    assert two["p_side"] == 0.97 and two["expected_pnl"] == pytest.approx(910.0) and two["expected_pnl_haircut"] == pytest.approx(890.0)
    assert two["break_even_refers_to"].startswith("NO") and two["break_even_yes_p"] == 0.94
    three = e["3"]  # NO 500 @ 0.40, p_yes 0.50 -> p_side 0.50: 50; haircut 40
    assert three["expected_pnl"] == pytest.approx(50.0) and three["expected_pnl_haircut"] == pytest.approx(40.0)
    unpriced = P.expected_pnl_rows(P.build_rows(_inputs({}), MAP, RAW_MAP, ["senate:RI"], JOURNAL))
    assert all(x["expected_pnl"] is None for x in unpriced)


def test_race_sums_same_bet_legs_total_and_stress():
    rows = P.build_rows(_inputs(), MAP, RAW_MAP, ["senate:RI"], JOURNAL)
    positions = P.expected_pnl_rows(rows)
    races = {r["race"]: r for r in P.race_expected(rows, positions, MAP)}
    ri = races["senate:RI"]  # D YES and R NO: one bet on D, 20 + 910 = 930
    assert ri["level"] == "race" and ri["direction"] == "long D" and ri["cost"] == 1010.0 and ri["expected_pnl"] == pytest.approx(930.0)
    assert ri["worst_pnl"] == -1010.0 and json.loads(ri["scenario_pnl"])["D"] == pytest.approx(990.0)
    az = races["senate:AZ"]  # NO D 500 @ 0.40 = long R: worst case D wins -> -200
    assert az["direction"] == "long R" and az["worst_pnl"] == -200.0
    ordered = P.race_expected(rows, positions, MAP)
    totals = P.expected_totals(positions, ordered, worst_n=1)
    assert totals["expected_pnl"] == pytest.approx(980.0) and totals["expected_pnl_haircut"] == pytest.approx(930.0)
    assert totals["stress_races"] == ["senate:RI"] and totals["stress_loss"] == -1010.0  # largest WORST-CASE loss, not cost
    assert P.expected_totals(positions, ordered, worst_n=5)["stress_loss"] == -1210.0


def test_swaps_against_candidates_without_double_counting(tmp_path):
    rows = P.build_rows(_inputs(), MAP, RAW_MAP, ["senate:RI"], JOURNAL)
    holds = {h["id"]: h for h in P.hold_positions(rows)}
    # YES 1 sells at the YES bid 0.92, capped at the 500-share top level: 460 freed; yield (0.97 - 0.02 - 0.92) / 0.92
    assert holds["1 yes"]["capital"] == pytest.approx(460.0) and holds["1 yes"]["yield"] == pytest.approx(0.03 / 0.92)
    # NO 2 sells at 1 - YES ask 0.07 = 0.93, capped at the 700-share ask level: 651 freed; yield (0.97 - 0.02 - 0.93) / 0.93
    assert holds["2 no"]["capital"] == pytest.approx(651.0) and holds["2 no"]["yield"] == pytest.approx(0.02 / 0.93)
    path = tmp_path / "candidates.csv"
    path.write_text("exchange_id,race,side,allocated_cost,ev_0.02\n"
                    "7,senate:GA,yes,500,100\n"   # yield 0.20, capacity 500
                    "1,senate:RI,no,1000,500\n"   # same exchange as hold "1 yes": never paired with it
                    "8,senate:NV,yes,,\n", encoding="utf-8")  # not allocated: skipped
    cands = P.load_candidates(path)
    assert [c["id"] for c in cands] == ["7 yes", "1 no"] and cands[0]["yield"] == pytest.approx(0.2)
    swaps, total = P.swap_rows(list(holds.values()), cands)
    pairs = {(w["hold"], w["candidate"]): w for w in swaps}
    assert ("1 yes", "1 no") not in pairs
    # "1 no" (yield 0.5) takes NO 2's 651 first ... then GA's 500 goes to the remaining capital of "1 yes" (460) and nothing else.
    assert pairs[("2 no", "1 no")]["amount"] == pytest.approx(651.0)
    assert pairs[("1 yes", "7 yes")]["amount"] == pytest.approx(460.0)
    assert sum(w["amount"] for w in swaps if w["candidate"] == "7 yes") <= 500 + 1e-9
    assert total == pytest.approx(sum(w["gain"] for w in swaps), abs=0.02) and swaps[0]["rank"] == 1
    assert P.load_candidates(None) == [] and P.load_candidates(tmp_path / "absent.csv") == []


# ---------------------------------------------------------------------- inputs: live and saved


class FakeReader:
    def __init__(self, tmp_path):
        self.out_dir = tmp_path
        self.calls = []
        self.saved = []
        self.raw_log = type("Log", (), {"append": lambda _self, rec: self.saved.append(rec)})()

    async def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if path == "/tournaments/t":
            return {"id": "tid", "initialBalance": 10_000}
        if path == "/portfolio/positions":
            return {"positions": POSITIONS}
        if path == "/relationships":
            return {"data": [], "pagination": {"hasMore": False}}
        if path.startswith("/tournaments/t/markets/"):
            return DETAILS[path.rsplit("/", 1)[1]]
        if path.startswith("/markets/") and path.endswith("/nodes"):
            return NODES.get(path.split("/")[2], {})
        raise AssertionError(f"unexpected GET {path}")


class FakeOracle:
    def __init__(self, quotes):
        self.quotes = quotes

    async def get_quotes(self, refs):
        return {r: q for r, q in self.quotes.items() if r in refs}


def test_gather_reads_and_saves_oracle_mids():
    reader = FakeReader(Path("."))
    inp = asyncio.run(P.gather(reader, "t", MAP, RAW_MAP, ["senate:RI"], FakeOracle({"tok-1": _q("tok-1", 0.95, 0.97)}), FakeOracle({})))
    assert len(reader.calls) == 3 + 2 * 3 and inp.equity == 10_000.0 and inp.oracle_mids["1"] == 0.96
    assert reader.saved[0]["path"] == P.ORACLE_RECORD_PATH and reader.saved[0]["body"]["mids"]["1"] == 0.96


def test_inputs_from_raw_round_trip():
    records = [{"ts": "a", "path": "/tournaments/t", "body": {"id": "tid", "initialBalance": 10_000}},
               {"ts": "b", "path": "/portfolio/positions", "body": {"positions": POSITIONS}},
               {"ts": "c", "path": "/relationships", "params": {"cursor": None}, "body": {"data": [{"id": "r1"}]}},
               *[{"ts": "d", "path": f"/tournaments/t/markets/{k}", "body": v} for k, v in DETAILS.items()],
               *[{"ts": "e", "path": f"/markets/{k}/nodes", "body": v} for k, v in NODES.items()],
               {"ts": "2026-10-12T12:00:00Z", "path": P.ORACLE_RECORD_PATH, "body": {"mids": ORACLE_MIDS}},
               {"ts": "f", "path": "/markets/m9/nodes", "error": {"status": 500}}]
    inp = P.inputs_from_raw(records)
    assert inp.equity == 10_000.0 and len(inp.positions) == 4 and inp.relationships == [{"id": "r1"}]
    assert set(inp.details) == {"m1", "m2", "m3", "m4"} and inp.oracle_mids == ORACLE_MIDS and inp.oracle_from == "saved 2026-10-12T12:00:00Z"


@pytest.fixture
def quiet_main(monkeypatch, tmp_path):
    import config

    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(P, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(P.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    raw_map_path = tmp_path / "market_map.json"
    raw_map_path.write_text(json.dumps(RAW_MAP), encoding="utf-8")
    monkeypatch.setattr(P, "MARKET_MAP_PATH", raw_map_path)
    monkeypatch.setattr(P, "ORDER_JOURNAL_PATH", tmp_path / "journal.jsonl")
    monkeypatch.setattr(P, "OUT_ROOT", tmp_path / "edge_scan")
    monkeypatch.setattr(P, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.setattr(P, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no SIG reader offline")))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    (tmp_path / "out").mkdir()
    return tmp_path


def test_without_a_flag_only_the_estimate_is_logged(quiet_main, caplog):
    with caplog.at_level("INFO"):
        assert P.main([]) == 0
    assert any(r.getMessage() == "Planned SIG reads" for r in caplog.records)
    assert not (quiet_main / "out" / "summary.md").exists()


def test_offline_without_saved_read_does_nothing(quiet_main, caplog):
    with caplog.at_level("INFO"):
        assert P.main(["--offline"]) == 0
    assert any(r.getMessage() == "No saved position_review read to replay: run once with --yes" for r in caplog.records)
    assert not (quiet_main / "out" / "summary.md").exists()


def test_offline_replays_saved_reads_and_writes_every_output(quiet_main):
    run = quiet_main / "edge_scan" / "position_review" / "2026-10-12"
    run.mkdir(parents=True)
    records = [{"ts": "a", "path": "/tournaments/t", "body": {"initialBalance": 10_000}}, {"ts": "b", "path": "/portfolio/positions", "body": {"positions": POSITIONS}},
               *[{"ts": "d", "path": f"/tournaments/t/markets/{k}", "body": v} for k, v in DETAILS.items()],
               *[{"ts": "e", "path": f"/markets/{k}/nodes", "body": v} for k, v in NODES.items()]]
    (run / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (run / "position_review.csv").write_text("exchange_id,oracle_mid_now\n1,0.97\n2,0.03\n3,0.5\n", encoding="utf-8")  # an older run's output
    scan = quiet_main / "edge_scan" / "settlement_carry_scan" / "2026-10-13"
    scan.mkdir(parents=True)
    (scan / "candidates.csv").write_text("exchange_id,race,side,allocated_cost,ev_0.02\n7,senate:GA,yes,500,100\n", encoding="utf-8")
    assert P.main(["--offline", "--races", "senate:RI"]) == 0
    out = quiet_main / "out"
    for name in ("position_review.csv", "expected_pnl.csv", "swaps.csv", "summary.md"):
        assert (out / name).exists(), name
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "HYPOTHETICAL, oracle-based" in md and "oracle error is not modelled" in md and "| senate:RI | long D |" in md
    assert "## Swaps" in md and "total expected gain" in md and "settlement_carry_scan" in md
    levels = (out / "expected_pnl.csv").read_text(encoding="utf-8")
    assert "position," in levels and "race," in levels and "total," in levels


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
