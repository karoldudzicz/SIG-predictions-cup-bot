"""scripts/gross_cap_sim.py: hand-computed cases, one binding cap per test - no network.

Run from the project root:  python -m pytest tests/test_gross_cap_sim.py -v

Equity 10,000 and the default RiskLimits: contract 0.02 (200), race worst loss 0.02 (200),
cluster 0.03 (300). senate:RI = D (1) / R (2), one cluster.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gross_cap_sim.py"
_spec = importlib.util.spec_from_file_location("gross_cap_sim", SCRIPT)
G = importlib.util.module_from_spec(_spec)
sys.modules["gross_cap_sim"] = G
_spec.loader.exec_module(G)

MAP = MarketMap({m.sig_exchange_id: m for m in [
    MarketMapping("1", "m1", "D RI", Race("senate", "RI"), "D", None, None),
    MarketMapping("2", "m2", "R RI", Race("senate", "RI"), "R", None, None),
]})
CLUSTERS = {"1": "senate:RI", "2": "senate:RI"}
EQUITY = 10_000.0


def _cand(eid, side, party, price=0.10, depth=5000.0, p_side=0.5, rank=1):
    return {"rank": rank, "exchange_id": eid, "race": "senate:RI", "party": party, "side": side, "price": price, "oracle_p_side": p_side,
            "depth_shares": depth, "depth_cost": price * depth, "book_read": True}


def _sim(cands, positions, fraction=0.40):
    result = G.simulate(fraction, cands, G.levels_from_csv(cands), positions, MAP, CLUSTERS, EQUITY)
    return result, {(r["exchange_id"], r["side"]): r for r in result["_rows"]}


def test_depth_binds():
    result, rows = _sim([_cand("1", "yes", "D", depth=500)], [])
    row = rows[("1", "yes")]
    assert row["binding"] == "depth" and row["allocated_shares"] == 500 and row["allocated_cost"] == pytest.approx(50.0)
    assert row["added_ev"] == pytest.approx(500 * 0.48 - 50)  # p_side 0.5 with the 0.02 haircut


def test_contract_binds():
    # Held R NO 1000 @ 0.05 (cost 50). Buying D NO @ 0.10: contract room 200 -> 2000 shares;
    # cluster room 250 -> 2500; race: "D wins" 950 - 0.1 q >= -200 -> q <= 11500.
    held = [{"exchangeId": "2", "quantity": -1000, "costBasis": 50.0}]
    _, rows = _sim([_cand("1", "no", "D")], held)
    row = rows[("1", "no")]
    assert row["binding"] == "contract" and row["allocated_shares"] == pytest.approx(2000) and row["allocated_cost"] == pytest.approx(200.0)


def test_race_binds():
    # Held R NO 1000 @ 0.10 (cost 100). Buying D YES @ 0.10: "R wins" -100 - 0.1 q >= -200 -> q <= 1000,
    # while contract and cluster each allow 2000.
    held = [{"exchangeId": "2", "quantity": -1000, "costBasis": 100.0}]
    result, rows = _sim([_cand("1", "yes", "D")], held)
    row = rows[("1", "yes")]
    assert row["binding"] == "race" and row["allocated_shares"] == 1000 and row["allocated_cost"] == pytest.approx(100.0)
    assert result["added_worst_loss"] == pytest.approx(100.0)  # RI worst loss 100 -> 200


def test_cluster_binds():
    # Held R NO 1000 @ 0.25 (cost 250). Buying D NO @ 0.10: cluster room 50 -> 500 shares; contract 2000;
    # race allows [56, 9500] (the "R wins" scenario must be pulled back above -200).
    held = [{"exchangeId": "2", "quantity": -1000, "costBasis": 250.0}]
    _, rows = _sim([_cand("1", "no", "D")], held)
    row = rows[("1", "no")]
    assert row["binding"] == "cluster" and row["allocated_shares"] == pytest.approx(500) and row["allocated_cost"] == pytest.approx(50.0)


def test_gross_binds():
    # No positions, gross fraction 0.0105 -> room 105 -> 1050 shares @ 0.10; contract and race allow 2000.
    result, rows = _sim([_cand("1", "yes", "D")], [], fraction=0.0105)
    row = rows[("1", "yes")]
    assert row["binding"] == "gross" and row["allocated_shares"] == pytest.approx(1050) and result["gross_room"] == pytest.approx(105.0)


def test_sweep_adds_ev_until_another_cap_binds_and_reports_risk():
    # 3,900 of cost already held elsewhere (unmapped, so outside every race): at 0.40 the gross room is 100.
    held = [{"exchangeId": "99", "quantity": 39_000, "costBasis": 3900.0}]
    base = [_cand("1", "yes", "D")]
    at_40, _ = _sim(base, held, 0.40)
    at_45, rows_45 = _sim(base, held, 0.45)
    ceiling, _ = _sim(base, held, G.NO_GROSS_CAP)
    assert (at_40["gross_room"], at_40["allocated_cost"], at_40["added_ev"]) == (100.0, 100.0, 380.0)  # 1000 * 0.48 - 100
    assert (at_45["allocated_cost"], at_45["added_ev"]) == (200.0, 760.0) and rows_45[("1", "yes")]["binding"] == "contract"  # race also allows exactly 2000: not lower
    assert ceiling["allocated_cost"] == 200.0  # the other caps are the ceiling
    assert (at_40["stress_before"], at_40["stress_after"], at_40["added_worst_loss"]) == (0.0, 100.0, 100.0)
    assert at_40["new_contracts"] == 1 and at_40["new_races"] == 1 and at_40["added_ev_per_susq"] == 3.8
    assert at_45["_top"][0]["allocated_cost"] == 200.0


def test_no_book_candidates_get_nothing():
    cand = {**_cand("1", "yes", "D"), "book_read": False}
    result, rows = _sim([cand], [])
    assert rows[("1", "yes")]["binding"] == "no book" and result["allocated_cost"] == 0.0


def test_load_candidates_and_levels_from_the_scan_csv(tmp_path):
    path = tmp_path / "candidates.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "exchange_id", "race", "party", "side", "price", "oracle_p_side", "book_read", "depth_shares", "depth_cost", "allocated_shares"])
        w.writeheader()
        w.writerow({"rank": 2, "exchange_id": "1", "race": "senate:RI", "party": "D", "side": "yes", "price": 0.1, "oracle_p_side": 0.5,
                    "book_read": "True", "depth_shares": 300, "depth_cost": 31.5, "allocated_shares": 7})
        w.writerow({"rank": 1, "exchange_id": "2", "race": "senate:RI", "party": "R", "side": "no", "price": 0.9, "oracle_p_side": 0.95,
                    "book_read": "False", "depth_shares": "", "depth_cost": "", "allocated_shares": ""})
    cands = G.load_candidates(path)
    assert [c["rank"] for c in cands] == [1, 2] and "allocated_shares" not in cands[0]  # the scan's allocation is recomputed
    assert G.levels_from_csv(cands) == {("1", "yes"): [(0.105, 300.0)]}


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "oracle_api", "aiohttp"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden


def test_main_writes_both_outputs_next_to_the_scan(monkeypatch, tmp_path):
    import config

    run = tmp_path / "settlement_carry_scan" / "2026-10-17"
    run.mkdir(parents=True)
    with (run / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "exchange_id", "race", "party", "side", "price", "oracle_p_side", "book_read", "depth_shares", "depth_cost"])
        w.writeheader()
        w.writerow({"rank": 1, "exchange_id": "1", "race": "senate:RI", "party": "D", "side": "yes", "price": 0.1, "oracle_p_side": 0.5,
                    "book_read": "True", "depth_shares": 5000, "depth_cost": 500})
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(G, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(G.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.setattr(G, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(G.scan, "saved_review", lambda root=None: ([{"exchangeId": "99", "quantity": 39_000, "costBasis": 3900.0}], [], "saved test"))
    assert G.main(["--equity", "10000"]) == 0
    out = run / "gross_cap"
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert md.startswith("# Gross cap sensitivity") and "HYPOTHETICAL" in md and "no gross cap" in md and "| 0.40 | 100.0 | 100.0 |" in md
    with (out / "gross_cap_sim.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert sum(1 for r in rows if r["level"] == "summary") == 5 and sum(1 for r in rows if r["level"] == "candidate") == 5
    monkeypatch.setattr(G.scan, "saved_review", lambda root=None: (None, [], None))
    assert G.main(["--equity", "10000"]) == 2
