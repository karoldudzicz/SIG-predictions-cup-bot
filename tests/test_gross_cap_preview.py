"""scripts/gross_cap_preview.py on mocked data - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # the script imports rebalance_preview / directional_stress_preview
SCRIPT = ROOT / "scripts" / "gross_cap_preview.py"
_spec = importlib.util.spec_from_file_location("gross_cap_preview", SCRIPT)
G = importlib.util.module_from_spec(_spec)
sys.modules["gross_cap_preview"] = G  # dataclasses look their module up while it executes
_spec.loader.exec_module(G)


def _m(eid, race, party, token=True):
    return MarketMapping(eid, eid, f"{party} {race.key}", race, party, f"tok-{eid}" if token else None, None)


# Ten held favourites, one race each, 3,000 of market value apiece: 30% of a 100k bankroll by
# market value (the sizing portfolio cap is full), 15k per direction (the nominal D and R caps
# are full), but little stress per dollar (prices 0.90).
HELD = [(f"{i}", Race("house", "ZZ", f"{i:02d}"), "D" if i <= 5 else "R") for i in range(1, 11)]
TN, TX28, GA = Race("governor", "TN"), Race("house", "TX", "28"), Race("senate", "GA")
MAP = MarketMap({m.sig_exchange_id: m for m in [
    *[_m(eid, race, party) for eid, race, party in HELD],
    _m("1000", TN, "R"), _m("1001", TN, "D", token=False),  # TN R YES: fair 0.951 vs ask 0.855
    _m("936", TX28, "R"), _m("937", TX28, "D", token=False),  # TX-28 R YES: fair 0.8875 vs ask 0.835
    _m("50", GA, "D"), _m("51", GA, "R"),  # GA: D YES held at its race cap, R NO (pro-D) has an edge
]})
POSITIONS = [{"exchangeId": eid, "quantity": 3_333, "marketValue": 3_000.0, "costBasis": 2_800.0} for eid, _, _ in HELD]
POSITIONS.append({"exchangeId": "50", "quantity": 4_000, "marketValue": 1_200.0, "costBasis": 2_400.0})
PRICES = {eid: {"bestBid": 0.89, "bestAsk": 0.91} for eid, _, _ in HELD}
PRICES.update({
    "1000": {"bestBid": 0.84, "bestAsk": 0.855}, "936": {"bestBid": 0.82, "bestAsk": 0.835},
    "50": {"bestBid": 0.29, "bestAsk": 0.31}, "51": {"bestBid": 0.40, "bestAsk": 0.42},
})


def _q(eid, mid):
    return OracleQuote("polymarket", f"tok-{eid}", mid - 0.005, mid + 0.005, 500.0, 500.0, None, time.time())


PM = {f"tok-{eid}": _q(eid, 0.90) for eid, _, _ in HELD}
PM.update({"tok-1000": _q("1000", 0.951), "tok-936": _q("936", 0.8875), "tok-50": _q("50", 0.30), "tok-51": _q("51", 0.30)})


def _inputs(**overrides):
    from rebalance_preview import PreviewInputs

    kwargs = dict(market_map=MAP, positions=POSITIONS, open_orders=[], bankroll=100_000.0, prices=PRICES, pm_quotes=PM, ks_quotes={})
    return PreviewInputs(**{**kwargs, **overrides})


def _run(fracs=G.DEFAULT_GROSS_FRACS):
    blocked, summary, traces = G.build_gross_preview(_inputs(), fracs)
    return {b.exchange_id: b for b in blocked}, summary, traces


def test_variant_limits_replace_both_30pct_caps_and_lift_the_global_one():
    from risk import RiskLimits

    limits = G.variant_limits(RiskLimits(), 0.5)
    assert (limits.max_total_fraction, limits.max_gross_exposure_fraction, limits.max_total_gross_exposure_fraction) == (0.5, 0.5, 0.5)
    assert G.variant_limits(RiskLimits(), 0.3).max_total_gross_exposure_fraction == 0.40  # never lowered


def test_blocked_targets_with_their_first_limit():
    blocked, summary, _ = _run()
    tn, tx, ga = blocked["1000"], blocked["936"], blocked["51"]
    assert (tn.side, tn.buy_price, tn.fair_side) == ("yes", 0.855, 0.951)
    assert tn.edge == pytest.approx(0.951 - 0.01 - 0.855) and tx.edge == pytest.approx(0.8875 - 0.01 - 0.835)
    # Today: the nominal R cap (15k of held R market value) cuts every R target in sizing.
    assert tn.first_block == tx.first_block == "directional 15% (sizing, market value)"
    assert ga.side == "no" and ga.first_block == "directional 15% (sizing, market value)"
    assert tn.target_shares > 0 and tn.expected_profit == pytest.approx(tn.target_shares * tn.edge, abs=0.01)
    profits = [item["expected_profit"] for item in summary["top_blocked_by_expected_profit"]]
    assert profits == sorted(profits, reverse=True) and len(profits) == 3  # largest expected profit first


def test_variants():
    _, summary, _ = _run()
    by_frac = {v["gross_frac"]: v for v in summary["variants"]}
    # Stress directional limit alone: the 30% portfolio cap (held market value) still cuts everything.
    base = by_frac[0.30]
    assert base["unlocked"] == 0 and base["still_blocked_by"].get("portfolio 30% (sizing, held market value)") == 3
    # 40%: room in sizing; TN and TX-28 go through, GA R NO still stops at the GA race cap.
    loose = by_frac[0.40]
    assert {o["exchange_id"] for o in loose["unlocked_orders"]} == {"1000", "936"}
    assert loose["still_blocked_by"] == {"race 2% (order)": 1}
    assert loose["unlocked_expected_profit"] == pytest.approx(sum(o["quantity"] * o["edge"] for o in loose["unlocked_orders"]), abs=0.05)
    assert loose["gross_after_pct"] > base["gross_after_pct"]
    assert loose["stress_after"]["R@0.75"]["loss"] > loose["stress_after"]["R@0.5"]["loss"] > base["stress_after"]["R@0.5"]["loss"]


def test_explain_trace_names_every_step():
    _, _, traces = _run((0.30,))
    steps = [step["step"] for step in traces["1000"]]
    assert steps[:2] == ["buy filter", "buy filter"] and steps[-1] == "result"
    assert any(step.get("result") == "ok" for step in traces["1000"] if step["step"] == "buy filter")
    assert traces["1000"][-1]["first_block"] == "directional 15% (sizing, market value)"


def test_oracle_error_scenario():
    _, summary, _ = _run((0.40,))
    today = summary["today"]["oracle_error"]
    # D: five held D favourites (cost 2,800 each) + GA D YES (cost 2,400, not a favourite at 0.30).
    assert today["race_worst_loss_by_direction"]["D"] == pytest.approx(5 * 2_800 + 2_400)
    assert today["all_favourites_lose"]["D"] == pytest.approx(5 * 2_800)
    assert today["all_favourites_lose_pct"]["R"] == pytest.approx(14.0)
    variant = summary["variants"][0]["oracle_error"]
    assert variant["all_favourites_lose"]["R"] > today["all_favourites_lose"]["R"]  # TN / TX-28 are R favourites


def test_outputs(tmp_path):
    blocked, summary, traces = G.build_gross_preview(_inputs())
    G.write_outputs(blocked, summary, traces, ("1000", "936"), tmp_path / "blocked.csv", tmp_path / "variants.json")
    rows = list(csv.DictReader((tmp_path / "blocked.csv").open(encoding="utf-8")))
    assert [r["exchange_id"] for r in rows] == [b.exchange_id for b in blocked] and set(rows[0]) == set(G.BLOCKED_FIELDS)
    data = json.loads((tmp_path / "variants.json").read_text(encoding="utf-8"))
    assert set(data["explain"]) == {"1000", "936"} and len(data["variants"]) == 4


def test_script_never_references_order_methods():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "OrderRequest", '"POST"', '"DELETE"'):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("argv", [["--max-reads-per-min", "31"], ["--gross-frac", "2"], ["--explain", "abc"]])
def test_cli_validation(argv):
    with pytest.raises(SystemExit) as exit_info:
        G.main(argv)
    assert exit_info.value.code == 2
