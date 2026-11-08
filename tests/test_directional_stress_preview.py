"""scripts/directional_stress_preview.py on mocked data - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from edge_scan import SigReader
from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # the script imports rebalance_preview
SCRIPT = ROOT / "scripts" / "directional_stress_preview.py"
_spec = importlib.util.spec_from_file_location("directional_stress_preview", SCRIPT)
S = importlib.util.module_from_spec(_spec)
sys.modules["directional_stress_preview"] = S  # dataclasses look their module up while it executes
_spec.loader.exec_module(S)


def _m(eid, race, party, token=True):
    return MarketMapping(eid, eid, f"{party} {race.key}", race, party, f"tok-{eid}" if token else None, None)


RI, NH01, NV, ME, GA, AK = (Race("senate", "RI"), Race("house", "NH", "01"), Race("senate", "NV"),
                            Race("senate", "ME"), Race("senate", "GA"), Race("house", "AK", "00"))
MAP = MarketMap({m.sig_exchange_id: m for m in [
    _m("1", RI, "D"), _m("2", RI, "R"),
    _m("3", NH01, "R"), _m("4", NH01, "D", token=False),
    _m("5", NV, "D"), _m("6", NV, "R", token=False),
    _m("8", ME, "R"),
    _m("10", GA, "D"), _m("11", GA, "R"),
    _m("12", AK, "D", token=False),
]})
POSITIONS = [
    {"exchangeId": "1", "quantity": 21_000, "marketValue": 19_950.0, "costBasis": 19_000.0},  # RI D YES favourite: D way over 15%
    {"exchangeId": "3", "quantity": 1_000, "marketValue": 400.0, "costBasis": 380.0},  # NH-01 R YES (pro-R)
    {"exchangeId": "10", "quantity": 4_000, "marketValue": 1_200.0, "costBasis": 2_400.0},  # GA D YES, bought high: GA race at its cap
    {"exchangeId": "12", "quantity": 100, "marketValue": 30.0, "costBasis": 25.0},  # no oracle -> priced at its mark
]
PRICES = {
    "1": {"bestBid": 0.94, "bestAsk": 0.96}, "2": {"bestBid": 0.04, "bestAsk": 0.06},
    "3": {"bestBid": 0.39, "bestAsk": 0.41},
    "5": {"bestBid": 0.48, "bestAsk": 0.50},  # D YES: fair 0.62 -> a clear edge, blocked today by the D cap
    "8": {"bestBid": 0.58, "bestAsk": 0.60},  # R YES: fair 0.70 -> an R target today already
    "10": {"bestBid": 0.29, "bestAsk": 0.31}, "11": {"bestBid": 0.40, "bestAsk": 0.42},  # GA R NO: edge, but the GA race is full
    "12": {"bestBid": 0.28, "bestAsk": 0.32},
}


def _q(eid, mid):
    return OracleQuote("polymarket", f"tok-{eid}", mid - 0.005, mid + 0.005, 500.0, 500.0, None, time.time())


PM = {f"tok-{eid}": _q(eid, mid) for eid, mid in {"1": 0.95, "2": 0.05, "3": 0.40, "5": 0.62, "8": 0.70, "10": 0.30, "11": 0.30}.items()}


def _inputs(**overrides):
    from rebalance_preview import PreviewInputs

    kwargs = dict(market_map=MAP, positions=POSITIONS, open_orders=[], bankroll=100_000.0, prices=PRICES, pm_quotes=PM, ks_quotes={})
    return PreviewInputs(**{**kwargs, **overrides})


def _run():
    rows, summary = S.build_stress_preview(_inputs())
    return {r.exchange_id: r for r in rows}, summary


# ---------------------------------------------------------------------- maths


def test_stress_moves_the_waves_party_up_and_the_other_down():
    assert S.stressed_yes(0.5, "D", "D", 0.5) == pytest.approx(S.logistic(0.5))
    assert S.stressed_yes(0.5, "R", "D", 0.5) == pytest.approx(S.logistic(-0.5))
    assert S.stressed_yes(0.5, "I", "D", 0.5) == 0.5
    # A D YES at 0.95 loses 0.95 - logistic(logit(0.95) - 0.5) per share in an R-wave; a NO gains (0 loss).
    assert S.side_loss_per_share(0.95, "D", "yes", "R", 0.5) == pytest.approx(0.95 - S.logistic(S.logit(0.95) - 0.5))
    assert S.side_loss_per_share(0.95, "D", "no", "R", 0.5) == 0.0
    assert S.adverse_wave("D") == "R" and S.adverse_wave("N") is None


# ---------------------------------------------------------------------- 1-3


def test_directional_exposure_as_the_taker_counts_it():
    _, summary = _run()
    today = summary["directional_today"]
    assert today["D"]["exposure"] == pytest.approx(19_950 + 1_200 + 30)  # held market value, pro-D
    assert today["D"]["room"] == pytest.approx(15_000 - 21_180)  # below zero
    assert today["R"]["exposure"] == pytest.approx(400.0)


def test_stress_totals_contributions_and_focus_races():
    by_id, summary = _run()
    ri_loss = 21_000 * S.side_loss_per_share(0.95, "D", "yes", "R", 0.5)
    assert by_id["1"].losses["R@0.5"] == pytest.approx(ri_loss, abs=0.01) and by_id["1"].losses["D@0.5"] == 0.0
    assert by_id["12"].price_source == "mark" and by_id["12"].p_yes == pytest.approx(0.30)
    r_wave = summary["stress"]["R@0.5"]
    assert r_wave["loss"] == pytest.approx(sum(r.losses["R@0.5"] for r in by_id.values()), abs=0.05)
    assert r_wave["pct_bankroll"] == pytest.approx(100 * r_wave["loss"] / 100_000, abs=0.01)
    assert summary["stress"]["R@0.75"]["loss"] > r_wave["loss"] > summary["stress"]["R@0.25"]["loss"]
    assert summary["top_contributions_s0.5"]["R"][0]["race"] == "senate:RI"
    assert summary["focus_races"]["senate:RI"]["R@0.5"] == pytest.approx(ri_loss, abs=0.01)
    assert summary["focus_races"]["house:NH-01"]["D@0.5"] > 0 and summary["focus_races"]["house:NH-01"]["R@0.5"] == 0.0


def test_alternative_capacity():
    _, summary = _run()
    d = summary["alternative_capacity"]["D"]
    assert d["free_nominal_today"] < 0
    assert d["free_stress"] == pytest.approx(4_000 - d["stress_loss_s0.5"], abs=0.01)
    assert d["free_nominal_alternative"] == pytest.approx(d["free_stress"] / d["loss_per_nominal_dollar"], rel=1e-3)
    assert d["nominal_gained"] > 0  # favourites carry little stress per nominal dollar


# ---------------------------------------------------------------------- 4-5


def test_alternative_limit_unlocks_the_clear_edge_and_the_race_cap_still_binds():
    _, summary = _run()
    sim = summary["simulation"]
    unlocked = {o["exchange_id"]: o for o in sim["unlocked_orders"]}
    assert set(unlocked) == {"5"}  # NV D YES: no room today (D cap), room under the stress limit
    order = unlocked["5"]
    assert order["edge"] == pytest.approx(0.62 - 0.01 - 0.50)
    assert sim["unlocked_stake"] == pytest.approx(order["quantity"] * order["price"], abs=0.01)
    assert sim["unlocked_expected_profit"] == pytest.approx(order["quantity"] * order["edge"], abs=0.05)
    assert sim["unlocked_stress_s0.5"] > 0
    assert sim["still_blocked_by"] == {"race (2%)": 1}  # GA R NO (pro-D): the GA race is at its cap
    assert summary["summary"]["unlocked_with_clear_edge"] == 1
    assert summary["summary"]["expected_profit_per_stress_dollar"] == pytest.approx(sim["unlocked_expected_profit"] / sim["unlocked_stress_s0.5"], abs=1e-3)


def test_outputs(tmp_path):
    rows, summary = S.build_stress_preview(_inputs())
    S.write_outputs(rows, summary, tmp_path / "positions.csv", tmp_path / "summary.json")
    header = (tmp_path / "positions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "R@0.5" in header and "losses" not in header
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["simulation"]["unlocked"] == 1


def test_script_never_references_order_methods():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "OrderRequest", '"POST"', '"DELETE"'):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("value", ["0", "31"])
def test_read_rate_is_validated(value):
    with pytest.raises(SystemExit) as exit_info:
        S.main(["--max-reads-per-min", value])
    assert exit_info.value.code == 2
