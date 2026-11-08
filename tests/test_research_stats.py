"""research/stats.py on synthetic data - pure functions, no I/O.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from research import stats as S


# ---------------------------------------------------------------------- distributions


def test_percentile_matches_linear_interpolation():
    data = [1.0, 2.0, 3.0, 4.0]
    assert S.percentile(data, 0) == 1.0
    assert S.percentile(data, 100) == 4.0
    assert S.percentile(data, 50) == pytest.approx(2.5)
    assert S.percentile(data, 25) == pytest.approx(1.75)  # numpy.percentile([1,2,3,4], 25)
    assert S.percentile([7.0], 90) == 7.0
    with pytest.raises(ValueError):
        S.percentile([], 50)
    with pytest.raises(ValueError):
        S.percentile(data, 101)


def test_describe_and_empty():
    out = S.describe([3, 1, 2], percentiles=(50,))
    assert out == {"count": 3, "sum": 6.0, "mean": 2.0, "min": 1.0, "max": 3.0, "p50": 2.0}
    empty = S.describe([], percentiles=(50,))
    assert empty["count"] == 0 and empty["p50"] is None and empty["sum"] is None


@pytest.mark.parametrize("price, label", [(0.0, "<0.15"), (0.149, "<0.15"), (0.15, "0.15-0.85"), (0.849, "0.15-0.85"), (0.85, ">0.85"), (1.0, ">0.85")])
def test_price_bucket_edges(price, label):
    assert S.price_bucket(price) == label


@pytest.mark.parametrize("size, label", [(1, "1-9"), (10, "10-99"), (499, "100-499"), (999, "500-999"), (2999, "1000-2999"), (3000, "3000+")])
def test_size_bucket_edges(size, label):
    assert S.size_bucket(size) == label


def test_share_table_keeps_ordered_zeros():
    table = S.share_table(["YES", "YES", "NO", "?"], order=["YES", "NO", "N/A"])
    assert list(table) == ["YES", "NO", "N/A", "?"]
    assert table["YES"] == {"count": 2, "share": 0.5}
    assert table["N/A"] == {"count": 0, "share": 0.0}
    assert S.share_table([]) == {}


# ---------------------------------------------------------------------- splits


def test_split_halves_is_by_time_not_input_order():
    rows = [{"t": 3}, {"t": 1}, {"t": 5}, {"t": 2}, {"t": 4}]
    first, second = S.split_halves(rows, key=lambda r: r["t"])
    assert [r["t"] for r in first] == [1, 2, 3] and [r["t"] for r in second] == [4, 5]
    assert S.split_halves([], key=lambda r: r) == ([], [])


def test_by_utc_day_groups_and_orders():
    def at(day, hour):
        return datetime(2026, 10, day, hour, tzinfo=timezone.utc)

    groups = S.by_utc_day([at(3, 1), at(2, 23), at(3, 5)], ts=lambda t: t)
    assert list(groups) == ["2026-10-02", "2026-10-03"] and len(groups["2026-10-03"]) == 2


# ---------------------------------------------------------------------- leaderboard


def test_pnl_decomposition_adds_up_and_propagates_missing():
    parts = S.pnl_decomposition(1000.0, 300.0, 50.0)
    assert parts == {"early": 700.0, "mid": 250.0, "today": 50.0}
    assert sum(parts.values()) == 1000.0
    assert S.pnl_decomposition(1000.0, None, 50.0) == {"early": None, "mid": None, "today": 50.0}
    assert S.pnl_decomposition(None, 300.0, None) == {"early": None, "mid": None, "today": None}


def test_zero_sum_check():
    balanced = S.zero_sum_check([100, -60, -40, 0])
    assert balanced["sum_pnl"] == 0 and balanced["sum_over_gross"] == 0
    assert (balanced["winners"], balanced["losers"], balanced["flat"]) == (1, 2, 1)
    leaky = S.zero_sum_check([100, -300])  # a counterparty outside the ranking gained 200
    assert leaky["sum_pnl"] == -200 and leaky["sum_over_gross"] == pytest.approx(-0.5)
    assert S.zero_sum_check([])["sum_over_gross"] is None


# ---------------------------------------------------------------------- tape


def test_tick_rule_upticks_downticks_and_zero_ticks():
    assert S.tick_rule([0.50, 0.51, 0.51, 0.49, 0.49, 0.50]) == [None, "buy", "buy", "sell", "sell", "buy"]
    assert S.tick_rule([0.5, 0.5]) == [None, None]  # no price change yet -> no call
    assert S.tick_rule([]) == []


# ---------------------------------------------------------------------- books


def test_large_levels_both_sides_with_rank_and_distance():
    book = {
        "bids": [{"price": 0.40, "quantity": 100}, {"price": 0.38, "quantity": 5000}, {"price": 0.30, "quantity": 2999}],
        "asks": [{"price": 0.42, "quantity": 3000}, {"price": 0.50, "quantity": -4000}],  # signed size still counts
    }
    levels = S.large_levels(book, 3000)
    assert levels == [
        {"side": "bid", "price": 0.38, "quantity": 5000.0, "rank": 1, "distance": 0.02},
        {"side": "ask", "price": 0.42, "quantity": 3000.0, "rank": 0, "distance": 0.0},
        {"side": "ask", "price": 0.50, "quantity": 4000.0, "rank": 1, "distance": 0.08},
    ]
    assert S.large_levels({"bids": [], "asks": None}, 1) == []


# ---------------------------------------------------------------------- valuation


def test_position_value_yes_and_no_at_bid_and_mid():
    assert S.position_value(100, 0.40, 0.44, "bid") == pytest.approx(40.0)
    assert S.position_value(100, 0.40, 0.44, "mid") == pytest.approx(42.0)
    assert S.position_value(-100, 0.40, 0.44, "bid") == pytest.approx(56.0)  # NO sells at 1 - YES ask
    assert S.position_value(-100, 0.40, 0.44, "mid") == pytest.approx(58.0)
    assert S.position_value(100, None, 0.44, "bid") is None
    assert S.position_value(-100, 0.40, None, "bid") is None
    assert S.position_value(100, 0.40, None, "mid") is None
    assert S.position_value(0, None, None, "mid") == 0.0
    with pytest.raises(ValueError):
        S.position_value(1, 0.4, 0.5, "last")


def test_closest_rule_ignores_missing_candidates():
    best, residuals = S.closest_rule(1000.0, {"bid": 990.0, "mid": 1001.0, "broken": None})
    assert best == "mid"
    assert residuals == {"bid": -10.0, "mid": 1.0, "broken": None}
    assert S.closest_rule(1.0, {"x": None}) == (None, {"x": None})


def test_mark_denomination_from_market_value():
    assert S.mark_denomination(100, 0.40, 40.0) == "ambiguous"  # a YES position cannot tell
    assert S.mark_denomination(-100, 0.40, 60.0) == "yes"  # NO valued at 1 - YES mark
    assert S.mark_denomination(-100, 0.60, 60.0) == "side"  # NO valued at its own price
    assert S.mark_denomination(-100, 0.50, 50.0) == "ambiguous"
    assert S.mark_denomination(-100, 0.40, 10.0) is None
    assert S.mark_denomination(100, None, 40.0) is None and S.mark_denomination(0, 0.4, 0.0) is None


def test_classify_mark_matches_and_spread_position():
    out = S.classify_mark(0.42, bid=0.40, ask=0.44, last=0.43)
    assert out["matches"] == ["mid"] and out["vs_spread"] == "inside spread"
    assert out["diff_last"] == pytest.approx(-0.01) and out["diff_bid"] == pytest.approx(0.02)
    assert S.classify_mark(0.43, 0.40, 0.44, 0.43)["matches"] == ["last"]
    assert S.classify_mark(0.42, 0.40, 0.44, 0.42)["matches"] == ["last", "mid"]  # coincide -> both
    assert S.classify_mark(0.30, 0.40, 0.44, None)["vs_spread"] == "below bid"
    assert S.classify_mark(0.50, 0.40, 0.44, None)["vs_spread"] == "above ask"
    one_sided = S.classify_mark(0.40, 0.40, None, None)
    assert one_sided["matches"] == ["bid"] and one_sided["vs_spread"] == "no two-sided quote" and one_sided["diff_mid"] is None


# ---------------------------------------------------------------------- episodes


def test_episodes_break_on_gap_and_unflagged():
    times = [0, 60, 120, 180, 300, 360, 420]
    flags = [True, True, False, True, True, True, False]
    runs = S.episodes(times, flags, max_gap_s=90)
    # 0-60 (2 samples); 180 alone (gap 120 to 300 breaks it); 300-360
    assert [(r["start"], r["end"], r["samples"], r["duration_s"]) for r in runs] == [(0, 60, 2, 60), (180, 180, 1, 0), (300, 360, 2, 60)]
    assert S.episodes([], [], 90) == []
    with pytest.raises(ValueError):
        S.episodes([0], [], 90)


def test_median_or_none():
    assert S.median_or_none([None, 3, 1, 2]) == 2
    assert S.median_or_none([None]) is None


def test_premium_deviation_shares_fixed_thresholds():
    excess = [0.02, 0.02, 0.02, 0.03, 0.035, None]
    out = S.premium_deviation_shares(excess, 0.02, (0.005, 0.01), direction=+1)
    assert out == {"0.005": 0.4, "0.01": 0.4}  # 0.03 and 0.035 are >= 0.01 above
    low = S.premium_deviation_shares([-0.01, 0.0, 0.02], 0.02, (0.005,), direction=-1)
    assert low == {"0.005": round(2 / 3, 4)}
    assert S.premium_deviation_shares([], 0.0, (0.005,), +1) == {"0.005": None}
    assert S.premium_deviation_shares([0.1], None, (0.005,), +1) == {"0.005": None}


# ---------------------------------------------------------------------- tape buckets


def test_vwap():
    assert S.vwap([(0.40, 100), (0.50, 300)]) == pytest.approx(0.475)
    assert S.vwap([(0.40, 0), (None, 5)]) is None
    assert S.vwap([]) is None


def test_bucket_index_edges():
    start = datetime(2026, 10, 1, 16, tzinfo=timezone.utc)
    assert S.bucket_index(start, start, 10, 12) == 0
    assert S.bucket_index(datetime(2026, 10, 1, 16, 9, 59, tzinfo=timezone.utc), start, 10, 12) == 0
    assert S.bucket_index(datetime(2026, 10, 1, 16, 10, tzinfo=timezone.utc), start, 10, 12) == 1
    assert S.bucket_index(datetime(2026, 10, 1, 17, 59, 59, tzinfo=timezone.utc), start, 10, 12) == 11
    assert S.bucket_index(datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc), start, 10, 12) is None
    assert S.bucket_index(datetime(2026, 10, 1, 15, 59, tzinfo=timezone.utc), start, 10, 12) is None


def test_share_outside_is_strict():
    assert S.share_outside([0.05, 0.10, 0.50, 0.90, 0.95], 0.10, 0.90) == 0.4
    assert S.share_outside([], 0.1, 0.9) is None


# ---------------------------------------------------------------------- tape patterns


def test_size_concentration_needs_count_and_share():
    sizes = [100] * 25 + [7] * 20 + [1] * 455  # 500 trades: 100 -> 5%, 7 -> 4%, 1 -> 91%
    out = S.size_concentration(sizes)
    assert [r["size"] for r in out] == [1.0, 100.0] and out[1] == {"size": 100.0, "count": 25, "share": 0.05}


def test_round_notional_tolerance_and_expected_share():
    assert S.is_round_notional(1004.0, 1000) and S.is_round_notional(996.0, 1000)
    assert not S.is_round_notional(1006.0, 1000) and not S.is_round_notional(4.0, 1000)
    assert S.is_round_notional(10_040.0, 1000)  # tolerance grows with k: 0.5% of 10,000 = 50
    assert S.expected_round_share([1000.0, 300.0], 1000) == pytest.approx((0.01 + 0.0) / 2)
    assert S.expected_round_share([500_000.0], 500) == 1.0  # every big notional looks round
    assert S.expected_round_share([], 500) is None


def test_gap_spikes_against_neighbours():
    gaps = [30] * 40 + [29] * 3 + [31] * 4 + [0] * 500 + [5] * 25 + [4] * 20 + [6] * 22
    out = S.gap_spikes(gaps)
    assert [r["gap_s"] for r in out] == [30]  # 5 s is common but so are 4 and 6; 0 s = sweeps, ignored


def test_poisson_binomial_tail():
    assert S.poisson_binomial_tail([0.5, 0.5], 1) == pytest.approx(0.75)
    assert S.poisson_binomial_tail([0.5, 0.5], 2) == pytest.approx(0.25)
    assert S.poisson_binomial_tail([0.1], 2) == 0.0
    assert S.poisson_binomial_tail([], 0) == 1.0


def test_recurring_minutes_flags_a_daily_bot_not_sparse_noise():
    times = []
    for day in range(1, 8):  # 7 days, 50 scattered trades a day + one trade at 14:07 every day
        base = datetime(2026, 10, day, tzinfo=timezone.utc)
        times += [base + timedelta(minutes=(day * 37 + i * 29) % 1440) for i in range(50)]
        times.append(base + timedelta(hours=14, minutes=7))
    out = S.recurring_minutes(times)
    assert out["days"] == 7 and [m["minute"] for m in out["minutes"]] == ["14:07"] and out["minutes"][0]["days"] == 7


def test_recurring_minutes_null_follows_trading_hours():
    # 4 days, a trade every 90 s from 10:00 to 18:00 - the same minutes every day, but
    # inside busy hours that is expected: nothing is flagged.
    times = [datetime(2026, 10, d, 10, tzinfo=timezone.utc) + timedelta(seconds=90 * k) for d in range(1, 5) for k in range(320)]
    assert S.recurring_minutes(times)["minutes"] == []


def test_near_running_day_extreme():
    def at(day, minute):
        return datetime(2026, 10, day, 12, minute, tzinfo=timezone.utc)

    times = [at(1, 0), at(1, 1), at(1, 2), at(1, 3), at(2, 0), at(2, 1)]
    prices = [0.50, 0.52, 0.51, 0.45, 0.30, 0.30]
    assert S.near_running_day_extreme(times, prices) == [None, True, False, True, None, True]


def test_drift_after_uses_last_print_and_refuses_beyond_data():
    times = [0.0, 30.0, 70.0, 400.0]
    prices = [0.50, 0.52, 0.55, 0.40]
    assert S.drift_after(times, prices, 0, 60) == pytest.approx(0.02)  # last print at <= 60 s is the 30 s one
    assert S.drift_after(times, prices, 0, 300) == pytest.approx(0.05)
    assert S.drift_after(times, prices, 2, 300) == pytest.approx(0.0)  # 370 s <= 400: no print in between
    assert S.drift_after(times, prices, 3, 60) is None  # beyond the data


def test_t_stat():
    assert S.t_stat([1.0, 2.0, 3.0]) == pytest.approx(2.0 / (1.0 / math.sqrt(3)))
    assert S.t_stat([1.0]) is None and S.t_stat([2.0, 2.0]) is None
    assert S.t_stat([0.1 + 0.2, 0.3, 0.3]) is None  # float noise, not spread


def test_welch_t():
    assert S.welch_t([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) == pytest.approx(1.0 / math.sqrt(2 / 3))
    assert S.welch_t([1.0], [0.0, 1.0]) is None and S.welch_t([1.0, 1.0], [2.0, 2.0]) is None


# ---------------------------------------------------------------------- dry-run replay


def test_dedupe_repeats_anchored_vs_chained():
    rows = [("a", 0), ("a", 300), ("a", 590), ("a", 650), ("a", 1200), ("b", 10), ("a", 3000)]
    anchored = S.dedupe_repeats(rows, key=lambda r: r[0], when=lambda r: r[1], window_s=600)
    assert [(c["row"], c["repeats"]) for c in anchored] == [(("a", 0), 3), (("b", 10), 1), (("a", 650), 2), (("a", 3000), 1)]
    chained = S.dedupe_repeats(rows, key=lambda r: r[0], when=lambda r: r[1], window_s=600, chained=True)
    assert [(c["row"], c["repeats"]) for c in chained] == [(("a", 0), 5), (("b", 10), 1), (("a", 3000), 1)]  # 1200 -> 3000 is a 30-min gap
    assert S.dedupe_repeats([], key=lambda r: r, when=lambda r: 0, window_s=600) == []


def test_pnl_vs_mark():
    assert S.pnl_vs_mark("buy", 100, 0.40, 0.45) == pytest.approx(5.0)
    assert S.pnl_vs_mark("sell", -100, 0.40, 0.45) == pytest.approx(-5.0)
    assert S.pnl_vs_mark("buy", 100, 0.40, None) is None


# ---------------------------------------------------------------------- opening opportunity


def test_deviation_volume():
    assert S.deviation_volume([(0.40, 100), (0.50, 200), (None, 5)], 0.45) == pytest.approx(5.0 + 10.0)
    assert S.deviation_volume([(0.40, 100)], None) is None
    assert S.deviation_volume([], 0.5) == 0.0


def test_nearest_pairs_within_gap():
    a = [(100.0, 0.55), (200.0, 0.60), (500.0, 0.50)]
    b = [(90.0, 0.50), (130.0, 0.48), (205.0, 0.45)]
    pairs = S.nearest_pairs(a, b, 60)
    assert [(p["t_a"], p["t_b"]) for p in pairs] == [(100.0, 90.0), (200.0, 205.0)]  # 500 has no partner within 60 s
    assert pairs[1]["sum"] == pytest.approx(1.05) and pairs[0]["gap_s"] == 10.0
    assert S.nearest_pairs([(10.0, 0.5)], [(0.0, 0.5), (20.0, 0.4)], 60)[0]["t_b"] == 0.0  # tie -> earlier
    assert S.nearest_pairs([], b, 60) == []
    extra = S.nearest_pairs([(100.0, 0.55, 300)], [(90.0, 0.50, 40)], 60)[0]
    assert extra["item_a"][2] == 300 and extra["item_b"][2] == 40  # extra fields ride along


# ---------------------------------------------------------------------- SIG vs oracle wording


def test_text_keywords_from_titles_slugs_and_tickers():
    kw = S.text_keywords("Will the Democratic candidate win the 2026 Rhode Island Senate race?")
    assert kw["parties"] == {"D"} and kw["offices"] == {"senate"} and kw["states"] == {"rhode island"} and kw["names"] == set()
    slug = S.text_keywords("which-party-will-win-the-house-in-2026 Democratic Party CONTROLH-2026 Democratic Party")
    assert slug["parties"] == {"D"} and slug["offices"] == {"house"}
    named = S.text_keywords("rhode-island-senate-election-winner Jack Reed")
    assert named["names"] == {"jack", "reed"} and named["states"] == {"rhode island"}
    assert S.text_keywords("West Virginia")["states"] == {"virginia", "west virginia"}  # both match; compared as sets


def test_wording_mismatch_flags_only_real_differences():
    sig = "Will the Democratic candidate win the 2026 Rhode Island Senate race?"
    assert S.wording_mismatch(sig, "rhode-island-senate-election-winner Democratic Party", "D") == []
    assert S.wording_mismatch(sig, "rhode-island-senate-election-winner Jack Reed", "D") == []  # oracle names someone, SIG does not: not a clash
    party = S.wording_mismatch(sig, "rhode-island-senate-election Republican Party", "D")
    assert len(party) == 1 and party[0].startswith("party")
    state = S.wording_mismatch(sig, "massachusetts-senate-election Democratic Party", "D")
    assert state and state[0].startswith("state")
    cand = S.wording_mismatch("Will Sheldon Whitehouse win?", "Allan Fung wins", None)
    assert cand and cand[0].startswith("candidate")
    crit = S.wording_mismatch(sig + " rules: margin of victory", "rhode-island-senate Democratic", "D")
    assert crit == ["criterion: SIG only ['margin'], oracle only []"]


# ---------------------------------------------------------------------- calibration


def test_parse_claimed_edge_and_bins():
    assert S.parse_claimed_edge("edge 0.069; fitted to book 300/500") == 0.069
    assert S.parse_claimed_edge("shrunk_edge 0.05") is None and S.parse_claimed_edge(None) is None  # word boundary
    assert S.parse_claimed_edge("flip: close opposite side") is None
    assert [S.edge_bin(e) for e in (0.0, 0.0199, 0.02, 0.0399, 0.04, 0.069, 0.07, 0.3)] == ["<0.02", "<0.02", "0.02-0.04", "0.02-0.04", "0.04-0.07", "0.04-0.07", ">=0.07", ">=0.07"]
    assert S.edge_bin(None) is None


def test_clustered_t_counts_clusters_not_fills():
    t, n = S.clustered_t({"a": 0.01, "b": 0.02, "c": 0.03})
    assert n == 3 and t == pytest.approx(0.02 / (0.01 / math.sqrt(3)))
    assert S.clustered_t({"a": 0.01}) == (None, 1)


def test_markout_table_groups_weights_and_clusters():
    fills = [
        {"group": "x", "shares": 100, "pnl": 2.0, "cluster": "r1"},
        {"group": "x", "shares": 300, "pnl": 3.0, "cluster": "r1"},  # same race: one observation
        {"group": "x", "shares": 100, "pnl": -1.0, "cluster": "r2"},
        {"group": "y", "shares": 50, "pnl": None, "cluster": "r3"},  # unvalued: skipped
        {"group": None, "shares": 50, "pnl": 5.0, "cluster": "r3"},  # no group: skipped
    ]
    rows = {r["group"]: r for r in S.markout_table(fills, group=lambda f: f["group"], order=["w", "x"])}
    assert list(rows) == ["w", "x"] and rows["w"]["fills"] == 0 and rows["w"]["mean_markout"] is None
    x = rows["x"]
    assert (x["fills"], x["shares"], x["clusters"]) == (3, 500.0, 2)
    assert x["mean_markout"] == pytest.approx(4.0 / 500) and x["winners_share"] == pytest.approx(2 / 3, abs=1e-4)
    r1, r2 = 5.0 / 400, -1.0 / 100
    assert x["t_clustered"] == pytest.approx(S.t_stat([r1, r2]))


def test_rises():
    assert S.rises([0.001, None, 0.002, 0.002]) is True
    assert S.rises([0.003, 0.001]) is False
    assert S.rises([None, 0.1]) is None


def test_loss_concentration():
    out = S.loss_concentration({"a": -100.0, "b": -50.0, "c": 30.0, "d": -10.0}, worst=2)
    assert out["worst"] == ["a", "b"] and out["worst_pnl"] == -150.0 and out["net_pnl"] == -130.0
    assert out["share_of_net_loss"] == pytest.approx(150 / 130, abs=1e-4) and out["share_of_gross_losses"] == pytest.approx(150 / 160, abs=1e-4)
    assert S.loss_concentration({"a": 5.0})["share_of_net_loss"] is None


# ---------------------------------------------------------------------- party-winner wording, carry, direction


def test_oracle_parties_from_labels_and_tickers():
    assert S.oracle_parties("Jack Reed (D)", "SENATERI-26-D") == {"polymarket": "D", "kalshi": "D"}
    assert S.oracle_parties("Democratic Party", None) == {"polymarket": "D", "kalshi": None}
    assert S.oracle_parties("Jack Reed", "SENATERI-26-REED") == {"polymarket": None, "kalshi": None}
    assert S.oracle_parties(None, "controlh-2026-r") == {"polymarket": None, "kalshi": "R"}


def test_party_winner_mismatch_compares_party_only():
    linked = {"polymarket": True, "kalshi": True}
    assert S.party_winner_mismatch("D", {"polymarket": "D", "kalshi": "D"}, linked) == []
    assert S.party_winner_mismatch("D", {"polymarket": "R", "kalshi": "D"}, linked) == ["party: SIG D vs polymarket R"]
    assert S.party_winner_mismatch("D", {"polymarket": None, "kalshi": None}, {"polymarket": True}) == ["party: polymarket names no party"]
    assert S.party_winner_mismatch("D", {"polymarket": None, "kalshi": "D"}, {"polymarket": False, "kalshi": True}) == []  # unlinked: not checked


def test_carry_pnl():
    assert S.carry_pnl(1000, 0.97, 0.95) == pytest.approx(20.0)
    assert S.carry_pnl(-500, 0.40, 0.45) == pytest.approx(-25.0)
    assert S.carry_pnl(10, None, 0.5) is None and S.carry_pnl(10, 0.5, None) is None


def test_leg_and_race_direction():
    two = ("D", "R")
    assert S.leg_direction("D", "yes", two) == "D" and S.leg_direction("R", "no", two) == "D"  # same bet
    assert S.leg_direction("D", "no", ("D", "R", "I")) == "not D"
    assert S.leg_direction(None, "yes", two) is None
    assert S.race_direction(["D", "D"]) == "long D" and S.race_direction(["D", "R"]) == "mixed" and S.race_direction([None]) == "flat"


# ---------------------------------------------------------------------- opening convergence


def _c(hhmm_day, close):
    day, hhmm = hhmm_day
    return {"time": f"2026-10-0{day}T{hhmm}:00Z", "close": close}


def test_first_candle_from_and_close_as_of():
    candles = [_c((1, "15:00"), 0.45), _c((1, "17:00"), 0.40), _c((1, "16:00"), 0.50), _c((1, "21:00"), 0.20), {"time": "2026-10-01T18:00:00Z", "close": None}]
    open_ = datetime(2026, 10, 1, 16, tzinfo=timezone.utc)
    assert S.first_candle_from(candles, open_)["close"] == 0.50
    # 20:00Z: the 17:00 candle (ended 18:00) is the last finished one; 21:00 has not ended yet
    assert S.close_as_of(candles, datetime(2026, 10, 1, 20, tzinfo=timezone.utc)) == (0.40, datetime(2026, 10, 1, 17, tzinfo=timezone.utc))
    assert S.close_as_of(candles, datetime(2026, 10, 1, 22, tzinfo=timezone.utc))[0] == 0.20
    assert S.close_as_of(candles, datetime(2026, 10, 1, 15, 30, tzinfo=timezone.utc)) == (None, None)
    assert S.first_candle_from([], open_) is None


def test_mis_initialised():
    assert S.mis_initialised(0.50, 0.95) and S.mis_initialised(0.40, 0.10)
    assert not S.mis_initialised(0.50, 0.80) and not S.mis_initialised(0.62, 0.95)
    assert not S.mis_initialised(None, 0.95) and not S.mis_initialised(0.5, None)


def test_move_toward_oracle():
    up = S.move_toward(0.50, 0.80, 0.95)
    assert up["move"] == pytest.approx(0.30) and up["toward"] == pytest.approx(0.30) and up["gap_closed"] == pytest.approx(0.30 / 0.45)
    away = S.move_toward(0.50, 0.55, 0.05)
    assert away["toward"] == pytest.approx(-0.05) and away["gap_closed"] == pytest.approx(-0.05 / 0.45)
    assert S.move_toward(0.5, None, 0.9)["move"] is None and S.move_toward(0.5, 0.6, 0.5)["gap_closed"] is None


def test_candle_start_accepts_every_time_format():
    want = datetime(2026, 10, 1, 20, tzinfo=timezone.utc)
    for raw in ("2026-10-01T20:00:00Z", "2026-10-01T20:00:00.000Z", "2026-10-01T20:00:00", "2026-10-01T22:00:00+02:00",
                want.timestamp(), int(want.timestamp() * 1000), str(int(want.timestamp())), want):
        assert S.candle_start({"time": raw}) == want, raw


def test_hour_coverage_counts_exchanges_per_hour():
    hours = [datetime(2026, 10, 1, h, tzinfo=timezone.utc) for h in (17, 18, 20)]
    candles = {"a": [{"time": "2026-10-01T17:00:00Z"}, {"time": "2026-10-01T20:00:00.000Z"}], "b": [{"time": 1790874000}], "c": []}  # b: 17:00 as epoch
    assert S.hour_coverage(candles, hours) == {"2026-10-01T17:00Z": 2, "2026-10-01T18:00Z": 0, "2026-10-01T20:00Z": 1}


# ---------------------------------------------------------------------- tail prints


def test_tail_level_buckets():
    assert [S.tail_level(p) for p in (0.01, 0.05, 0.10, 0.50, 0.85, 0.94, 0.95, 0.99)] == [
        "<0.05", "0.05-0.15", "0.05-0.15", "0.15-0.85", "0.85-0.95", "0.85-0.95", ">0.95", ">0.95"]


def test_tail_prints_against_neighbour_median():
    times = [0.0, 60.0, 120.0, 180.0, 240.0, 900.0, 2000.0]
    prices = [0.50, 0.51, 0.05, 0.50, 0.49, 0.50, 0.95]
    out = S.tail_prints(times, prices)
    assert out[2]["tail"] is True and out[2]["median"] == pytest.approx(0.5) and out[2]["deviation"] == pytest.approx(-0.45)
    assert out[0]["tail"] is False  # neighbours 0.51, 0.05, 0.50, 0.49 -> median 0.495
    assert out[5] is None and out[6] is None  # nobody within 5 minutes: not judged
    exactly = S.tail_prints([0, 10, 20, 30], [0.50, 0.50, 0.50, 0.75])
    assert exactly[3]["tail"] is True  # |0.75 - 0.50| = 0.25 counts


# ---------------------------------------------------------------------- settlement carry scan


def test_side_buy_prices_and_carry_edge():
    assert S.side_buy_prices(0.10, 0.12) == {"yes": 0.12, "no": 0.90}
    assert S.side_buy_prices(None, 0.12)["no"] is None
    edge, yld = S.carry_edge(0.97, 0.90, 0.02)
    assert edge == pytest.approx(0.05) and yld == pytest.approx(0.05 / 0.90)
    assert S.carry_edge(None, 0.9) == (None, None) and S.carry_edge(0.5, 0.0) == (None, None)


def test_combine_oracles_requires_every_linked_source():
    both = {"polymarket": True, "kalshi": True}
    out = S.combine_oracles({"polymarket": 0.90, "kalshi": 0.94}, both)
    assert out["p"] == pytest.approx(0.92) and out["gap"] == pytest.approx(0.04) and out["disagree"] is True
    assert S.combine_oracles({"polymarket": 0.90, "kalshi": 0.92}, both)["disagree"] is False
    assert S.combine_oracles({"polymarket": 0.90}, both)["reason"] == "no mid from kalshi"
    assert S.combine_oracles({"kalshi": 0.3}, {"kalshi": True})["p"] == 0.3
    assert S.combine_oracles({}, {})["reason"] == "no oracle linked"


def test_side_levels_and_walk():
    book = {"bids": [{"price": 0.10, "quantity": 500}, {"price": 0.09, "quantity": 1000}], "asks": [{"price": 0.12, "quantity": 300}, {"price": 0.13, "quantity": 200}, {"price": 0.20, "quantity": 9}]}
    assert S.side_levels(book, "yes") == [(0.12, 300.0), (0.13, 200.0), (0.20, 9.0)]
    assert S.side_levels(book, "no") == [(0.90, 500.0), (0.91, 1000.0)]
    assert S.walk_levels(S.side_levels(book, "yes"), 0.13) == (500.0, pytest.approx(300 * 0.12 + 200 * 0.13))
    assert S.walk_levels(S.side_levels(book, "no"), 0.91, max_shares=600) == (600.0, pytest.approx(500 * 0.90 + 100 * 0.91))
    assert S.walk_levels([], 1.0) == (0.0, 0.0)
    shares, cost = S.walk_levels(S.side_levels(book, "yes"), 0.13, max_cost=49.0)  # 36 for 300 @ 0.12, then 13 / 0.13 = 100 more
    assert shares == pytest.approx(400.0) and cost == pytest.approx(49.0)
    assert S.walk_levels(S.side_levels(book, "yes"), 0.13, max_cost=0.0) == (0.0, 0.0)


# ---------------------------------------------------------------------- party-only check, hold yield, swaps


def test_party_only_check_never_compares_candidate_names():
    # The false positive this fixes: party in the SIG title, a bare candidate name in the oracle label.
    sig = S.sig_title_party("Will the Democratic candidate win the 2026 Rhode Island Senate race?", "D")
    assert sig == "D"
    assert S.party_only_check(sig, S.oracle_parties("Jack Reed", None)) == ("unknown", [])
    assert S.party_only_check(sig, S.oracle_parties("Jack Reed (D)", "SENATERI-26-D")) == ("match", [])
    assert S.party_only_check(sig, S.oracle_parties("Allan Fung (R)", None)) == ("mismatch", ["party: SIG D vs polymarket R"])
    assert S.sig_title_party("Who wins: Democrats or Republicans?", "R") == "R"  # two parties named: fall back to the map
    assert S.party_only_check(None, {"polymarket": "D"}) == ("unknown", [])


def test_hold_yield():
    assert S.hold_yield(0.97, 0.90) == pytest.approx(0.07 / 0.90)
    assert S.hold_yield(0.97, 0.90, 0.02) == pytest.approx(0.05 / 0.90)
    assert S.hold_yield(0.5, 0.0) is None and S.hold_yield(None, 0.5) is None


def test_greedy_swaps_use_capital_and_capacity_once():
    holds = [{"id": "h1", "capital": 100.0, "yield": 0.01}, {"id": "h2", "capital": 50.0, "yield": 0.05}]
    cands = [{"id": "c1", "capacity": 120.0, "yield": 0.20}, {"id": "c2", "capacity": 1000.0, "yield": 0.03}, {"id": "h1", "capacity": 99.0, "yield": 0.9}]
    swaps = S.greedy_swaps(holds, cands)
    # Best gains first: h2 -> "h1" (0.85; never h1 -> itself), h1 -> c1 (0.19); h2 is then empty,
    # h1 -> c2 (0.02) has no capital left.
    assert [(s["hold"], s["candidate"], s["amount"]) for s in swaps] == [("h2", "h1", 50.0), ("h1", "c1", 100.0)]
    assert sum(s["gain"] for s in swaps) == pytest.approx(50 * 0.85 + 100 * 0.19)
    two = S.greedy_swaps([{"id": "a", "capital": 100.0, "yield": 0.0}, {"id": "b", "capital": 100.0, "yield": 0.01}], [{"id": "c", "capacity": 150.0, "yield": 0.1}])
    assert [(s["hold"], s["amount"]) for s in two] == [("a", 100.0), ("b", 50.0)]  # capacity is not double-counted
    assert S.greedy_swaps(holds, []) == []


# ---------------------------------------------------------------------- basket arbitrage


def test_basket_no_metrics_hand_computed():
    # D bid 0.55, R bid 0.48: sum 1.03. Buy NO at 0.45 + 0.52 = 0.97 locked; one leg pays 1 -> +0.03.
    m = S.basket_no_metrics([0.55, 0.48])
    assert m["legs"] == 2 and m["excess"] == pytest.approx(0.03) and m["capital"] == pytest.approx(0.97)
    assert m["excess_after_slippage"] == pytest.approx(0.02) and m["excess_ticks_after_slippage"] == pytest.approx(4.0)
    assert m["profit_per_capital"] == pytest.approx(0.02 / 0.97) and m["profit_per_capital_haircut"] == pytest.approx(0.01 / 0.97)
    assert m["break_even_fee_per_leg"] == pytest.approx(0.015) and m["break_even_fee_of_capital"] == pytest.approx(0.03 / 0.97)
    three = S.basket_no_metrics([0.40, 0.40, 0.21])  # 3 legs: capital 1.99, raw +0.01, after slippage -0.005
    assert three["capital"] == pytest.approx(1.99) and three["excess_after_slippage"] == pytest.approx(-0.005)
    assert S.basket_no_metrics([0.5, 0.5])["excess"] == pytest.approx(0.0)


def test_binding_caps():
    assert S.binding_caps({"depth": 500.0, "contract": 2000.0, "race": None}) == "depth"
    assert S.binding_caps({"depth": 900.0, "contract": 437.0, "race": 392.0}) == "race"
    assert S.binding_caps({"contract": 2000.0, "race": 2000.0, "gross": 5000.0}) == "contract+race"
    assert S.binding_caps({"race": None}) == "none"


# ---------------------------------------------------------------------- passive maker estimate


def test_maker_markouts_reference_needs_both_sides():
    times = [0.0, 10.0, 20.0, 30.0, 40.0, 400.0]
    prices = [0.50, 0.55, 0.50, 0.50, 0.49, 0.50]
    calls = [None, "buy", "sell", "buy", "sell", "buy"]
    out = S.maker_markouts(times, prices, calls, 60)
    assert out[0] == (None, "no aggressor")
    # maker SOLD at 0.55; in (10, 70] last buy 0.50, last sell 0.49 -> ref 0.495 -> +0.055
    assert out[1][1] == "ok" and out[1][0] == pytest.approx(0.055)
    # maker BOUGHT at 0.50; in (20, 80] last buy 0.50, last sell 0.49 -> ref 0.495 -> -0.005
    assert out[2][1] == "ok" and out[2][0] == pytest.approx(-0.005)
    assert out[3] == (None, "one side missing") and out[4] == (None, "one side missing")
    assert out[5] == (None, "beyond data")
    assert S.maker_sign("buy") == 1 and S.maker_sign("sell") == -1 and S.maker_sign(None) is None


def _p(day, hour, price, size, call):
    return (datetime(2026, 10, day, hour, tzinfo=timezone.utc), price, size, call)


def test_simulate_maker_inventory_daily_marks_and_caps():
    prints = [_p(1, 10, 0.40, 1000, "sell"),  # maker buys 100 (25% of 1000 = 250, capped at Q 100)
              _p(1, 11, 0.42, 400, "buy"),    # maker sells 100 -> flat, +2
              _p(1, 12, 0.45, 2000, "sell"),  # maker buys 100 at 0.45
              _p(2, 10, 0.30, 2000, "sell")]  # maker buys 100 more at 0.30
    out = S.simulate_maker_inventory(prints, 0.25, 100, 300)
    # day 1: cash -40 + 42 - 45 = -43, marked 100 @ 0.45 -> +2; day 2: cash -73 + 200 @ 0.30 = -13 -> -15
    assert out["daily"]["2026-10-01"]["pnl"] == pytest.approx(2.0) and out["daily"]["2026-10-02"]["pnl"] == pytest.approx(-15.0)
    assert out["max_abs_inventory"] == 200 and out["max_drawdown"] == pytest.approx(15.0)
    assert out["daily"]["2026-10-01"]["fills"] == 3 and out["daily"]["2026-10-01"]["shares"] == 300
    capped = S.simulate_maker_inventory(prints, 0.25, 100, 150)
    assert capped["max_abs_inventory"] == 150 and capped["daily"]["2026-10-02"]["shares"] == 50  # stops buying at the cap
    fees = S.simulate_maker_inventory(prints[:2], 0.25, 100, 300, fee_per_share=0.01)
    assert fees["daily"]["2026-10-01"]["pnl"] == pytest.approx(2.0 - 200 * 0.01)
    assert S.simulate_maker_inventory([], 0.25, 100, 300) == {"daily": {}, "max_abs_inventory": 0.0, "max_drawdown": 0.0}


def test_cluster_bootstrap_mean():
    mean, lo, hi = S.cluster_bootstrap_mean([1.0, 1.0, 1.0, -1.0], ["a", "a", "a", "b"], n_boot=500)
    assert mean == pytest.approx(0.5) and -1.0 <= lo <= 0.5 <= hi <= 1.0
    assert S.cluster_bootstrap_mean([1.0, 1.0, 1.0, -1.0], ["a", "a", "a", "b"], n_boot=500) == (mean, lo, hi)  # seeded
    assert S.cluster_bootstrap_mean([2.0, 4.0], ["a", "a"]) == (3.0, None, None)  # one cluster: no CI
    assert S.cluster_bootstrap_mean([], []) == (None, None, None)
    tight = S.cluster_bootstrap_mean([1.0] * 10 + [1.2] * 10, ["a"] * 10 + ["b"] * 10)
    assert 1.0 <= tight[1] <= tight[2] <= 1.2


def test_quartile_label():
    assert [S.quartile_label(v, [10, 50, 100]) for v in (5, 10, 11, 100, 101)] == ["Q1", "Q1", "Q2", "Q3", "Q4"]


# ---------------------------------------------------------------------- quotes


def test_prevailing_quote_respects_max_age():
    times, quotes = [0.0, 30.0, 60.0], ["q0", "q30", "q60"]
    assert S.prevailing_quote(times, quotes, 45.0, 45.0) == "q30"
    assert S.prevailing_quote(times, quotes, 30.0, 45.0) == "q30"  # at the snapshot itself
    assert S.prevailing_quote(times, quotes, 200.0, 45.0) is None  # too old
    assert S.prevailing_quote(times, quotes, -1.0, 45.0) is None  # before the first snapshot


def test_lee_ready_quote_rule():
    assert S.lee_ready(0.12, 0.10, 0.12) == "buy" and S.lee_ready(0.13, 0.10, 0.12) == "buy"
    assert S.lee_ready(0.10, 0.10, 0.12) == "sell" and S.lee_ready(0.09, 0.10, 0.12) == "sell"
    assert S.lee_ready(0.11, 0.10, 0.12) is None  # inside: fall back to the tick rule
    assert S.lee_ready(0.11, None, None) is None


def test_spread_bucket():
    assert S.spread_bucket(0.10, 0.105) == "1" and S.spread_bucket(0.10, 0.11) == "2" and S.spread_bucket(0.10, 0.13) == ">=3"
    assert S.spread_bucket(0.10, 0.10) == "1" and S.spread_bucket(None, 0.1) is None  # a locked book counts as 1


def test_msum_partials_matches_fsum_and_extends_in_place():
    values = [1e16, 1.0, -1e16, 0.1, 0.2, 0.3] * 50
    partials = S.msum_partials(values[:100])
    S.msum_partials(values[100:], partials)
    assert math.fsum(partials) == math.fsum(values) and S.msum_partials([]) == []


def test_cluster_bootstrap_sums_equals_the_mean_version():
    values = [0.5, -1.0, 2.0, 0.25, 1.5, -0.5, 3.0]
    clusters = ["a", "a", "b", "c", "c", "c", "d"]
    sums = {}
    for v, c in zip(values, clusters):
        s, n = sums.get(c, (0.0, 0))
        sums[c] = (s + v, n + 1)
    assert S.cluster_bootstrap_sums(math.fsum(values) / len(values), list(sums.values())) == S.cluster_bootstrap_mean(values, clusters)


def test_time_halves_reproduces_split_halves_with_ties():
    times = [5.0, 1.0, 3.0, 3.0, 3.0, 2.0, 7.0, 3.0]  # the cut falls inside the run of 3.0s
    rows = list(enumerate(times))
    first, _ = S.split_halves(rows, key=lambda r: r[1])
    halves = S.TimeHalves(times)
    labels = [halves.label(t) for t in times]  # original row order
    assert [i for i, lab in enumerate(labels) if lab == "first_half"] == sorted(i for i, _ in first)
    assert S.TimeHalves([]).label(1.0) is None


def test_cut_aggregator_equals_cluster_bootstrap_over_rows():
    rows = [("m1", "d1", 0.01, 10.0), ("m1", "d1", None, 11.0), ("m1", "d2", -0.005, 50.0), ("m2", "d1", 0.02, 5.0), ("m2", "d1", 0.0, 12.0)]
    agg = S.CutAggregator(0.005)
    for market in ("m1", "m2"):
        agg.add_market("k", market, [(d, v) for m, d, v, _ in rows if m == market])
    ok = [r for r in rows if r[2] is not None]
    mean, lo, hi = S.cluster_bootstrap_mean([r[2] / 0.005 for r in ok], [(r[0], r[1]) for r in ok])
    out = agg.result("k")
    assert (out["prints"], out["qualifying"], out["drop_rate"], out["markets"]) == (5, 4, 0.2, 2)
    assert (out["mean_ticks"], out["ci_low_ticks"], out["ci_high_ticks"]) == (round(mean, 3), round(lo, 3), round(hi, 3))
    timed = S.CutAggregator(0.005)  # with times: clusters in time order (m2-d1 at t=5 first), as a time-sorted row list
    for market in ("m1", "m2"):
        timed.add_market("k", market, [(d, v) for m, d, v, _ in rows if m == market], [t for m, *_, t in rows if m == market])
    by_time = sorted(ok, key=lambda r: r[3])
    mean, lo, hi = S.cluster_bootstrap_mean([r[2] / 0.005 for r in by_time], [(r[0], r[1]) for r in by_time])
    assert (timed.result("k")["ci_low_ticks"], timed.result("k")["ci_high_ticks"]) == (round(lo, 3), round(hi, 3))
    assert agg.result("missing")["prints"] == 0 and agg.result("missing")["mean_ticks"] is None


def test_oracle_fair_and_uncertainty_match_the_bot():
    import itertools
    import time as _time

    from ladder import uncertainty_band
    from oracle_api import OracleQuote
    from strategy import compute_fair_price

    def q(source, bid, ask, age):
        return OracleQuote(source, "r", bid, ask, 1.0, 1.0, None, _time.time() - age)

    shapes = [None, (0.40, 0.42, 1), (0.40, 0.42, 30), (0.30, 0.45, 1), (0.42, 0.40, 1), (None, 0.5, 1), (0.55, 0.57, 1), (0.04, 0.05, 1)]
    for a, b in itertools.product(shapes, repeat=2):
        pm = None if a is None else q("polymarket", *a)
        ks = None if b is None else q("kalshi", *b)
        fair, _ = compute_fair_price(pm, ks, 10.0)
        mine, reason = S.oracle_fair((pm, ks), 10.0)
        assert (None if fair is None else fair.prob) == mine and (reason == "ok") == (fair is not None), (a, b)
        assert S.oracle_uncertainty((pm, ks), 10.0) == uncertainty_band(pm, ks, 10.0), (a, b)
    assert S.oracle_fair((q("polymarket", 0.10, 0.12, 1), q("kalshi", 0.30, 0.32, 1)), 10.0) == (None, "disagree")
    assert S.oracle_fair((None, q("kalshi", 0.10, 0.12, 30)), 10.0) == (None, "stale")


def test_quoter_prices_and_edge_match_the_quoter():
    import itertools

    from quoter import plan_quotes

    fairs = (0.05, 0.081, 0.30, 0.5, 0.72, 0.95)
    us = (0.005, 0.0075, 0.02, 0.04)
    yes_bids = (None, 0.02, 0.06, 0.25, 0.49, 0.70, 0.93)
    no_bids = (None, 0.03, 0.27, 0.45, 0.50, 0.90)
    for fair, u, yb, nb in itertools.product(fairs, us, yes_bids, no_bids):
        plan = plan_quotes(fair, u, yb, nb, "D", 0.0)
        assert S.quoter_prices(fair, u, yb, nb) == (plan.yes, plan.no), (fair, u, yb, nb)
        assert round(S.quoter_edge(u), 4) == plan.edge


def test_join_improve_quotes_and_reach():
    # fair 0.10, u 0.005 -> edge 0.015; touch 0.07 / 0.085 (3 ticks)
    assert S.join_quotes(0.07, 0.085, 0.10, 0.015, 2) == (0.07, None)  # ask 0.085 < fair + edge
    assert S.join_quotes(0.07, 0.085, 0.10, 0.015, 4) == (None, None)  # spread below S_min
    assert S.join_quotes(None, 0.085, 0.10, 0.015, 1) == (None, None)  # one-sided
    assert S.join_quotes(0.06, 0.12, 0.09, 0.015, 1) == (0.06, 0.12)
    assert S.improve_quotes(0.07, 0.085, 0.10, 0.005, 2) == (0.075, 0.115)  # a tick inside the bid; the ask capped at fair + edge
    assert S.improve_quotes(0.09, 0.095, 0.10, 0.005, 2) == (None, None)
    assert S.touch_ticks(0.07, 0.085) == 3 and S.touch_ticks(None, 0.1) is None
    assert S.reaches("sell", 0.07, 0.07, None, exact=True) and not S.reaches("sell", 0.065, 0.07, None, exact=True)
    assert S.reaches("sell", 0.065, 0.07, None, exact=False) and not S.reaches("sell", 0.075, 0.07, None, exact=False)
    assert S.reaches("buy", 0.13, None, 0.12, exact=False) and not S.reaches("buy", 0.13, 0.07, None, exact=False)
    assert not S.reaches(None, 0.07, 0.07, 0.12, exact=True)
    assert S.capped_fill(1000) == 100 and S.capped_fill(40) == 10 and S.capped_fill(-40) == 10


def test_sell_levels_and_sweep():
    book = {"bids": [{"price": 0.90, "quantity": 100}, {"price": 0.92, "quantity": 50}], "asks": [{"price": 0.95, "quantity": 30}, {"price": 0.96, "quantity": 200}]}
    assert S.sell_levels(book, "yes") == [(0.92, 50.0), (0.90, 100.0)]
    assert S.sell_levels(book, "no") == [(0.05, 30.0), (0.04, 200.0)]  # sell NO = buy YES at the asks
    s = S.sweep_sell(S.sell_levels(book, "yes"), 120)
    assert s["filled"] == 120 and s["proceeds"] == pytest.approx(50 * 0.92 + 70 * 0.90) and s["levels"] == 2 and s["enough"]
    assert s["best"] == 0.92 and s["last"] == 0.90 and s["vwap"] == pytest.approx(s["proceeds"] / 120)
    short = S.sweep_sell(S.sell_levels(book, "yes"), 500)
    assert short["filled"] == 150 and not short["enough"]
    empty = S.sweep_sell([], 10)
    assert empty["filled"] == 0 and empty["vwap"] is None and empty["best"] is None and not empty["enough"]


def test_shrink_and_race_scenarios():
    assert S.shrink_toward_half(0.93, 0.02) == pytest.approx(0.91) and S.shrink_toward_half(0.10, 0.02) == pytest.approx(0.12)
    assert S.shrink_toward_half(0.51, 0.02) == 0.5 and S.shrink_toward_half(0.49, 0.02) == 0.5
    legs = [("D", "yes", 100, 90.0), ("R", "no", 50, 46.0)]  # both pay when D wins
    pnl = S.race_scenario_pnl(legs, ["D", "R", "other"])
    assert pnl == {"D": pytest.approx(14.0), "R": pytest.approx(-136.0), "other": pytest.approx(-86.0)}  # other wins: D YES 0 (-90), R NO pays 50 (+4)
