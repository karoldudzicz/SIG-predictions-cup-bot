"""Env overrides of the SIG GET knobs: invalid values fall back to the defaults, never raise."""

from __future__ import annotations

import pytest

import config


@pytest.fixture(autouse=True)
def _clean_warnings():
    before = list(config.CONFIG_WARNINGS)
    config.CONFIG_WARNINGS.clear()
    yield
    config.CONFIG_WARNINGS[:] = before


def _timeout(raw):
    return config._env_override("SIG_GET_TIMEOUT_S", 4.0, cast=float, low=0.5, high=60.0, environ={"SIG_GET_TIMEOUT_S": raw})


def _attempts(raw):
    return config._env_override("SIG_GET_MAX_ATTEMPTS", 4, cast=int, low=1, high=10, environ={"SIG_GET_MAX_ATTEMPTS": raw})


@pytest.mark.parametrize("raw", ["abc", "-1", "0", "999", "nan", "inf"])
def test_invalid_timeout_override_falls_back_to_default(raw):
    assert _timeout(raw) == 4.0
    assert config.CONFIG_WARNINGS[-1]["variable"] == "SIG_GET_TIMEOUT_S"


@pytest.mark.parametrize("raw", ["3.5", "0", "11", "x"])
def test_invalid_attempts_override_falls_back_to_default(raw):
    assert _attempts(raw) == 4
    assert len(config.CONFIG_WARNINGS) == 1


def test_valid_overrides_are_used_without_warnings():
    assert _timeout(" 2.5 ") == 2.5
    assert _attempts("6") == 6
    assert config.CONFIG_WARNINGS == []


def test_unset_or_blank_uses_default_silently():
    assert config._env_override("X", 4.0, cast=float, low=0.5, high=60.0, environ={}) == 4.0
    assert config._env_override("X", 4.0, cast=float, low=0.5, high=60.0, environ={"X": "  "}) == 4.0
    assert config.CONFIG_WARNINGS == []


def test_shipped_defaults_are_unchanged():
    # Holds as long as no override is set in the environment / .env of the test machine.
    assert (config.SIG_GET_TIMEOUT_S, config.SIG_GET_MAX_ATTEMPTS, config.SIG_GET_DEADLINE_S) == (4.0, 4, 20.0)


# ---------------------------------------------------------------------- ladder overrides


def test_ladder_bool_override():
    assert config._env_bool_override("LADDER_ENABLED", False, environ={"LADDER_ENABLED": "true"}) is True
    assert config._env_bool_override("LADDER_ENABLED", False, environ={"LADDER_ENABLED": "OFF"}) is False
    assert config._env_bool_override("LADDER_ENABLED", False, environ={"LADDER_ENABLED": "maybe"}) is False
    assert config.CONFIG_WARNINGS[-1]["variable"] == "LADDER_ENABLED"


@pytest.mark.parametrize(
    "env, expected_warning",
    [
        ({"LADDER_OFFSETS": "0.05,abc"}, "LADDER_OFFSETS"),
        ({"LADDER_OFFSETS": "0.05,1.5"}, "LADDER_OFFSETS"),
        ({"LADDER_OFFSETS": "0.10,0.05", "LADDER_SIZES": "100,200"}, "LADDER_OFFSETS"),  # not increasing
        ({"LADDER_SIZES": "100,-5,200,300"}, "LADDER_SIZES"),
        ({"LADDER_SIZES": "100,200"}, "LADDER_OFFSETS/LADDER_SIZES"),  # length mismatch with default offsets
    ],
)
def test_bad_ladder_level_lists_fall_back_to_defaults(env, expected_warning):
    defaults = ((0.07, 0.15, 0.30, 0.50), (300, 500, 800, 1200))
    assert config._ladder_levels_override(*defaults, environ=env) == defaults
    assert config.CONFIG_WARNINGS[-1]["variable"] == expected_warning


def test_valid_ladder_level_lists_are_used():
    env = {"LADDER_OFFSETS": "0.10, 0.20", "LADDER_SIZES": "100,200"}
    assert config._ladder_levels_override((0.07,), (300,), environ=env) == ((0.10, 0.20), (100, 200))
    assert config.CONFIG_WARNINGS == []


def test_bad_env_values_do_not_break_startup(monkeypatch):
    """Reload config with garbage in every ladder variable: no exception, safe defaults, warnings."""
    import importlib

    garbage = {
        "LADDER_ENABLED": "yes please",
        "LADDER_OFFSETS": "x,y",
        "LADDER_SIZES": "1e9",
        "LADDER_MAX_COST_PER_MARKET": "-1",
        "LADDER_MAX_MARKETS": "lots",
        "LADDER_BOOK_FRAC": "2",
        "LADDER_REQUOTE_MOVE": "nan",
        "LADDER_ORDER_TTL_S": "60",
        "SIG_WRITES_PER_MIN": "25",  # 8 burst + 25 > 30
        "LADDER_BUCKET_RESERVE": "8",  # >= burst: the ladder could never write
        "TOTAL_GROSS_EXPOSURE_FRAC": "0",
        "QUOTER_ENABLED": "maybe",
        "QUOTER_SIZE": "0",
        "QUOTER_MIN_EDGE": "0.001",
        "QUOTER_TAIL_LOW": "0.6",
        "QUOTER_TTL_S": "5",
        "SIG_READS_PER_MIN": "100",  # above SIG's own 100/min limit minus headroom
        "RELATIONSHIPS_CHECK_ENABLED": "sometimes",
        "QUOTER_DRY_RUN": "maybe",
        "QUOTER_MARKET_ALLOWLIST": "abc",
        "QUOTER_MIN_SPREAD_TICKS": "0",
        "QUOTER_ALLOW_ONE_SIDED": "maybe",
    }
    for name, value in garbage.items():
        monkeypatch.setenv(name, value)
    try:
        reloaded = importlib.reload(config)
        assert reloaded.LADDER_ENABLED is False
        assert (reloaded.LADDER_OFFSETS, reloaded.LADDER_SIZES) == ((0.07, 0.15, 0.30, 0.50), (300, 500, 800, 1200))
        assert (reloaded.LADDER_MAX_COST_PER_MARKET, reloaded.LADDER_MAX_MARKETS, reloaded.LADDER_BOOK_FRAC) == (2000, 10, 0.10)
        assert (reloaded.LADDER_REQUOTE_MOVE, reloaded.LADDER_ORDER_TTL_S, reloaded.SIG_WRITES_PER_MIN) == (0.02, 7200, 20)
        assert (reloaded.QUOTER_ENABLED, reloaded.QUOTER_SIZE, reloaded.QUOTER_MIN_EDGE, reloaded.QUOTER_TAIL_LOW, reloaded.QUOTER_TTL_S) == (False, 100, 0.01, 0.05, 120)
        assert (reloaded.SIG_READS_PER_MIN, reloaded.RELATIONSHIPS_CHECK_ENABLED) == (80, "auto")
        assert (reloaded.QUOTER_MIN_SPREAD_TICKS, reloaded.QUOTER_ALLOW_ONE_SIDED) == (3, False)
        assert (reloaded.QUOTER_DRY_RUN, reloaded.QUOTER_MARKET_ALLOWLIST) == ("inherit", frozenset())  # broken allowlist: nothing
        assert (reloaded.SIG_WRITE_BURST, reloaded.LADDER_BUCKET_RESERVE, reloaded.TOTAL_GROSS_EXPOSURE_FRAC) == (8, 3, 0.40)
        assert {w["variable"] for w in reloaded.CONFIG_WARNINGS} >= set(garbage) - {"LADDER_SIZES"}
    finally:
        for name in garbage:
            monkeypatch.delenv(name)
        importlib.reload(config)  # back to the clean defaults for the other tests
