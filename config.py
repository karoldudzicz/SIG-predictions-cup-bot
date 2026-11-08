"""Runtime configuration loaded from environment variables and the .env file.

Real environment variables (e.g. set by systemd on the VPS) take priority over .env.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_SIG_BASE_URL = "https://www.thesuper.market/api/v1"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"
DEFAULT_TOURNAMENT_SLUG = "midterm-elections"
# Oracle quotes older than this are ignored; campaign-season prices move fast.
DEFAULT_ORACLE_STALENESS_THRESHOLD_S = 10.0
DEFAULT_CYCLE_INTERVAL_S = 15.0  # pause between strategy cycles

# ---------------------------------------------------------------------- risk thresholds
# Plain constants, not env vars: changing one is a code change visible in git history.
# All *_FRAC values are fractions of bankroll (account value from /portfolio/pnl).

# Uncertainty haircut on the oracle fair probability before sizing:
#   p_used = fair - margin,  margin = max(MIN_ORACLE_MARGIN, half the oracle bid-ask spread).
# No trade unless p_used > price.
MIN_ORACLE_MARGIN = 0.01
# Near-certain contracts: Kelly's 1 / (1 - price) turns a 2c edge into a max-size bet,
# and a 2c oracle error flips its sign. Above MAX_BUY_PRICE, require a bigger shrunk edge.
MAX_BUY_PRICE = 0.93
HIGH_PRICE_MIN_EDGE = 0.04
# Per-market cap on the cost of the target position (was 5%).
MAX_MARKET_FRAC = 0.02
# Worst-case loss of everything held and planned in ONE race, over every "who wins"
# scenario - so D YES and R NO in the same race count as the same bet.
MAX_RACE_WORST_LOSS_FRAC = 0.02
# Total cost of open positions plus orders planned this cycle: MAX_GROSS_EXPOSURE_FRAC, set
# from TAKER_GROSS_FRAC further down (env override).
# New buy notional per cycle - limits how fast capital is deployed.
MAX_NEW_NOTIONAL_PER_CYCLE_FRAC = 0.05
# When a cap shrinks a target below what we already hold: False = hold the excess
# (selling into a thin bid just to satisfy a newer, tighter cap costs the spread);
# True = sell down to the target. Buys stay blocked by the caps either way.
SELL_ABOVE_TARGET = False

# ---------------------------------------------------------------------- env overrides
# A few HTTP knobs can be tuned from the environment (or .env) without a code change.
# They are parsed at import, before logging is configured, so problems are collected
# here and main.py logs them right after setup_logging. Real environment variables win
# over .env (override=False), same as load_settings().
load_dotenv(PROJECT_ROOT / ".env", override=False)
CONFIG_WARNINGS: list[dict[str, Any]] = []


def _env_override(
    name: str,
    default: float | int,
    *,
    cast: type,
    low: float,
    high: float,
    environ: Mapping[str, str] | None = None,
) -> float | int:
    """Value of env var `name` cast to `cast` if set and within [low, high]; else `default`.

    Invalid, non-positive or out-of-bounds values never raise: they fall back to the
    default and are recorded in CONFIG_WARNINGS.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError:
        CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": default, "problem": f"not a valid {cast.__name__}"})
        return default
    if not low <= value <= high:  # NaN fails this too
        CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": default, "problem": f"outside [{low}, {high}]"})
        return default
    return value


def _env_bool_override(name: str, default: bool, environ: Mapping[str, str] | None = None) -> bool:
    """true/false/1/0/yes/no/on/off; anything else -> default + CONFIG_WARNINGS."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": default, "problem": "not a boolean"})
    return default


def _env_choice_override(name: str, default: str, choices: tuple[str, ...], environ: Mapping[str, str] | None = None) -> str:
    """One of `choices` (case-insensitive); anything else -> default + CONFIG_WARNINGS."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in choices:
        return value
    CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": default, "problem": f"not one of {choices}"})
    return default


def _env_id_list_override(name: str, environ: Mapping[str, str] | None = None) -> frozenset[str] | None:
    """Comma-separated exchange ids. Unset / empty -> None (no list). Items that are not
    numeric ids are dropped with a CONFIG_WARNING; if none is left the list is EMPTY (matches
    nothing) - a broken allowlist must never widen to "every market"."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in items if not item.isdigit()]
    if bad:
        CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": "only the valid ids", "problem": f"not exchange ids: {bad}"})
    return frozenset(item for item in items if item.isdigit())


def _env_tuple_override(
    name: str,
    default: tuple,
    *,
    cast: type,
    low: float,
    high: float,
    max_len: int = 10,
    environ: Mapping[str, str] | None = None,
) -> tuple:
    """Comma-separated list, every item cast and within [low, high]; else `default` + CONFIG_WARNINGS."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not 1 <= len(parts) <= max_len:
        CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": default, "problem": f"needs 1..{max_len} items"})
        return default
    try:
        values = tuple(cast(part) for part in parts)
    except ValueError:
        CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": default, "problem": f"items must be {cast.__name__}"})
        return default
    if not all(low <= v <= high for v in values):
        CONFIG_WARNINGS.append({"variable": name, "value": raw, "default": default, "problem": f"items outside [{low}, {high}]"})
        return default
    return values


def _ladder_levels_override(
    default_offsets: tuple, default_sizes: tuple, environ: Mapping[str, str] | None = None
) -> tuple[tuple, tuple]:
    """LADDER_OFFSETS / LADDER_SIZES as a consistent pair: same length, offsets strictly rising."""
    offsets = _env_tuple_override("LADDER_OFFSETS", default_offsets, cast=float, low=0.005, high=0.99, environ=environ)
    sizes = _env_tuple_override("LADDER_SIZES", default_sizes, cast=int, low=1, high=100_000, environ=environ)
    if len(offsets) != len(sizes):
        CONFIG_WARNINGS.append(
            {"variable": "LADDER_OFFSETS/LADDER_SIZES", "value": f"{offsets} / {sizes}", "default": (default_offsets, default_sizes), "problem": "different lengths"}
        )
        return default_offsets, default_sizes
    if any(a >= b for a, b in zip(offsets, offsets[1:])):
        CONFIG_WARNINGS.append({"variable": "LADDER_OFFSETS", "value": offsets, "default": default_offsets, "problem": "offsets must strictly increase"})
        return default_offsets, default_sizes
    return offsets, sizes


# ---------------------------------------------------------------------- SIG HTTP behaviour
# SIG latency is bimodal (~0.3 s or 5-15 s per request). Reads are idempotent, so a slow
# GET is abandoned early and retried at once instead of waiting out the long timeout.
# The three GET knobs below can be overridden by env vars of the same name.
SIG_GET_TIMEOUT_S = _env_override("SIG_GET_TIMEOUT_S", 4.0, cast=float, low=0.5, high=60.0)  # per GET attempt
SIG_GET_MAX_ATTEMPTS = _env_override("SIG_GET_MAX_ATTEMPTS", 4, cast=int, low=1, high=10)  # per logical GET
# hard bound on one logical GET, retries and waits included
SIG_GET_DEADLINE_S = _env_override("SIG_GET_DEADLINE_S", 20.0, cast=float, low=1.0, high=120.0)
# POST /orders, DELETE /orders/{id}, POST /orders/cancel-all: unchanged long timeout and
# retry policy - order placement retries rely on idempotencyKey, cancels must not give up early.
SIG_POST_TIMEOUT_S = 15.0
# Order placement only (POST /orders, /orders/multi-leg): hard bound on one logical POST,
# all retries and waits included. On expiry the order is journaled UNKNOWN. Cancels and
# cancel-all are NOT bounded by this - the kill switch keeps its own timeout.
SIG_POST_DEADLINE_S = 60.0
# A journaled UNKNOWN order (may have been accepted late) blocks new orders on the same
# exchange and side for this long, so a late fill is never doubled.
UNKNOWN_ORDER_BLOCK_S = 300.0
# Startup reconciliation (cancel-all, positions) keeps retrying transient SIG failures
# (timeouts, retryable 429/5xx, connection errors) for up to this long before giving up,
# so a slow or flaky SIG does not crash-loop the service.
STARTUP_RETRY_BUDGET_S = 600.0
# Visibility during outages: an ERROR "No successful cycle" once no cycle has succeeded for
# this long, then again every NO_SUCCESS_ALERT_REPEAT_S until one does. Logging only.
NO_SUCCESS_ALERT_S = 900.0
NO_SUCCESS_ALERT_REPEAT_S = 300.0
# Max concurrent SIG GETs (smooths the order-book burst). Never applied to cancels or
# order placement, so the kill switch / shutdown cancel-all never queue behind reads.
SIG_MAX_CONCURRENCY = 6
# Every SIG read (each GET attempt) takes a token: SIG_READS_PER_MIN per minute, bursts of
# SIG_READ_BURST. SIG allows 100 reads/min per account; the rest is headroom for retries
# and the read-only research scripts. Reads WAIT for a token (inside the GET deadline)
# instead of going out and coming back 429. Classes, best first: essential (positions,
# orders, pnl, tournament, account) > bulk prices > order books > everything else (env).
SIG_READS_PER_MIN = _env_override("SIG_READS_PER_MIN", 80, cast=float, low=20.0, high=95.0)
SIG_READ_BURST = 10
# After a 429 on a read, ALL reads pause until now + Retry-After (at most this long; this
# long when the header is missing): order books and non-essential reads fail fast,
# essential reads and bulk prices wait for the pause to end.
SIG_READ_PAUSE_MAX_S = 60.0
# Taker order books per cycle (strategy._fit_to_books). The cycle reads only as many books as
# the read bucket can serve now plus what it refills within BOOK_READ_WINDOW_S, minus what
# the next cycle's essential / bulk reads will need; a read waits for its token at most
# BOOK_READ_WINDOW_S, never up to the GET deadline. The other orders are deferred to the next
# cycle (no request, no wait). A book read within BOOK_CACHE_MAX_AGE_S is reused instead of
# read again; its age then counts in the order's quote age (ORDER_MAX_QUOTE_AGE_S).
BOOK_READ_WINDOW_S = 3.0
BOOK_CACHE_MAX_AGE_S = 10.0
# GET /relationships/constraints (observe-only violations): "auto" = called once at start,
# then every cycle only while SIG has relationships for our markets (none so far, so the
# read is skipped); "on" = every cycle; "off" = never (env).
RELATIONSHIPS_CHECK_ENABLED = _env_choice_override("RELATIONSHIPS_CHECK_ENABLED", "auto", ("auto", "on", "off"))
# At least this long between the end of one cycle and the start of the next, even after a
# cycle that took longer than the interval.
MIN_CYCLE_GAP_S = 5.0

# ---------------------------------------------------------------------- deep ladder (OFF)
# Resting BUY limits far below fair catch market orders that sweep thin books; each fill
# is sold back near fair the same day. See ladder.py. Separate from the taker strategy.
# The constants marked (env) can be overridden by env vars of the same name (or .env);
# invalid values fall back to the default with a WARNING at startup. Pilot defaults:
# small book (10%) on few markets (10).
LADDER_ENABLED = _env_bool_override("LADDER_ENABLED", False)  # (env)
# Distance below fair (favourite side's terms) and shares per level, same order (env, comma lists).
LADDER_OFFSETS, LADDER_SIZES = _ladder_levels_override((0.07, 0.15, 0.30, 0.50), (300, 500, 800, 1200))
# Resting ladder cost per market (sum of price * size) (env).
LADDER_MAX_COST_PER_MARKET = _env_override("LADDER_MAX_COST_PER_MARKET", 2000, cast=float, low=1.0, high=100_000.0)
LADDER_MAX_MARKETS = _env_override("LADDER_MAX_MARKETS", 10, cast=int, low=1, high=500)  # (env)
# The ladder's own budget (resting orders + ladder inventory), separate from the taker's
# MAX_GROSS_EXPOSURE_FRAC, which is now the "hold book" only (env).
LADDER_BOOK_FRAC = _env_override("LADDER_BOOK_FRAC", 0.10, cast=float, low=0.001, high=0.5)
# Global gross cap across both books: taker hold-book cost + ladder inventory cost + resting
# ladder orders at notional. The taker sizes into the room the ladder leaves; the ladder
# adds no level once the sum reaches it (env).
# The taker's two 30% caps, one env knob (env): the portfolio cap in sizing
# (risk.size_portfolio, held market value + targets) and the gross cap on orders
# (strategy.apply_exposure_caps, hold-book cost).
TAKER_GROSS_FRAC = _env_override("TAKER_GROSS_FRAC", 0.30, cast=float, low=0.10, high=0.80)
MAX_GROSS_EXPOSURE_FRAC = TAKER_GROSS_FRAC
# Never below the taker's own cap: effectively max(TOTAL_GROSS_EXPOSURE_FRAC, TAKER_GROSS_FRAC).
TOTAL_GROSS_EXPOSURE_FRAC = max(_env_override("TOTAL_GROSS_EXPOSURE_FRAC", 0.40, cast=float, low=0.05, high=1.0), TAKER_GROSS_FRAC)
# Directional limit of the taker's sizing (env). "nominal": held market value + target
# stakes per direction <= max_directional_fraction (15%) of bankroll. "stress": the loss of a
# direction's positions in its adverse national wave - every party's win-probability logit
# moved by DIRECTIONAL_STRESS_S - must stay <= DIRECTIONAL_STRESS_FRAC of bankroll; see
# risk.stress_directional_cap and scripts/directional_stress_preview.py (same code).
DIRECTIONAL_LIMIT_MODE = _env_choice_override("DIRECTIONAL_LIMIT_MODE", "nominal", ("nominal", "stress"))
DIRECTIONAL_STRESS_S = _env_override("DIRECTIONAL_STRESS_S", 0.5, cast=float, low=0.05, high=2.0)
DIRECTIONAL_STRESS_FRAC = _env_override("DIRECTIONAL_STRESS_FRAC", 0.04, cast=float, low=0.005, high=0.5)
# Fill markouts (markout.py): one fills read per cycle (none in DRY_RUN), data/markout.jsonl (env).
MARKOUT_ENABLED = _env_bool_override("MARKOUT_ENABLED", True)

# ---------------------------------------------------------------------- quoter
# Two-sided quoting around the oracle fair on a few markets (quoter.py), behind
# QUOTER_ENABLED. Constants marked (env) take env overrides; invalid values fall back
# to the default with a startup WARNING.
QUOTER_ENABLED = _env_bool_override("QUOTER_ENABLED", False)  # (env)
QUOTER_MAX_MARKETS = _env_override("QUOTER_MAX_MARKETS", 5, cast=int, low=1, high=50)  # (env)
QUOTER_SIZE = _env_override("QUOTER_SIZE", 100, cast=int, low=1, high=10_000)  # shares per side (env)
# Minimum distance of each quote from fair, in that side's terms; never less than u + QUOTER_EDGE_OVER_U (env).
QUOTER_MIN_EDGE = _env_override("QUOTER_MIN_EDGE", 0.01, cast=float, low=0.005, high=0.2)
QUOTER_EDGE_OVER_U = 0.01
# Price taken off the inventory-increasing side per 100 shares of net race delta (env).
QUOTER_SKEW_PER_100 = _env_override("QUOTER_SKEW_PER_100", 0.005, cast=float, low=0.0, high=0.05)
# |D - R| shares per race beyond which the increasing side is not quoted (env).
QUOTER_RACE_DELTA_CAP = _env_override("QUOTER_RACE_DELTA_CAP", 300, cast=int, low=1, high=100_000)
QUOTER_TTL_S = _env_override("QUOTER_TTL_S", 120, cast=float, low=30.0, high=3600.0)  # quote expiry (env)
QUOTER_EXPIRY_REQUOTE_S = 20  # requote a market when a quote is this close to expiry
# Jump guard: fair moved >= QUOTER_JUMP_MOVE within QUOTER_JUMP_WINDOW_S -> cancel, block (env).
QUOTER_JUMP_MOVE = _env_override("QUOTER_JUMP_MOVE", 0.03, cast=float, low=0.005, high=0.5)
QUOTER_JUMP_WINDOW_S = _env_override("QUOTER_JUMP_WINDOW_S", 30, cast=float, low=5.0, high=600.0)
QUOTER_JUMP_BLOCK_S = _env_override("QUOTER_JUMP_BLOCK_S", 600, cast=float, low=0.0, high=86_400.0)
# Tail guard: no quotes while fair is outside [QUOTER_TAIL_LOW, QUOTER_TAIL_HIGH] (env).
QUOTER_TAIL_LOW = _env_override("QUOTER_TAIL_LOW", 0.05, cast=float, low=0.005, high=0.49)
QUOTER_TAIL_HIGH = _env_override("QUOTER_TAIL_HIGH", 0.95, cast=float, low=0.51, high=0.995)
QUOTER_MAX_U = 0.04  # no quotes while the oracle uncertainty band is wider than this
# Quoter's own budget: resting quotes at notional + quoter inventory cost (env).
QUOTER_BOOK_FRAC = _env_override("QUOTER_BOOK_FRAC", 0.10, cast=float, low=0.001, high=0.5)
# Per-market kill: this many consecutive quoter fills with a 5-minute markout below
# QUOTER_KILL_MARKOUT_TICKS -> cancel and block the market for QUOTER_KILL_BLOCK_S.
QUOTER_KILL_FILLS = 3
QUOTER_KILL_MARKOUT_TICKS = -2.0
QUOTER_KILL_HORIZON_S = 300
QUOTER_KILL_BLOCK_S = 3600
# Pilot controls (env). QUOTER_MARKET_ALLOWLIST: comma-separated exchange ids; when set, the
# quoter quotes only these (every guard still applies); unset = any eligible market.
QUOTER_MARKET_ALLOWLIST = _env_id_list_override("QUOTER_MARKET_ALLOWLIST")
# QUOTER_DRY_RUN: "inherit" = follow DRY_RUN; "true" = virtual quotes even when DRY_RUN is
# false (the taker still trades live); "false" = same as inherit - DRY_RUN=true always wins.
QUOTER_DRY_RUN = _env_choice_override("QUOTER_DRY_RUN", "inherit", ("inherit", "true", "false"))
QUOTER_SUMMARY_INTERVAL_S = 60  # one INFO "Quoter summary" line at most this often
# A market is quoted only while the OTHER participants' spread (their best ask - best bid,
# our own orders taken out) is at least this many ticks; below it, quotes are withdrawn (env).
QUOTER_MIN_SPREAD_TICKS = _env_override("QUOTER_MIN_SPREAD_TICKS", 3, cast=int, low=1, high=50)
# One-sided or empty book (no other bid or no other ask): not quoted unless this is set to
# true explicitly - quotes would then sit at fair - edge with nothing to anchor them (env).
QUOTER_ALLOW_ONE_SIDED = _env_bool_override("QUOTER_ALLOW_ONE_SIDED", False)
# Micro-live pilot flags (env), ALL default off: with none set the quoter behaves exactly as
# before them (tests/test_quoter_default_identical.py replays a recorded baseline).
#   QUOTER_JOIN_TOUCH      bid AT the other participants' touch, not a tick inside it (still
#                          capped at fair - edge, all non-crossing guards kept)
#   QUOTER_SKIP_HELD       no quotes on a market holding any position the quoter did not create
#   QUOTER_PERSIST_BLOCKS  kill / jump blocks survive a restart (replayed from the journal)
#   QUOTER_DAILY_LOSS_CAP  SUSQ, 0 = off: quoter UTC-day P&L (inventory marked to the SIG mid)
#                          below -cap -> cancel every quoter order, no quotes until the next UTC day
#   QUOTER_MAX_RESTING_FRAC  0 = off: total resting quoter notional <= this x bankroll
#   QUOTER_VIRTUAL_DETAIL  virtual quoter: count placements, log each would-be quote, log a
#                          "virtual fill" (optimistic, no queue) when a later book crosses it
QUOTER_JOIN_TOUCH = _env_bool_override("QUOTER_JOIN_TOUCH", False)
QUOTER_SKIP_HELD = _env_bool_override("QUOTER_SKIP_HELD", False)
QUOTER_PERSIST_BLOCKS = _env_bool_override("QUOTER_PERSIST_BLOCKS", False)
QUOTER_DAILY_LOSS_CAP = _env_override("QUOTER_DAILY_LOSS_CAP", 0.0, cast=float, low=0.0, high=1_000_000.0)
QUOTER_MAX_RESTING_FRAC = _env_override("QUOTER_MAX_RESTING_FRAC", 0.0, cast=float, low=0.0, high=0.5)
QUOTER_VIRTUAL_DETAIL = _env_bool_override("QUOTER_VIRTUAL_DETAIL", False)
# Requote a market when fair moved more than this (env).
LADDER_REQUOTE_MOVE = _env_override("LADDER_REQUOTE_MOVE", 0.02, cast=float, low=0.005, high=0.5)
# Resting order lifetime (env). At least 900 s so it outlives LADDER_EXPIRY_REQUOTE_S (600 s);
# a shorter TTL would requote every market on every cycle.
LADDER_ORDER_TTL_S = _env_override("LADDER_ORDER_TTL_S", 7200, cast=float, low=900.0, high=86_400.0)
LADDER_MIN_EXTRA_DISTANCE = 0.03  # no level closer to fair than u + this
LADDER_EXPIRY_REQUOTE_S = 600  # requote when an order is this close to expiry
LADDER_EXIT_STEP_S = 600  # noise exit: one tick lower every this many seconds
LADDER_INFO_BLOCK_S = 2700  # information exit: block the market this long
LADDER_BREAKER_RACES = 3  # fills in this many races ...
LADDER_BREAKER_WINDOW_S = 60  # ... within this window -> cancel all ladder orders
LADDER_BREAKER_PAUSE_S = 900  # ... and pause the ladder this long
# All SIG writes (orders, batches, cancels) go through one token bucket: at most
# SIG_WRITES_PER_MIN per minute, bursts of SIG_WRITE_BURST. SIG allows 30 writes/min on a
# standard account; burst + rate stays under it in any 60 s window (8 + 20 = 28).
# Priority: cancels, then taker orders, then ladder placements - which only take a token
# when at least LADDER_BUCKET_RESERVE tokens are left after it.
# SIG_WRITES_PER_MIN is env-overridable, at most 22 so burst + rate never exceeds 30.
SIG_WRITE_BURST = 8
SIG_WRITES_PER_MIN = _env_override("SIG_WRITES_PER_MIN", 20, cast=float, low=1.0, high=30.0 - SIG_WRITE_BURST)
LADDER_BUCKET_RESERVE = _env_override("LADDER_BUCKET_RESERVE", 3, cast=int, low=0, high=SIG_WRITE_BURST - 1)  # (env)

# An order is not sent if the oracle quotes behind its fair price are older than this
# at send time (slow order-book fetches or earlier POSTs in the cycle can age them).
ORDER_MAX_QUOTE_AGE_S = 20.0

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def _parse_bool(name: str, raw: str | None, default: bool) -> bool:
    # Unknown values are an error, not a silent default: a typo in DRY_RUN
    # must never quietly switch the bot into live trading.
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be one of true/false/1/0/yes/no/on/off, got {raw!r}")


def _parse_log_level(raw: str | None) -> str:
    value = (raw or DEFAULT_LOG_LEVEL).strip().upper()
    if value not in _LOG_LEVELS:
        raise ConfigError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got {raw!r}")
    return value


def _parse_positive_float(name: str, raw: str | None, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {raw!r}")
    return value


def _resolve_dir(raw: str | None, default: str) -> Path:
    # Relative paths are anchored at the project root, not the current working
    # directory, so the bot behaves the same no matter where it is started from.
    path = Path(raw.strip() if raw and raw.strip() else default)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    sig_api_key: str = field(repr=False)
    sig_base_url: str
    dry_run: bool
    log_level: str
    log_dir: Path
    tournament_slug: str
    oracle_staleness_threshold_s: float
    cycle_interval_s: float

    def require_api_key(self) -> str:
        """Return the API key, or fail loudly if it has not been configured."""
        if not self.sig_api_key:
            raise ConfigError("SIG_API_KEY is not set - add it to .env")
        return self.sig_api_key

    def summary(self) -> dict[str, Any]:
        """Loggable view of the settings. Never contains the API key itself."""
        return {
            "sig_api_key": "<set>" if self.sig_api_key else "<missing>",
            "sig_base_url": self.sig_base_url,
            "dry_run": self.dry_run,
            "log_level": self.log_level,
            "log_dir": str(self.log_dir),
            "tournament_slug": self.tournament_slug,
            "oracle_staleness_threshold_s": self.oracle_staleness_threshold_s,
            "cycle_interval_s": self.cycle_interval_s,
        }


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from the environment, reading .env first (without overriding)."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    return Settings(
        sig_api_key=os.getenv("SIG_API_KEY", "").strip(),
        sig_base_url=(os.getenv("SIG_BASE_URL") or DEFAULT_SIG_BASE_URL).strip().rstrip("/"),
        dry_run=_parse_bool("DRY_RUN", os.getenv("DRY_RUN"), default=True),
        log_level=_parse_log_level(os.getenv("LOG_LEVEL")),
        log_dir=_resolve_dir(os.getenv("LOG_DIR"), DEFAULT_LOG_DIR),
        tournament_slug=(os.getenv("SIG_TOURNAMENT_SLUG") or DEFAULT_TOURNAMENT_SLUG).strip(),
        oracle_staleness_threshold_s=_parse_positive_float(
            "ORACLE_STALENESS_THRESHOLD_S",
            os.getenv("ORACLE_STALENESS_THRESHOLD_S"),
            DEFAULT_ORACLE_STALENESS_THRESHOLD_S,
        ),
        cycle_interval_s=_parse_positive_float(
            "CYCLE_INTERVAL_S", os.getenv("CYCLE_INTERVAL_S"), DEFAULT_CYCLE_INTERVAL_S
        ),
    )


if __name__ == "__main__":
    # Sanity check: `python config.py` loads settings and logs them (key masked).
    from logging_config import setup_logging

    settings = load_settings()
    log_path = setup_logging(settings.log_level, settings.log_dir)
    logger = logging.getLogger("config")
    logger.info("Settings loaded", extra=settings.summary())
    if not settings.sig_api_key:
        logger.warning("SIG_API_KEY is empty - exchange_api.py will not be able to authenticate")
    if not settings.dry_run:
        logger.warning("DRY_RUN is OFF - orders will be sent for real")
    logger.info("Log file written", extra={"log_path": str(log_path)})
