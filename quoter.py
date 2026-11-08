"""Quoter: two-sided quotes around the oracle fair on a few markets, skewed by race inventory.

Behind config.QUOTER_ENABLED (default OFF). When on, strategy.py calls
`QuoterManager.run_cycle()` once per cycle, after the taker executed and before the ladder.

Both quotes are BUYS: a YES bid and a NO bid (= a YES ask at 1 - price).

    edge   = max(QUOTER_MIN_EDGE, u + QUOTER_EDGE_OVER_U)
    bid YES = min(best other YES bid + tick, fair - edge)
    bid NO  = min(best other NO bid + tick, (1 - fair) - edge)      (rounded DOWN to the tick)
    skew    = QUOTER_SKEW_PER_100 * |net race delta| / 100, taken off the side that would
              grow the race delta further
    bid YES + bid NO <= 1 - 2 ticks

"Other" = the SIG book without our own resting orders (exchange_api.book_without_own).
A market qualifies only while the other participants' spread is at least
QUOTER_MIN_SPREAD_TICKS (checked every cycle; below it the quotes are withdrawn) and, unless
QUOTER_ALLOW_ONE_SIDED, both sides of their book are quoted. No quote is ever marketable
against them: the YES bid stays a tick under their best YES ask, the NO bid a tick under
their best NO ask (1 - best YES bid).
Net race delta = D - R shares over ALL positions in the race (taker, ladder, quoter): D YES
and R NO count +, D NO and R YES count -. The side that would take |delta| past
QUOTER_RACE_DELTA_CAP is not quoted. Only D / R markets are quoted.

Per cycle:
  1. SYNC     - resting quotes vs. this cycle's open-order read
  2. FILLS    - quoter fills (from strategy.py's FillFeed) -> quoter inventory (journal)
  3. KILL     - QUOTER_KILL_FILLS consecutive fills with a 5-minute markout below
                QUOTER_KILL_MARKOUT_TICKS -> cancel, block the market QUOTER_KILL_BLOCK_S
  4. GUARDS   - jump (fair moved >= QUOTER_JUMP_MOVE within QUOTER_JUMP_WINDOW_S: cancel,
                block QUOTER_JUMP_BLOCK_S), tail (fair outside [LOW, HIGH]), stale oracle /
                u > QUOTER_MAX_U: no quotes, resting ones cancelled
  5. QUOTE    - per chosen market: requote (scoped cancel-all, re-post only after a clean
                200) when a target moved >= 1 tick or a quote is near expiry; otherwise
                post only a missing side. All new quotes in one batch.

Risk: quoter inventory + resting quotes (as if filled) count in the race worst case, the
global gross cap (TOTAL_GROSS_EXPOSURE_FRAC, with taker and ladder) and the directional
cap (with taker and ladder); its own budget is QUOTER_BOOK_FRAC. Markets never overlap
with the ladder's (each side excludes the other's markets in use), because both requote
with a scoped cancel-all. Writes rank below taker orders and above the ladder in the
write bucket and keep its reserve free. DRY_RUN: virtual quotes, no request at all.

Micro-live pilot flags (config.py), ALL default off - with none set nothing above changes:
  QUOTER_JOIN_TOUCH      bid AT the other best bid (min(other bid, fair - edge)), every
                         non-crossing guard and the 2-tick sum rule kept
  QUOTER_SKIP_HELD       a market holding any position the quoter did not create is not quoted
                         (its resting quotes are withdrawn) - no netting against carry
  QUOTER_PERSIST_BLOCKS  kill / jump blocks still running are replayed from the journal at start
  QUOTER_DAILY_LOSS_CAP  quoter P&L since 00:00 UTC (realized + inventory at the SIG mid) below
                         -cap -> every quoter quote withdrawn, no quotes until the next UTC day;
                         the day's baseline and the block are journalled, so both survive a restart
  QUOTER_MAX_RESTING_FRAC  total resting quoter notional <= this x bankroll
  QUOTER_VIRTUAL_DETAIL  virtual mode: placements counted, each would-be quote logged at INFO, and
                         a "virtual fill" logged (OPTIMISTIC: no queue) when a later book crosses
                         a virtual quote or a later last trade is at or through it
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from config import (
    QUOTER_ALLOW_ONE_SIDED,
    QUOTER_BOOK_FRAC,
    QUOTER_DAILY_LOSS_CAP,
    QUOTER_DRY_RUN,
    QUOTER_EDGE_OVER_U,
    QUOTER_EXPIRY_REQUOTE_S,
    QUOTER_JUMP_BLOCK_S,
    QUOTER_JUMP_MOVE,
    QUOTER_JUMP_WINDOW_S,
    QUOTER_JOIN_TOUCH,
    QUOTER_KILL_BLOCK_S,
    QUOTER_KILL_FILLS,
    QUOTER_KILL_HORIZON_S,
    QUOTER_KILL_MARKOUT_TICKS,
    QUOTER_MARKET_ALLOWLIST,
    QUOTER_MAX_MARKETS,
    QUOTER_MAX_RESTING_FRAC,
    QUOTER_MAX_U,
    QUOTER_MIN_SPREAD_TICKS,
    QUOTER_MIN_EDGE,
    QUOTER_PERSIST_BLOCKS,
    QUOTER_RACE_DELTA_CAP,
    QUOTER_SIZE,
    QUOTER_SKIP_HELD,
    QUOTER_SKEW_PER_100,
    QUOTER_SUMMARY_INTERVAL_S,
    QUOTER_TAIL_HIGH,
    QUOTER_TAIL_LOW,
    QUOTER_TTL_S,
    QUOTER_VIRTUAL_DETAIL,
)
from exchange_api import (
    MAX_BATCH_ORDERS,
    MAX_LIMIT_PRICE,
    MIN_LIMIT_PRICE,
    PRICE_TICK,
    WRITE_RANK_QUOTER,
    ExchangeClient,
    OrderRequest,
    ReadPausedError,
    RequestTimeoutError,
    book_without_own,
    new_idempotency_key,
)
from fills import fill_price_side, fill_shares
from ladder import floor_tick, uncertainty_band
from market_mapper import DATA_DIR, MarketMap
from markout import Markout, unix_time
from oracle_api import OracleQuote
from risk import Direction, RaceLeg, RiskLimits, direction_of, directional_exposure, max_shares_within_race_cap, race_legs_from_holdings

logger = logging.getLogger(__name__)

Side = Literal["yes", "no"]

QUOTER_JOURNAL_PATH = DATA_DIR / "quoter_journal.jsonl"
VIRTUAL_PREFIX = "dry-"  # order ids of virtual quotes (DRY_RUN or QUOTER_DRY_RUN)
_EPS = 1e-9


def quoter_is_virtual(global_dry_run: bool, mode: str = QUOTER_DRY_RUN) -> bool:
    """Virtual quotes when DRY_RUN is on (always) or QUOTER_DRY_RUN is "true"."""
    return global_dry_run or mode == "true"


# ---------------------------------------------------------------------- pure maths


def party_sign(party: str) -> int:
    """+1 for a D market, -1 for an R market, 0 otherwise."""
    return {"D": 1, "R": -1}.get(party, 0)


def delta_change(party: str, side: Side, shares: float) -> float:
    """Change of the race's D - R delta from buying `shares` of `side` on a `party` market."""
    return party_sign(party) * (shares if side == "yes" else -shares)


def race_delta(market_map: MarketMap, race_key: str, holdings: Mapping[str, Any]) -> float:
    """D - R shares over every position in the race (D YES and R NO are the same bet)."""
    total = 0.0
    for mapping in market_map.by_race().get(race_key, []):
        held = holdings.get(mapping.sig_exchange_id)
        if held is not None and held.shares:
            total += delta_change(mapping.party, held.side, held.shares)
    return total


@dataclass(frozen=True)
class QuotePlan:
    yes: float | None  # YES bid, None = not quoted
    no: float | None  # NO bid
    edge: float
    skew: float
    delta_blocked: tuple[Side, ...] = ()


def plan_quotes(
    fair_yes: float,
    u: float,
    other_yes_bid: float | None,
    other_no_bid: float | None,
    party: str,
    net_delta: float,
    *,
    size: int = QUOTER_SIZE,
    min_edge: float = QUOTER_MIN_EDGE,
    skew_per_100: float = QUOTER_SKEW_PER_100,
    delta_cap: float = QUOTER_RACE_DELTA_CAP,
    join_touch: bool = QUOTER_JOIN_TOUCH,
) -> QuotePlan:
    """`join_touch`: bid AT the other best bid instead of a tick above it (QUOTER_JOIN_TOUCH)."""
    edge = max(min_edge, u + QUOTER_EDGE_OVER_U)
    targets: dict[Side, float] = {}
    for side, fair_side, other in (("yes", fair_yes, other_yes_bid), ("no", 1.0 - fair_yes, other_no_bid)):
        target = fair_side - edge
        if other is not None:
            target = min(other if join_touch else other + PRICE_TICK, target)
        targets[side] = target

    skew = skew_per_100 * abs(net_delta) / 100.0
    increasing: Side | None = None
    if abs(net_delta) > _EPS and party_sign(party):
        increasing = "yes" if delta_change(party, "yes", 1) * net_delta > 0 else "no"
        targets[increasing] -= skew

    # Never marketable against other participants: the YES bid at least a tick under their
    # best YES ask (= 1 - their best NO bid), the NO bid a tick under their best NO ask
    # (= 1 - their best YES bid). A quote that would cross is lowered, never kept.
    if other_no_bid is not None:
        targets["yes"] = min(targets["yes"], round(1.0 - other_no_bid - PRICE_TICK, 6))
    if other_yes_bid is not None:
        targets["no"] = min(targets["no"], round(1.0 - other_yes_bid - PRICE_TICK, 6))

    prices = {side: floor_tick(target) for side, target in targets.items()}
    while prices["yes"] + prices["no"] > 1.0 - 2 * PRICE_TICK + _EPS:
        lower = increasing or ("yes" if prices["yes"] >= prices["no"] else "no")
        prices[lower] = round(prices[lower] - PRICE_TICK, 3)

    blocked = tuple(
        side for side in ("yes", "no")
        if abs(net_delta + delta_change(party, side, size)) > delta_cap + _EPS
        and abs(net_delta + delta_change(party, side, size)) > abs(net_delta)
    )
    quoted = {
        side: (price if MIN_LIMIT_PRICE - _EPS <= price <= MAX_LIMIT_PRICE + _EPS and side not in blocked else None)
        for side, price in prices.items()
    }
    return QuotePlan(quoted["yes"], quoted["no"], round(edge, 4), round(skew, 4), blocked)


def in_tails(fair_yes: float, low: float = QUOTER_TAIL_LOW, high: float = QUOTER_TAIL_HIGH) -> bool:
    """True when fair is in a tail: outside [low, high]."""
    return not low - _EPS <= fair_yes <= high + _EPS


def utc_day(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%d")


def _journal_time(raw: str) -> float:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()


def virtual_cross(side: Side, price: float, book: Any, last_at_placement: float | None) -> float | None:
    """The price that would have filled a virtual quote (OPTIMISTIC, no queue), or None: the other
    side of a later book at or through it (YES bid: ask <= price; NO bid: YES bid >= 1 - price), or
    a later last trade at or through it that differs from the last trade seen at placement."""
    if book is None:
        return None
    if side == "yes":
        if book.yes_ask is not None and book.yes_ask <= price + _EPS:
            return book.yes_ask
        through = book.last is not None and book.last <= price + _EPS
    else:
        if book.yes_bid is not None and book.yes_bid >= 1.0 - price - _EPS:
            return book.yes_bid
        through = book.last is not None and book.last >= 1.0 - price - _EPS
    return book.last if through and book.last != last_at_placement else None


def jumped(history: Iterable[tuple[float, float]], now: float, fair_now: float, window_s: float = QUOTER_JUMP_WINDOW_S, move: float = QUOTER_JUMP_MOVE) -> bool:
    """Fair moved by >= `move` against any reading of the last `window_s` seconds."""
    return any(now - t <= window_s and abs(fair_now - f) >= move - _EPS for t, f in history)


def _needs_requote(resting: QuoterOrder | None, target: float | None, now: float) -> bool:
    """A resting quote is replaced when its side is no longer wanted, its target moved by at
    least one tick, or it is about to expire. A missing side is simply posted."""
    if resting is None:
        return False
    if target is None:
        return True
    return abs(resting.price - target) >= PRICE_TICK - _EPS or resting.expires_at - now < QUOTER_EXPIRY_REQUOTE_S


def spread_block_reason(
    other_yes_bid: float | None,
    other_yes_ask: float | None,
    min_ticks: int = QUOTER_MIN_SPREAD_TICKS,
    allow_one_sided: bool = QUOTER_ALLOW_ONE_SIDED,
) -> str | None:
    """Why the other participants' book rules a market out (None = it qualifies)."""
    if other_yes_bid is None or other_yes_ask is None:
        return None if allow_one_sided else "one-sided book"
    ticks = round((other_yes_ask - other_yes_bid) / PRICE_TICK)
    if ticks < min_ticks:
        return f"other spread {ticks} ticks < {min_ticks}"
    return None


def other_bids(book: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    """(best YES bid, best NO bid) of a book that holds only OTHER participants' orders."""
    if book is None:
        return None, None
    bids, asks = book.get("bids") or [], book.get("asks") or []
    yes_bid = max((float(level["price"]) for level in bids), default=None)
    yes_ask = min((float(level["price"]) for level in asks), default=None)
    return yes_bid, (round(1.0 - yes_ask, 3) if yes_ask is not None else None)


# ---------------------------------------------------------------------- state


@dataclass
class QuoterOrder:
    order_id: str
    exchange_id: str
    side: Side
    price: float  # side terms
    quantity: int  # remaining
    expires_at: float


@dataclass
class QuoterPosition:
    """Quoter inventory on one exchange, YES-signed: > 0 long YES, < 0 long NO (positions net)."""

    exchange_id: str
    shares: float = 0.0
    avg_yes: float = 0.0  # average entry, YES terms
    realized: float = 0.0

    def apply(self, side: Side, shares: float, side_price: float) -> None:
        delta = shares if side == "yes" else -shares
        price_yes = side_price if side == "yes" else 1.0 - side_price
        if abs(self.shares) < _EPS or (self.shares > 0) == (delta > 0):
            total = abs(self.shares) + abs(delta)
            self.avg_yes = (abs(self.shares) * self.avg_yes + abs(delta) * price_yes) / total
            self.shares += delta
            return
        closing = min(abs(delta), abs(self.shares))
        self.realized += closing * (price_yes - self.avg_yes) * (1 if self.shares > 0 else -1)
        flipped = abs(delta) > abs(self.shares) + _EPS
        self.shares += delta
        if flipped:
            self.avg_yes = price_yes
        elif abs(self.shares) < _EPS:
            self.shares, self.avg_yes = 0.0, 0.0

    @property
    def side(self) -> Side:
        return "yes" if self.shares > 0 else "no"

    @property
    def open_cost(self) -> float:
        return abs(self.shares) * (self.avg_yes if self.shares > 0 else 1.0 - self.avg_yes)


@dataclass
class QuoterContext:
    """What one strategy cycle hands the quoter."""

    now: float  # unix seconds
    bankroll: float
    tournament_id: str | None
    market_map: MarketMap
    fair: Mapping[str, float]  # exchange_id -> fair P(YES), fresh this cycle
    quotes: Mapping[str, tuple[OracleQuote | None, OracleQuote | None]]
    books: Mapping[str, Any]  # exchange_id -> SigBook (top of book incl. our own orders)
    holdings: Mapping[str, Any]  # ALL positions (taker, ladder, quoter)
    open_orders: list[dict[str, Any]]
    limits: RiskLimits
    max_quote_age_s: float
    other_gross: float = 0.0  # taker hold book + ladder cost
    other_directional: Mapping[Direction, float] = field(default_factory=dict)
    other_race_legs: Mapping[str, list[RaceLeg]] = field(default_factory=dict)  # resting ladder orders
    exclude_markets: frozenset[str] = frozenset()  # the ladder's markets in use
    fills: list[dict[str, Any]] | None = None  # this cycle's new fills (all sources)
    markouts: list[Markout] = field(default_factory=list)  # markouts written this cycle


@dataclass
class QuoterReport:
    markets: int = 0
    placed: int = 0
    cancels: int = 0
    requotes: int = 0
    fills: int = 0
    killed: int = 0
    blocked_jump: int = 0
    blocked_tail: int = 0
    blocked_oracle: int = 0  # no fresh fair or u > QUOTER_MAX_U
    blocked_delta: int = 0  # sides not quoted: race delta cap
    blocked_budget: int = 0
    blocked_global: int = 0
    blocked_direction: int = 0
    blocked_race: int = 0
    blocked_held: int = 0  # QUOTER_SKIP_HELD: markets with a non-quoter position
    blocked_resting: int = 0  # QUOTER_MAX_RESTING_FRAC: sides not quoted
    loss_blocked: bool = False  # QUOTER_DAILY_LOSS_CAP: no quotes for the rest of the UTC day
    virtual_placed: int = 0  # QUOTER_VIRTUAL_DETAIL
    virtual_fills: int = 0  # QUOTER_VIRTUAL_DETAIL (optimistic)
    skipped: str | None = None


class QuoterJournal:
    """Append-only JSONL: intents, placements, requotes, fills, guards."""

    def __init__(self, path: Path = QUOTER_JOURNAL_PATH) -> None:
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def replay(self, include_closed: bool = False) -> dict[str, QuoterPosition]:
        """Quoter inventory rebuilt from every journalled fill (prices in the fill's side terms);
        `include_closed` keeps flat positions too (their realized P&L)."""
        positions: dict[str, QuoterPosition] = {}
        if not self.path.exists():
            return positions
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if entry.get("event") == "fill":
                    eid = entry["exchange_id"]
                    positions.setdefault(eid, QuoterPosition(eid)).apply(entry["side"], entry["quantity"], entry["price"])
        return positions if include_closed else {eid: p for eid, p in positions.items() if abs(p.shares) > _EPS}

    def replay_blocks(self, now: float) -> tuple[dict[str, float], dict[str, float], set[str]]:
        """(kill / jump blocks still running at `now` - journal time + block_s -, the daily-loss
        baseline per UTC day ("day_open"), the UTC days blocked by the daily loss cap)."""
        blocks: dict[str, float] = {}
        day_open: dict[str, float] = {}
        loss_days: set[str] = set()
        if not self.path.exists():
            return blocks, day_open, loss_days
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            event = entry.get("event")
            if event in ("kill", "jump") and entry.get("block_s"):
                until = _journal_time(entry["ts"]) + float(entry["block_s"])
                if until > now:
                    blocks[entry["exchange_id"]] = max(blocks.get(entry["exchange_id"], 0.0), until)
            elif event == "day_open":
                day_open[entry["day"]] = float(entry["value"])
            elif event == "daily_loss_block":
                loss_days.add(entry["day"])
        return blocks, day_open, loss_days


# ---------------------------------------------------------------------- manager


class QuoterManager:
    def __init__(
        self,
        exchange: ExchangeClient,
        journal: QuoterJournal | None = None,
        clock: Callable[[], float] = time.time,
        dry_run_mode: str = QUOTER_DRY_RUN,
        allowlist: frozenset[str] | None = QUOTER_MARKET_ALLOWLIST,
        min_spread_ticks: int = QUOTER_MIN_SPREAD_TICKS,
        allow_one_sided: bool = QUOTER_ALLOW_ONE_SIDED,
        join_touch: bool = QUOTER_JOIN_TOUCH,
        skip_held: bool = QUOTER_SKIP_HELD,
        persist_blocks: bool = QUOTER_PERSIST_BLOCKS,
        daily_loss_cap: float = QUOTER_DAILY_LOSS_CAP,
        max_resting_frac: float = QUOTER_MAX_RESTING_FRAC,
        virtual_detail: bool = QUOTER_VIRTUAL_DETAIL,
    ) -> None:
        self._exchange = exchange
        self._dry_run_mode = dry_run_mode
        self._allowlist = allowlist
        self._min_spread_ticks = min_spread_ticks
        self._allow_one_sided = allow_one_sided
        self._journal = journal or QuoterJournal()
        self._clock = clock
        self.orders: dict[str, QuoterOrder] = {}  # order_id -> resting quote
        self._order_meta: dict[str, QuoterOrder] = {}  # every quote placed by this process
        self.positions = self._journal.replay()
        self.markets: list[str] = []  # quoted last cycle (kept while eligible)
        self._fair_history: dict[str, deque[tuple[float, float]]] = {}
        self._blocked_until: dict[str, float] = {}
        self._bad_streak: dict[str, int] = {}
        self._fills_since = clock()
        self._seen_fill_ids: set[Any] = set()
        self._virtual_ids = 0
        # "Quoter summary" (INFO, every QUOTER_SUMMARY_INTERVAL_S): totals since the last one.
        self._summary_at = clock()
        self._summary: Counter[str] = Counter()
        self._blocked_sides: Counter[str] = Counter()
        self._last_skips: Counter[str] = Counter()
        # Pilot flags (all off by default: then nothing below is used).
        self._join_touch, self._skip_held, self._virtual_detail = join_touch, skip_held, virtual_detail
        self._daily_loss_cap, self._max_resting_frac = daily_loss_cap, max_resting_frac
        self._day: str | None = None  # UTC day of the daily-loss baseline
        self._day_open: float | None = None  # quoter P&L at that day's first cycle
        self._loss_blocked_day: str | None = None
        self._realized_closed = 0.0  # realized P&L of positions flat at start (replay drops them)
        self._last_mid: dict[str, float] = {}
        self._virtual_last: dict[str, float | None] = {}  # virtual order id -> last trade at placement
        self._tops: dict[str, tuple[float | None, float | None]] = {}  # last cycle's other top of book (virtual quote log)
        if persist_blocks or daily_loss_cap > 0:
            blocks, day_open, loss_days = self._journal.replay_blocks(clock())
            if persist_blocks:
                self._blocked_until.update(blocks)
                if blocks:
                    logger.info("Quoter blocks restored from the journal", extra={"blocks_s": {eid: round(t - clock()) for eid, t in sorted(blocks.items())}})
            if daily_loss_cap > 0:
                today = utc_day(clock())
                if today in day_open:
                    self._day, self._day_open = today, day_open[today]
                if today in loss_days:
                    self._loss_blocked_day = today
                    logger.warning("Quoter daily loss block restored from the journal: no quotes until the next UTC day", extra={"day": today})
                self._realized_closed = sum(p.realized for eid, p in self._journal.replay(include_closed=True).items() if eid not in self.positions)
        pilot = {"join_touch": join_touch, "skip_held": skip_held, "persist_blocks": persist_blocks, "daily_loss_cap": daily_loss_cap,
                 "max_resting_frac": max_resting_frac, "virtual_detail": virtual_detail}
        if any(pilot.values()):
            logger.info("Quoter pilot flags", extra=pilot)
        logger.info(
            "Quoter enabled",
            extra={
                "open_positions": len(self.positions), "max_markets": QUOTER_MAX_MARKETS, "size": QUOTER_SIZE,
                "virtual": self.virtual, "dry_run_mode": dry_run_mode,
                "allowlist": sorted(allowlist) if allowlist is not None else None,
            },
        )

    @property
    def virtual(self) -> bool:
        """True: quotes are only simulated - no quoter request reaches the API."""
        return quoter_is_virtual(self._exchange.dry_run, self._dry_run_mode)

    # ------------------------------------------------------------ views used by strategy.py

    def order_ids(self) -> set[str]:
        return set(self.orders)

    def order_action(self, order_id: str) -> str | None:
        return "buy" if order_id in self._order_meta else None

    def markets_in_use(self) -> frozenset[str]:
        """Chosen markets, markets with resting quotes or quoter inventory - off limits to the ladder."""
        return frozenset(self.markets) | {o.exchange_id for o in self.orders.values()} | {
            eid for eid, p in self.positions.items() if abs(p.shares) > _EPS
        }

    def hold_book(self, holdings: Mapping[str, Any]) -> dict[str, Any]:
        """Holdings minus quoter inventory: what the taker strategy (and its gross cap) sees."""
        result = dict(holdings)
        for eid, pos in self.positions.items():
            held = result.get(eid)
            if held is None or abs(pos.shares) < _EPS or held.side != pos.side:
                continue
            remaining = held.shares - abs(pos.shares)
            if remaining <= 0:
                del result[eid]
            else:
                scale = remaining / held.shares
                result[eid] = type(held)(held.side, remaining, held.market_value * scale, held.cost_basis * scale)
        return result

    def _counted_orders(self, for_others: bool) -> list[QuoterOrder]:
        """Resting quotes that count against limits. Virtual quotes count for the quoter
        itself, and for the other books only when the whole bot is in DRY_RUN - with only
        the quoter virtual (QUOTER_DRY_RUN) they must not shrink the live taker's room."""
        if not for_others or self._exchange.dry_run:
            return list(self.orders.values())
        return [o for o in self.orders.values() if not o.order_id.startswith(VIRTUAL_PREFIX)]

    def resting_race_legs(self, market_map: MarketMap, for_others: bool = True) -> dict[str, list[RaceLeg]]:
        legs: dict[str, list[RaceLeg]] = {}
        for order in self._counted_orders(for_others):
            mapping = market_map.get(order.exchange_id)
            if mapping is not None and order.quantity > 0:
                legs.setdefault(mapping.race.key, []).append(RaceLeg(mapping.party, order.side, order.quantity, order.quantity * order.price))
        return legs

    def book_cost(self, for_others: bool = True) -> float:
        """Charged to QUOTER_BOOK_FRAC (and the global gross cap): resting quotes at notional + inventory cost."""
        return sum(o.quantity * o.price for o in self._counted_orders(for_others)) + sum(p.open_cost for p in self.positions.values())

    def directional_cost(self, market_map: MarketMap, for_others: bool = True) -> dict[Direction, float]:
        return directional_exposure(
            market_map,
            [(o.exchange_id, o.side, o.quantity * o.price) for o in self._counted_orders(for_others)]
            + [(p.exchange_id, p.side, p.open_cost) for p in self.positions.values() if abs(p.shares) > _EPS],
        )

    # ------------------------------------------------------------ the cycle

    async def run_cycle(self, ctx: QuoterContext) -> QuoterReport:
        report = QuoterReport()
        if not ctx.tournament_id:
            report.skipped = "no tournament id"
            logger.warning("Quoter skipped: tournament id unknown")
            return report
        self._sync_resting(ctx.open_orders)
        if self._virtual_detail and self.virtual:
            self._check_virtual_fills(ctx, report)
        self._apply_fills(ctx, report)
        await self._apply_markouts(ctx, report)
        if self._daily_loss_cap > 0:
            self._check_daily_loss(ctx, report)
        await self._guard_and_quote(ctx, report)
        self._log_summary(ctx, report)
        return report

    def _log_summary(self, ctx: QuoterContext, report: QuoterReport) -> None:
        for name in ("requotes", "placed", "cancels", "fills", "killed"):
            self._summary[name] += getattr(report, name)
        for name in ("delta", "budget", "global", "direction", "race", "held", "resting"):
            self._blocked_sides[name] += getattr(report, f"blocked_{name}")  # zeros are never logged
        if self._virtual_detail and self.virtual:
            self._summary["virtual_placed"] += report.virtual_placed
            self._summary["virtual_fills"] += report.virtual_fills
        if report.loss_blocked:
            self._summary["loss_blocked_cycles"] += 1
        self._summary["cycles"] += 1
        if ctx.now - self._summary_at < QUOTER_SUMMARY_INTERVAL_S:
            return
        logger.info(
            "Quoter summary",
            extra={
                "interval_s": round(ctx.now - self._summary_at, 1),
                "virtual": self.virtual,
                "markets": list(self.markets),
                "active_quotes": len(self.orders),
                **dict(self._summary),
                "skipped_markets": dict(self._last_skips),  # last cycle: reason -> markets
                "blocked_sides": {k: v for k, v in self._blocked_sides.items() if v},  # whole interval
            },
        )
        self._summary_at = ctx.now
        self._summary.clear()
        self._blocked_sides.clear()

    def _check_virtual_fills(self, ctx: QuoterContext, report: QuoterReport) -> None:
        """QUOTER_VIRTUAL_DETAIL: a virtual quote crossed by this cycle's book or last trade is logged
        as a virtual fill (OPTIMISTIC: queue position not modelled) and removed, so its side is
        posted again. Virtual fills never touch the quoter inventory."""
        for order_id in [oid for oid in self._virtual_last if oid not in self.orders]:
            del self._virtual_last[order_id]
        for order_id, order in list(self.orders.items()):
            if not order_id.startswith(VIRTUAL_PREFIX):
                continue
            at = virtual_cross(order.side, order.price, ctx.books.get(order.exchange_id), self._virtual_last.get(order_id))
            if at is None:
                continue
            del self.orders[order_id]
            self._virtual_last.pop(order_id, None)
            report.virtual_fills += 1
            fields = {"exchange_id": order.exchange_id, "side": order.side, "price": order.price, "quantity": order.quantity, "crossed_at_yes": at,
                      "fair": None if ctx.fair.get(order.exchange_id) is None else round(ctx.fair[order.exchange_id], 4)}
            logger.info("Quoter virtual fill (OPTIMISTIC: queue position not modelled)", extra=fields)
            self._journal.write("virtual_fill", order_id=order_id, **fields)

    def _quoter_pnl(self, ctx: QuoterContext) -> float:
        """All-time quoter P&L: realized + inventory marked to the SIG mid (the last two-sided mid
        seen for that market; a position never marked counts at its entry price)."""
        for eid in self.positions:
            book = ctx.books.get(eid)
            if book is not None and book.yes_bid is not None and book.yes_ask is not None:
                self._last_mid[eid] = (book.yes_bid + book.yes_ask) / 2
        total = self._realized_closed
        for eid, pos in self.positions.items():
            total += pos.realized
            mid = self._last_mid.get(eid)
            if mid is not None and abs(pos.shares) > _EPS:
                total += pos.shares * (mid - pos.avg_yes)  # YES-signed shares: right for NO inventory too
        return total

    def _check_daily_loss(self, ctx: QuoterContext, report: QuoterReport) -> None:
        """QUOTER_DAILY_LOSS_CAP: the first cycle of a UTC day fixes its baseline (journalled); P&L
        since then below -cap blocks quoting until the next UTC day (journalled: survives a restart)."""
        today = utc_day(ctx.now)
        value = self._quoter_pnl(ctx)
        if self._day != today:
            self._day, self._day_open = today, value
            self._journal.write("day_open", day=today, value=value)
        day_pnl = value - self._day_open
        if self._loss_blocked_day != today and day_pnl < -self._daily_loss_cap - _EPS:
            self._loss_blocked_day = today
            logger.warning("Quoter daily loss cap hit: every quoter quote withdrawn, no quotes until the next UTC day",
                           extra={"day": today, "day_pnl": round(day_pnl, 2), "cap": self._daily_loss_cap})
            self._journal.write("daily_loss_block", day=today, day_pnl=round(day_pnl, 4), cap=self._daily_loss_cap)
        report.loss_blocked = self._loss_blocked_day == today

    def _held_elsewhere(self, holdings: Mapping[str, Any], eid: str) -> bool:
        """QUOTER_SKIP_HELD: the exchange holds shares the quoter did not create (YES-signed net
        position minus quoter inventory)."""
        held = holdings.get(eid)
        if held is None or not held.shares:
            return False
        signed = held.shares if held.side == "yes" else -held.shares
        pos = self.positions.get(eid)
        return abs(signed - (pos.shares if pos is not None else 0.0)) > _EPS

    def _sync_resting(self, open_orders: list[dict[str, Any]]) -> None:
        if self.virtual:
            return  # virtual quotes never appear in the open-order read
        open_by_id = {str(o.get("id")): o for o in open_orders}
        for order_id in list(self.orders):
            live = open_by_id.get(order_id)
            if live is None:
                del self.orders[order_id]  # filled, expired or cancelled; fills come from the fills read
            else:
                self.orders[order_id].quantity = int(abs(live.get("quantity") or 0))

    def _apply_fills(self, ctx: QuoterContext, report: QuoterReport) -> None:
        for fill in ctx.fills or []:
            order = self._order_meta.get(str(fill.get("orderId")))
            if order is None or fill.get("id") in self._seen_fill_ids or unix_time(fill.get("filledAt")) < self._fills_since:
                continue
            self._seen_fill_ids.add(fill.get("id"))
            price = fill_price_side(fill, "rest", order.side)  # side terms, as QuoterPosition.apply expects
            quantity = fill_shares(fill)
            if price is None or quantity <= 0:
                continue
            pos = self.positions.setdefault(order.exchange_id, QuoterPosition(order.exchange_id))
            pos.apply(order.side, quantity, price)
            report.fills += 1
            self._journal.write(
                "fill", exchange_id=order.exchange_id, side=order.side, price=price, quantity=quantity,
                order_id=fill.get("orderId"), fill_id=fill.get("id"), net_shares=round(pos.shares, 4),
            )

    async def _apply_markouts(self, ctx: QuoterContext, report: QuoterReport) -> None:
        """Per market, consecutive 5-minute markouts below the threshold -> kill the market."""
        for markout in ctx.markouts:
            if markout.source != "quoter" or markout.horizon_s != QUOTER_KILL_HORIZON_S or markout.ticks is None:
                continue
            eid = markout.exchange_id
            self._bad_streak[eid] = self._bad_streak.get(eid, 0) + 1 if markout.ticks < QUOTER_KILL_MARKOUT_TICKS else 0
            if self._bad_streak[eid] >= QUOTER_KILL_FILLS:
                self._bad_streak[eid] = 0
                self._blocked_until[eid] = ctx.now + QUOTER_KILL_BLOCK_S
                report.killed += 1
                logger.warning(
                    "Quoter market killed: consecutive adverse markouts",
                    extra={"exchange_id": eid, "fills": QUOTER_KILL_FILLS, "threshold_ticks": QUOTER_KILL_MARKOUT_TICKS, "block_s": QUOTER_KILL_BLOCK_S},
                )
                self._journal.write("kill", exchange_id=eid, block_s=QUOTER_KILL_BLOCK_S)
                await self._cancel_market(ctx, eid, report)

    async def _cancel_market(self, ctx: QuoterContext, eid: str, report: QuoterReport) -> bool:
        """Scoped cancel-all; True only when it came back clean (nothing left uncancelled)."""
        if not any(o.exchange_id == eid for o in self.orders.values()):
            return True
        report.cancels += 1
        if self.virtual:
            result = None  # virtual quotes: nothing to cancel at the exchange
        else:
            result = await self._exchange.cancel_all_orders(exchange_id=eid, tournament_id=ctx.tournament_id)
        clean = result is None or not result.get("errors")  # None = DRY_RUN / virtual
        if clean:
            for order_id in [oid for oid, o in self.orders.items() if o.exchange_id == eid]:
                del self.orders[order_id]
        return clean

    def _block_reason(self, ctx: QuoterContext, eid: str) -> str | None:
        """Why this market gets no quotes this cycle (None = may be quoted)."""
        if ctx.now < self._blocked_until.get(eid, 0.0):
            return "blocked"
        fair = ctx.fair.get(eid)
        if fair is None:
            return "oracle"
        pm, ks = ctx.quotes.get(eid, (None, None))
        if uncertainty_band(pm, ks, ctx.max_quote_age_s) > QUOTER_MAX_U + _EPS:
            return "oracle"
        if in_tails(fair):
            return "tail"
        return None

    def _choose_markets(self, eligible: set[str], tops: Mapping[str, tuple[float | None, float | None]]) -> list[str]:
        """Keep last cycle's markets while eligible, fill up with the widest OTHER spreads
        (only markets that passed the spread threshold are in `eligible`)."""
        kept = [eid for eid in self.markets if eid in eligible]

        def spread(eid: str) -> float:
            bid, ask = tops.get(eid, (None, None))
            return 0.0 if bid is None or ask is None else ask - bid

        rest = sorted(eligible - set(kept), key=lambda eid: (-spread(eid), eid))
        return (kept + rest)[:QUOTER_MAX_MARKETS]

    async def _other_top(self, ctx: QuoterContext, eid: str) -> tuple[float | None, float | None]:
        """Best OTHER YES bid and YES ask. The bulk top of book already excludes us unless one
        of our own orders (any book's) sits at it; only then is the book read (never when
        virtual: virtual quotes are not in the real book)."""
        book = ctx.books.get(eid)
        yes_bid = book.yes_bid if book is not None else None
        yes_ask = book.yes_ask if book is not None else None
        own = self._exchange.own_orders(eid) if not self.virtual else []
        at_top = any((side == "bid" and price == yes_bid) or (side == "ask" and price == yes_ask) for side, price, _ in own)
        if at_top:
            other_yes_bid, other_no_bid = other_bids(book_without_own(await self._exchange.get_orderbook(eid), own))
            return other_yes_bid, (round(1.0 - other_no_bid, 3) if other_no_bid is not None else None)
        return yes_bid, yes_ask

    async def _guard_and_quote(self, ctx: QuoterContext, report: QuoterReport) -> None:
        # Jump guard over every priced market (a jump blocks the market for a while).
        for eid, fair in ctx.fair.items():
            history = self._fair_history.setdefault(eid, deque())
            while history and ctx.now - history[0][0] > QUOTER_JUMP_WINDOW_S:
                history.popleft()
            if jumped(history, ctx.now, fair) and ctx.now >= self._blocked_until.get(eid, 0.0):
                self._blocked_until[eid] = ctx.now + QUOTER_JUMP_BLOCK_S
                report.blocked_jump += 1
                logger.info("Quoter jump guard", extra={"exchange_id": eid, "fair": round(fair, 4), "history": list(history), "block_s": QUOTER_JUMP_BLOCK_S})
                self._journal.write("jump", exchange_id=eid, fair=round(fair, 4), block_s=QUOTER_JUMP_BLOCK_S)
            history.append((ctx.now, fair))

        eligible: set[str] = set()
        reasons: dict[str, str] = {}
        for eid in set(ctx.fair) | set(self.markets) | {o.exchange_id for o in self.orders.values()}:
            mapping = ctx.market_map.get(eid)
            if mapping is None or party_sign(mapping.party) == 0:
                reasons[eid] = "not D / R"
                continue
            if report.loss_blocked:
                reasons[eid] = "daily loss cap"
                continue
            if self._allowlist is not None and eid not in self._allowlist:
                reasons[eid] = "not allowlisted"
                continue
            if eid in ctx.exclude_markets:
                reasons[eid] = "ladder market"
                continue
            if self._skip_held and self._held_elsewhere(ctx.holdings, eid):
                reasons[eid] = "held (not quoter)"
                report.blocked_held += 1
                continue
            reason = self._block_reason(ctx, eid)
            if reason is None:
                eligible.add(eid)
            else:
                reasons[eid] = reason

        # The other participants' top of book, and the spread threshold - every cycle, so a
        # quoted market whose spread narrowed is withdrawn below. A market whose book cannot
        # be read this cycle keeps what rests there (if it was quoted) and is not re-quoted.
        tops: dict[str, tuple[float | None, float | None]] = {}
        unreadable: set[str] = set()
        for eid in sorted(eligible):
            try:
                tops[eid] = await self._other_top(ctx, eid)
            except (RequestTimeoutError, ReadPausedError) as exc:
                logger.warning("Quoter: order book unavailable, market skipped this cycle", extra={"exchange_id": eid, "error": repr(exc)})
                unreadable.add(eid)
                continue
            reason = spread_block_reason(*tops[eid], self._min_spread_ticks, self._allow_one_sided)
            if reason is not None:
                reasons[eid] = reason
        eligible = {eid for eid in eligible if eid in tops and eid not in reasons}
        self.markets = self._choose_markets(eligible | (unreadable & set(self.markets)), tops)
        for eid in eligible - set(self.markets):
            reasons[eid] = "not chosen (max markets)"
        report.markets = len(self.markets)
        self._last_skips = Counter(reason for reason in reasons.values() if reason != "not D / R")
        if logger.isEnabledFor(logging.DEBUG):
            for eid, reason in sorted(reasons.items()):
                if reason != "not D / R" and eid in ctx.fair:
                    logger.debug("Quoter market skipped", extra={"exchange_id": eid, "fair": round(ctx.fair[eid], 4), "reason": reason})
        report.blocked_tail = sum(1 for r in reasons.values() if r == "tail")
        report.blocked_oracle = sum(1 for r in reasons.values() if r == "oracle")

        # Pass 1 - cancels: markets no longer quoted, and requotes (re-posted below only
        # after a clean cancel). Targets are computed here, from the books before any post.
        plans: dict[str, QuotePlan] = {}
        to_post: dict[str, tuple[Side, ...]] = {}
        for eid in sorted({o.exchange_id for o in self.orders.values()} - set(self.markets)):
            reason = reasons.get(eid, "not chosen")
            self._journal.write("withdraw", exchange_id=eid, reason=reason)
            logger.info("Quoter quotes withdrawn", extra={"exchange_id": eid, "reason": reason})
            await self._cancel_market(ctx, eid, report)
        for eid in self.markets:
            if eid in unreadable:
                continue  # keep whatever rests there, try again next cycle
            mapping = ctx.market_map.get(eid)
            yes_bid, yes_ask = tops[eid]
            no_bid = round(1.0 - yes_ask, 3) if yes_ask is not None else None
            net = race_delta(ctx.market_map, mapping.race.key, ctx.holdings)
            plan = plan_quotes(ctx.fair[eid], uncertainty_band(*ctx.quotes.get(eid, (None, None)), ctx.max_quote_age_s), yes_bid, no_bid, mapping.party, net,
                               join_touch=self._join_touch)
            plans[eid] = plan
            logger.debug(
                "Quoter market",
                extra={
                    "exchange_id": eid, "fair": round(ctx.fair[eid], 4),
                    "other_yes_bid": yes_bid, "other_yes_ask": yes_ask,
                    "bid_yes": plan.yes, "bid_no": plan.no, "edge": plan.edge, "skew": plan.skew, "race_delta": net,
                    "reason": f"delta cap: {', '.join(plan.delta_blocked)} not quoted" if plan.delta_blocked else None,
                },
            )
            report.blocked_delta += len(plan.delta_blocked)
            resting = {o.side: o for o in self.orders.values() if o.exchange_id == eid}
            targets = {"yes": plan.yes, "no": plan.no}
            if any(_needs_requote(resting.get(side), targets[side], ctx.now) for side in ("yes", "no")):
                self._journal.write("requote", exchange_id=eid, yes=plan.yes, no=plan.no, resting={s: o.price for s, o in resting.items()})
                report.requotes += 1
                if not await self._cancel_market(ctx, eid, report):
                    continue
                resting = {}
            to_post[eid] = tuple(side for side in ("yes", "no") if targets[side] is not None and side not in resting)

        # Pass 2 - sizing from the state after the cancels: own budget, global gross and
        # directional caps (with taker + ladder), race worst case incl. everything resting.
        race_legs = race_legs_from_holdings(ctx.market_map, ctx.holdings)
        for source in (ctx.other_race_legs, self.resting_race_legs(ctx.market_map, for_others=False)):
            for race, legs in source.items():
                race_legs.setdefault(race, []).extend(legs)
        race_parties = {key: [m.party for m in group] for key, group in ctx.market_map.by_race().items()}
        race_cap = ctx.limits.max_race_worst_loss_fraction * ctx.bankroll
        own_cost = self.book_cost(for_others=False)
        budget = QUOTER_BOOK_FRAC * ctx.bankroll - own_cost
        total_room = ctx.limits.max_total_gross_exposure_fraction * ctx.bankroll - ctx.other_gross - own_cost
        direction_cap = ctx.limits.max_directional_fraction * ctx.bankroll
        resting_room = None  # QUOTER_MAX_RESTING_FRAC: room left for resting quoter notional
        if self._max_resting_frac > 0:
            resting_room = self._max_resting_frac * ctx.bankroll - sum(o.quantity * o.price for o in self.orders.values())
        direction_used: dict[Direction, float] = dict(ctx.other_directional)
        for direction, cost in self.directional_cost(ctx.market_map, for_others=False).items():
            direction_used[direction] = direction_used.get(direction, 0.0) + cost

        items: list[tuple[OrderRequest, dict[str, Any]]] = []
        for eid, sides in to_post.items():
            mapping = ctx.market_map.get(eid)
            legs = race_legs.setdefault(mapping.race.key, [])
            parties = race_parties.get(mapping.race.key, [mapping.party])
            for side in sides:
                price = getattr(plans[eid], side)
                cost = QUOTER_SIZE * price
                direction = direction_of(ctx.market_map, eid, side)
                if cost > budget + _EPS:
                    report.blocked_budget += 1
                    continue
                if resting_room is not None and cost > resting_room + _EPS:
                    report.blocked_resting += 1
                    continue
                if cost > total_room + _EPS:
                    report.blocked_global += 1
                    continue
                if direction is not Direction.NEUTRAL and direction_used.get(direction, 0.0) + cost > direction_cap + _EPS:
                    report.blocked_direction += 1
                    continue
                quantity = max_shares_within_race_cap(parties, legs, mapping.party, side, price, QUOTER_SIZE, race_cap)
                if quantity <= 0:
                    report.blocked_race += 1
                    continue
                cost = quantity * price
                if resting_room is not None:
                    resting_room -= cost
                legs.append(RaceLeg(mapping.party, side, quantity, cost))
                budget -= cost
                total_room -= cost
                direction_used[direction] = direction_used.get(direction, 0.0) + cost
                meta = {"exchange_id": eid, "side": side, "price": price, "quantity": quantity, "fair": round(ctx.fair[eid], 4), "edge": plans[eid].edge, "skew": plans[eid].skew}
                self._journal.write("intent", **meta)
                items.append((self._buy(eid, side, price, quantity, ctx), meta))
        self._tops = tops
        report.placed += await self._send(ctx, items, report)

    def _buy(self, eid: str, side: Side, price: float, quantity: int, ctx: QuoterContext) -> OrderRequest:
        expires = datetime.fromtimestamp(ctx.now + QUOTER_TTL_S, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return OrderRequest(exchange_id=eid, side=side, action="buy", quantity=quantity, price=price, tournament_id=ctx.tournament_id, expiration_date=expires)

    async def _send(self, ctx: QuoterContext, items: list[tuple[OrderRequest, dict[str, Any]]], report: QuoterReport | None = None) -> int:
        """Batches of <= 50, one key each. A failed item is not retried here: the next cycle
        sees the side missing and posts it again."""
        placed = 0
        if self._exchange.halted:
            return 0
        if self.virtual:  # remember virtual quotes, send nothing
            for order, meta in items:
                self._virtual_ids += 1
                self._register(f"{VIRTUAL_PREFIX}{self._virtual_ids}", order, meta, ctx)
                if self._virtual_detail:
                    self._log_virtual_quote(f"{VIRTUAL_PREFIX}{self._virtual_ids}", meta, ctx, report)
            if items:
                logger.debug("Quoter: virtual quotes", extra={"quotes": len(items)})
            return 0
        for start in range(0, len(items), MAX_BATCH_ORDERS):
            chunk = items[start : start + MAX_BATCH_ORDERS]
            results = await self._exchange.place_batch([order for order, _ in chunk], new_idempotency_key("quoter"), write_rank=WRITE_RANK_QUOTER)
            if results is None:  # halted between the check and the send: nothing placed
                continue
            for (order, meta), result in zip(chunk, results):
                if result.get("ok"):
                    data = result.get("data") or {}
                    remaining = int(abs(data.get("remainingQuantity", order.quantity) or 0))
                    self._register(str(data.get("orderId")), order, meta, ctx, remaining)
                    placed += 1
                else:
                    self._journal.write("placement_failed", status=result.get("status"), data=result.get("data"), **meta)
        return placed

    def _log_virtual_quote(self, order_id: str, meta: Mapping[str, Any], ctx: QuoterContext, report: QuoterReport | None) -> None:
        """QUOTER_VIRTUAL_DETAIL: one INFO line per would-be quote, with the touch it was priced against."""
        eid = meta["exchange_id"]
        yes_bid, yes_ask = self._tops.get(eid, (None, None))
        book = ctx.books.get(eid)
        self._virtual_last[order_id] = None if book is None else book.last
        if report is not None:
            report.virtual_placed += 1
        logger.info("Quoter virtual quote", extra={
            **meta, "order_id": order_id, "other_yes_bid": yes_bid, "other_yes_ask": yes_ask,
            "spread_ticks": None if yes_bid is None or yes_ask is None else round((yes_ask - yes_bid) / PRICE_TICK),
            "join_touch": self._join_touch, "last_trade": self._virtual_last[order_id]})

    def _register(self, order_id: str, order: OrderRequest, meta: dict[str, Any], ctx: QuoterContext, remaining: int | None = None) -> None:
        self._journal.write("placement", order_id=order_id, **meta)
        quote = QuoterOrder(order_id, order.exchange_id, order.side, order.price, order.quantity if remaining is None else remaining, ctx.now + QUOTER_TTL_S)
        self._order_meta[order_id] = quote
        if quote.quantity > 0:
            self.orders[order_id] = quote
