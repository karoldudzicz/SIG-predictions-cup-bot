"""Deep ladder: resting BUY limits far below fair that catch sweeps of thin books.

Behind config.LADDER_ENABLED (default OFF). When on, strategy.py calls
`LadderManager.run_cycle()` once per cycle, after the taker strategy executed.

Per cycle:
  1. SYNC     - resting ladder orders vs. this cycle's open-order read (no extra request)
  2. FILLS    - new fills from GET /tournaments/{slug}/portfolio/fills; ladder buy fills
                become ladder inventory, exit fills close it (realized P&L per round)
  3. BREAKER  - ladder fills in >= LADDER_BREAKER_RACES races within LADDER_BREAKER_WINDOW_S
                -> one cancel-all for the tournament, ladder paused LADDER_BREAKER_PAUSE_S
  4. CLASSIFY - each new ladder fill: refetch both oracles for that market, then
                information exit (fair moved toward the fill by > half the distance:
                sell at best bid, block the market) / noise exit (moved < u: passive sell
                one tick under the best ask, a tick lower every LADDER_EXIT_STEP_S, never
                below fair - u) / hold (both oracles stale: hold, add nothing)
  5. QUOTE    - per chosen market: requote (scoped cancel-all, re-post only after a clean
                200) when fair moved > LADDER_REQUOTE_MOVE or an order is near expiry;
                otherwise place missing levels. Batches of <= 50, one key per batch.
  6. EXITS    - place / reprice exit sells.

Risk: resting ladder orders count as filled in the race worst-case check (here and in the
taker's apply_exposure_caps); ladder orders + inventory use LADDER_BOOK_FRAC and never the
taker's gross ("hold book") cap. All books (taker, quoter, ladder) share
TOTAL_GROSS_EXPOSURE_FRAC, and a new ladder level must also fit the directional cap with
the taker's and quoter's books counted in. The ladder never quotes a market the quoter
uses (and vice versa): both requote with a scoped cancel-all. Every write goes through ExchangeClient's write
bucket (SIG_WRITES_PER_MIN; cancels, then taker orders, then ladder placements, which keep
LADDER_BUCKET_RESERVE tokens free) and respects DRY_RUN and the kill switch.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from config import (
    LADDER_BOOK_FRAC,
    LADDER_BREAKER_PAUSE_S,
    LADDER_BREAKER_RACES,
    LADDER_BREAKER_WINDOW_S,
    LADDER_EXIT_STEP_S,
    LADDER_EXPIRY_REQUOTE_S,
    LADDER_INFO_BLOCK_S,
    LADDER_MAX_COST_PER_MARKET,
    LADDER_MAX_MARKETS,
    LADDER_MIN_EXTRA_DISTANCE,
    LADDER_OFFSETS,
    LADDER_ORDER_TTL_S,
    LADDER_REQUOTE_MOVE,
    LADDER_SIZES,
)
from fills import fill_price_side, fill_shares
from exchange_api import (
    MAX_BATCH_ORDERS,
    MAX_LIMIT_PRICE,
    MIN_LIMIT_PRICE,
    PRICE_TICK,
    WRITE_RANK_LADDER,
    ExchangeClient,
    OrderRequest,
    new_idempotency_key,
)
from market_mapper import DATA_DIR, MarketMap
from oracle_api import OracleQuote
from risk import (
    Direction,
    RaceLeg,
    RiskLimits,
    direction_of,
    directional_exposure,
    max_shares_within_race_cap,
    race_legs_from_holdings,
)

logger = logging.getLogger(__name__)

Side = Literal["yes", "no"]

LADDER_JOURNAL_PATH = DATA_DIR / "ladder_journal.jsonl"
HOLES_CSV_PATH = DATA_DIR / "edge_scan" / "holes.csv"
INFO_EXIT_TTL_S = 60  # information exit sells at the bid; short-lived, re-posted if unfilled
_EPS = 1e-9


# ---------------------------------------------------------------------- pure maths


def floor_tick(price: float) -> float:
    return round(math.floor(price / PRICE_TICK + _EPS) * PRICE_TICK, 3)


def ceil_tick(price: float) -> float:
    return round(math.ceil(price / PRICE_TICK - _EPS) * PRICE_TICK, 3)


def uncertainty_band(polymarket: OracleQuote | None, kalshi: OracleQuote | None, max_age_s: float) -> float:
    """u = max(tick, half of the tightest usable oracle spread, |p_pm - p_ks| / 2)."""
    usable = [
        q
        for q in (polymarket, kalshi)
        if q is not None and q.age_seconds <= max_age_s and q.mid is not None and q.spread is not None and q.spread >= 0
    ]
    half_tightest = min(q.spread for q in usable) / 2 if usable else 0.0
    disagreement = abs(usable[0].mid - usable[1].mid) / 2 if len(usable) == 2 else 0.0
    return max(PRICE_TICK, half_tightest, disagreement)


def favourite_side(fair_yes: float) -> tuple[Side, float]:
    """Quote the favourite only: fair YES >= 0.5 -> buy YES, else buy NO. Returns (side, fair in that side's terms)."""
    return ("yes", fair_yes) if fair_yes >= 0.5 else ("no", 1.0 - fair_yes)


@dataclass(frozen=True)
class Level:
    index: int
    offset: float
    price: float  # in the quoted side's terms
    quantity: int


def ladder_levels(
    fair_side: float,
    u: float,
    offsets: tuple[float, ...] = LADDER_OFFSETS,
    sizes: tuple[int, ...] = LADDER_SIZES,
    max_cost: float = LADDER_MAX_COST_PER_MARKET,
    min_extra: float = LADDER_MIN_EXTRA_DISTANCE,
) -> list[Level]:
    """Level price = fair_side - offset, rounded DOWN to the tick; skipped below 0.005 or closer
    to fair than u + min_extra. Sizes are trimmed so the market's total cost stays <= max_cost."""
    levels: list[Level] = []
    cost = 0.0
    for index, (offset, size) in enumerate(zip(offsets, sizes)):
        price = floor_tick(fair_side - offset)
        if price < MIN_LIMIT_PRICE - _EPS:
            continue
        if fair_side - price < u + min_extra - _EPS:
            continue
        quantity = min(size, math.floor((max_cost - cost) / price + _EPS))
        if quantity <= 0:
            continue
        levels.append(Level(index, offset, price, quantity))
        cost += quantity * price
    return levels


def classify_exit(fair_entry: float, fair_now: float, fill_price: float, u: float) -> Literal["information", "noise", "drift"]:
    """All in the held side's terms. information: fair moved toward the fill by more than half
    the entry distance. noise: fair moved by less than u. drift: anything else (treated like
    noise, priced off the new fair)."""
    move = fair_now - fair_entry
    distance = fair_entry - fill_price
    if move < 0 and -move > distance / 2 + _EPS:
        return "information"
    if abs(move) < u - _EPS:
        return "noise"
    return "drift"


def noise_exit_price(fair_now: float, u: float, anchor: float, steps: int) -> float:
    """Passive exit: start one tick under the best ask seen when the exit began (`anchor` is
    that start price), one tick lower per elapsed step, never below fair - u."""
    floor = min(ceil_tick(fair_now - u), MAX_LIMIT_PRICE)
    return round(max(floor, anchor - steps * PRICE_TICK, MIN_LIMIT_PRICE), 3)


def breaker_tripped(fills: Iterable[tuple[float, str]], now: float, window_s: float = LADDER_BREAKER_WINDOW_S, min_races: int = LADDER_BREAKER_RACES) -> bool:
    """fills: (unix time, race key) of recent ladder buy fills."""
    return len({race for t, race in fills if now - t <= window_s}) >= min_races


def hole_counts(path: Path = HOLES_CSV_PATH) -> Counter[str]:
    """Holes per exchange_id from edge_scan.py's holes.csv; empty if the file is missing."""
    if not path.exists():
        return Counter()
    with path.open(encoding="utf-8") as fh:
        return Counter(row["exchange_id"] for row in csv.DictReader(fh) if row.get("exchange_id"))


def choose_markets(priced: Iterable[str], holes: Counter[str], max_markets: int = LADDER_MAX_MARKETS) -> list[str]:
    """Priced markets ranked by hole count; without hole data, all priced markets. Capped."""
    priced = sorted(set(priced))
    with_holes = [eid for eid in priced if holes.get(eid)]
    ranked = sorted(with_holes, key=lambda eid: (-holes[eid], eid)) if with_holes else priced
    return ranked[:max_markets]


# ---------------------------------------------------------------------- state


@dataclass
class LadderOrder:
    """A resting ladder BUY (or, in DRY_RUN, a virtual one) we placed."""

    order_id: str
    exchange_id: str
    side: Side
    price: float
    quantity: int  # remaining
    level: int
    fair_side: float
    u: float
    expires_at: float


@dataclass
class LadderPosition:
    """Ladder inventory on one exchange, from the first ladder fill until it is sold back."""

    exchange_id: str
    side: Side
    bought: int = 0
    cost: float = 0.0
    sold: int = 0
    proceeds: float = 0.0
    fair_entry: float = 0.0
    last_fill_price: float = 0.0
    u: float = PRICE_TICK
    mode: str | None = None  # None (to classify) | "noise" | "information" | "hold"
    exit_started_at: float | None = None
    exit_anchor: float | None = None
    exit_order_id: str | None = None
    exit_price: float | None = None

    @property
    def shares(self) -> int:
        return self.bought - self.sold

    @property
    def open_cost(self) -> float:
        return self.cost * self.shares / self.bought if self.bought else 0.0


@dataclass
class LadderContext:
    """What one strategy cycle hands the ladder."""

    now: float  # unix seconds
    bankroll: float
    tournament_id: str | None
    tournament_slug: str
    market_map: MarketMap
    fair: Mapping[str, float]  # exchange_id -> fair P(YES), fresh this cycle
    quotes: Mapping[str, tuple[OracleQuote | None, OracleQuote | None]]  # exchange_id -> (pm, ks)
    books: Mapping[str, Any]  # exchange_id -> SigBook (yes_bid / yes_ask / buy_price / sell_price)
    holdings: Mapping[str, Any]  # exchange_id -> Holding (side / shares / cost_basis), ALL positions
    open_orders: list[dict[str, Any]]
    limits: RiskLimits
    max_quote_age_s: float
    # The other books this cycle - taker (hold-book cost + buys sent this cycle) and quoter
    # (resting quotes + inventory) - for the global gross cap and the directional cap.
    other_gross: float = 0.0
    other_directional: Mapping[Direction, float] = field(default_factory=dict)
    other_race_legs: Mapping[str, list[RaceLeg]] = field(default_factory=dict)  # resting quotes
    exclude_markets: frozenset[str] = frozenset()  # the quoter's markets in use
    # This cycle's new fills from strategy.py's FillFeed (oldest first); None: read them here.
    fills: list[dict[str, Any]] | None = None


@dataclass
class LadderReport:
    markets: int = 0
    placed: int = 0
    cancels: int = 0
    fills: int = 0
    exits_placed: int = 0
    rounds_closed: int = 0
    realized_pnl: float = 0.0
    realized_pnl_total: float = 0.0
    paused: bool = False
    skipped: str | None = None
    blocked_global: int = 0  # levels skipped: over the global gross cap
    blocked_direction: int = 0  # levels skipped: over the directional cap (all books)


class LadderJournal:
    """Append-only JSONL: one line per ladder intent, placement, fill, exit and closed round."""

    def __init__(self, path: Path = LADDER_JOURNAL_PATH) -> None:
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def replay(self) -> tuple[dict[str, LadderPosition], float]:
        """Open ladder inventory and all-time realized P&L, rebuilt after a restart.

        Journalled prices are in the position's own side's terms (fills.fill_price_side).
        """
        positions: dict[str, LadderPosition] = {}
        realized = 0.0
        if not self.path.exists():
            return positions, realized
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            eid = entry.get("exchange_id")
            if entry["event"] == "fill":
                pos = positions.setdefault(eid, LadderPosition(eid, entry["side"], fair_entry=entry.get("fair") or 0.0, u=entry.get("u") or PRICE_TICK))
                pos.bought += int(entry["quantity"])
                pos.cost += entry["quantity"] * entry["price"]
                pos.last_fill_price = entry["price"]
            elif entry["event"] == "exit_fill" and eid in positions:
                positions[eid].sold += int(entry["quantity"])
                positions[eid].proceeds += entry["quantity"] * entry["price"]
            elif entry["event"] == "round_closed":
                positions.pop(eid, None)
                realized += entry.get("pnl", 0.0)
        return {eid: p for eid, p in positions.items() if p.shares > 0}, realized


# ---------------------------------------------------------------------- manager


FairFn = Callable[[OracleQuote | None, OracleQuote | None, float], tuple[Any, str]]


class LadderManager:
    def __init__(
        self,
        exchange: ExchangeClient,
        polymarket: Any,
        kalshi: Any,
        fair_fn: FairFn,
        journal: LadderJournal | None = None,
        holes_path: Path = HOLES_CSV_PATH,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._exchange = exchange
        self._polymarket = polymarket
        self._kalshi = kalshi
        self._fair_fn = fair_fn  # strategy.compute_fair_price
        self._journal = journal or LadderJournal()
        self._clock = clock
        self._holes = hole_counts(holes_path)
        self.orders: dict[str, LadderOrder] = {}  # order_id -> resting ladder buy
        self.exit_orders: dict[str, str] = {}  # order_id -> exchange_id
        # Every ladder buy / exit order id placed by this process: a fill can arrive after
        # the order already left `orders` (fully filled before the next open-order read).
        self._order_meta: dict[str, LadderOrder] = {}
        self._exit_ids: set[str] = set()
        self.positions, self.realized_total = self._journal.replay()
        self._blocked_until: dict[str, float] = {}
        self._paused_until = 0.0
        self._recent_fills: list[tuple[float, str]] = []  # (time, race) for the breaker
        self._fills_since = clock()  # fills before this were made before this process
        self._seen_fill_ids: set[Any] = set()
        self._virtual_ids = 0
        logger.info(
            "Ladder enabled",
            extra={"markets_with_holes": len(self._holes), "open_positions": len(self.positions), "realized_total": round(self.realized_total, 2)},
        )

    # ------------------------------------------------------------ views used by strategy.py

    def order_ids(self) -> set[str]:
        return set(self.orders) | set(self.exit_orders)

    def order_action(self, order_id: str) -> str | None:
        """"buy" / "sell" for a ladder buy / exit placed by this process, else None."""
        if order_id in self._order_meta:
            return "buy"
        if order_id in self._exit_ids:
            return "sell"
        return None

    def hold_book(self, holdings: Mapping[str, Any]) -> dict[str, Any]:
        """Holdings minus ladder inventory: what the taker strategy (and its gross cap) sees."""
        result = dict(holdings)
        for eid, pos in self.positions.items():
            held = result.get(eid)
            if held is None or held.side != pos.side or pos.shares <= 0:
                continue
            remaining = held.shares - pos.shares
            if remaining <= 0:
                del result[eid]
            else:
                scale = remaining / held.shares
                result[eid] = type(held)(held.side, remaining, held.market_value * scale, held.cost_basis * scale)
        return result

    def resting_race_legs(self, market_map: MarketMap) -> dict[str, list[RaceLeg]]:
        """Resting ladder buys as if filled, grouped by race (for every race worst-case check)."""
        legs: dict[str, list[RaceLeg]] = {}
        for order in self.orders.values():
            mapping = market_map.get(order.exchange_id)
            if mapping is not None and order.quantity > 0:
                legs.setdefault(mapping.race.key, []).append(RaceLeg(mapping.party, order.side, order.quantity, order.quantity * order.price))
        return legs

    def markets_in_use(self) -> frozenset[str]:
        """Markets with resting ladder orders, exits or ladder inventory - off limits to the quoter."""
        return frozenset({o.exchange_id for o in self.orders.values()} | set(self.exit_orders.values())) | {
            eid for eid, p in self.positions.items() if p.shares > 0
        }

    def ladder_cost(self) -> float:
        """Cost charged to LADDER_BOOK_FRAC: resting ladder buys + open ladder inventory."""
        return sum(o.quantity * o.price for o in self.orders.values()) + sum(p.open_cost for p in self.positions.values())

    def book_cost(self) -> float:
        """The ladder's book for the global gross cap (same as ladder_cost)."""
        return self.ladder_cost()

    def directional_cost(self, market_map: MarketMap) -> dict[Direction, float]:
        """ladder_cost() split by direction (R / D / neutral), favourite side as bought."""
        return directional_exposure(
            market_map,
            [(o.exchange_id, o.side, o.quantity * o.price) for o in self.orders.values()]
            + [(p.exchange_id, p.side, p.open_cost) for p in self.positions.values()],
        )

    # ------------------------------------------------------------ the cycle

    async def run_cycle(self, ctx: LadderContext) -> LadderReport:
        report = LadderReport(realized_pnl_total=round(self.realized_total, 2))
        if not ctx.tournament_id:
            report.skipped = "no tournament id"
            logger.warning("Ladder skipped: tournament id unknown")
            return report
        self._sync_resting(ctx.open_orders)
        await self._read_fills(ctx, report)
        if breaker_tripped(self._recent_fills, ctx.now) and ctx.now >= self._paused_until:
            await self._trip_breaker(ctx, report)
        await self._classify_new_fills(ctx, report)
        report.paused = ctx.now < self._paused_until
        if not report.paused:
            await self._quote(ctx, report)
        await self._manage_exits(ctx, report)
        report.realized_pnl_total = round(self.realized_total, 2)
        return report

    def _sync_resting(self, open_orders: list[dict[str, Any]]) -> None:
        if self._exchange.dry_run:
            return  # virtual orders never appear in the open-order read
        open_by_id = {str(o.get("id")): o for o in open_orders}
        for order_id in list(self.orders):
            live = open_by_id.get(order_id)
            if live is None:
                del self.orders[order_id]  # filled, expired or cancelled; fills come from the fills read
            else:
                self.orders[order_id].quantity = int(abs(live.get("quantity") or 0))  # NO orders report negative remaining
        for order_id in list(self.exit_orders):
            if order_id not in open_by_id:
                eid = self.exit_orders.pop(order_id)
                pos = self.positions.get(eid)
                if pos is not None and pos.exit_order_id == order_id:
                    pos.exit_order_id, pos.exit_price = None, None

    async def _read_fills(self, ctx: LadderContext, report: LadderReport) -> None:
        if self._exchange.dry_run:
            return
        if ctx.fills is not None:
            oldest_first = list(ctx.fills)  # already read this cycle by strategy.py's FillFeed
        else:
            newest_first: list[dict[str, Any]] = []
            cursor: str | None = None
            while True:
                page = await self._exchange.get_tournament_fills(ctx.tournament_slug, cursor=cursor)
                data = page.get("data") or []
                newest_first.extend(data)
                cursor = (page.get("pagination") or {}).get("nextCursor")
                reached_old = any(_unix(f.get("filledAt")) < self._fills_since for f in data)
                if reached_old or not cursor or not (page.get("pagination") or {}).get("hasMore"):
                    break
            oldest_first = list(reversed(newest_first))
        for fill in oldest_first:
            if fill.get("id") in self._seen_fill_ids or _unix(fill.get("filledAt")) < self._fills_since:
                continue
            self._seen_fill_ids.add(fill.get("id"))
            order_id = str(fill.get("orderId"))
            if order_id in self._order_meta:
                self._on_ladder_fill(ctx, fill, report)
            elif order_id in self._exit_ids:
                self._on_exit_fill(fill, report)

    def _on_ladder_fill(self, ctx: LadderContext, fill: dict[str, Any], report: LadderReport) -> None:
        order = self._order_meta.get(str(fill.get("orderId")))
        side: Side = fill.get("side") or (order.side if order else "yes")
        price = fill_price_side(fill, "rest", side)  # REST: a NO fill carries a NO price
        quantity = int(fill_shares(fill))
        if price is None or quantity <= 0:
            return
        eid = str(fill.get("exchangeId"))
        pos = self.positions.get(eid)
        if pos is None:
            pos = self.positions[eid] = LadderPosition(eid, side, fair_entry=order.fair_side if order else price, u=order.u if order else PRICE_TICK)
        pos.bought += quantity
        pos.cost += quantity * price
        pos.last_fill_price = price
        pos.mode = None  # (re)classify with fresh oracles
        report.fills += 1
        mapping = ctx.market_map.get(eid)
        self._recent_fills.append((_unix(fill.get("filledAt")) or ctx.now, mapping.race.key if mapping else eid))
        self._recent_fills = [f for f in self._recent_fills if ctx.now - f[0] <= LADDER_BREAKER_WINDOW_S]
        self._journal.write(
            "fill", exchange_id=eid, side=side, price=price, quantity=quantity, fair=pos.fair_entry, u=pos.u,
            level=order.level if order else None, order_id=fill.get("orderId"), fill_id=fill.get("id"),
        )

    def _on_exit_fill(self, fill: dict[str, Any], report: LadderReport) -> None:
        eid = str(fill.get("exchangeId"))
        pos = self.positions.get(eid)
        if pos is None:
            return
        price = fill_price_side(fill, "rest", pos.side)
        quantity = int(fill_shares(fill))
        if price is None or quantity <= 0:
            return
        pos.sold += quantity
        pos.proceeds += quantity * price
        self._journal.write("exit_fill", exchange_id=eid, side=pos.side, price=price, quantity=quantity, fair=pos.fair_entry, u=pos.u, level=None)
        if pos.shares <= 0:
            pnl = round(pos.proceeds - pos.cost, 4)
            self.realized_total += pnl
            report.rounds_closed += 1
            report.realized_pnl = round(report.realized_pnl + pnl, 4)
            self._journal.write("round_closed", exchange_id=eid, side=pos.side, bought=pos.bought, cost=round(pos.cost, 4), proceeds=round(pos.proceeds, 4), pnl=pnl)
            del self.positions[eid]

    async def _trip_breaker(self, ctx: LadderContext, report: LadderReport) -> None:
        races = sorted({race for t, race in self._recent_fills if ctx.now - t <= LADDER_BREAKER_WINDOW_S})
        logger.error("Ladder circuit breaker: fills in many races at once, cancelling all ladder orders", extra={"races": races, "pause_s": LADDER_BREAKER_PAUSE_S})
        self._journal.write("breaker", races=races, pause_s=LADDER_BREAKER_PAUSE_S)
        # One tournament-wide cancel-all (1 write) rather than one per market: also cancels
        # the taker's short-lived resting orders, which is harmless.
        await self._exchange.cancel_all_orders(tournament_id=ctx.tournament_id)
        report.cancels += 1
        self.orders.clear()
        self._paused_until = ctx.now + LADDER_BREAKER_PAUSE_S

    async def _classify_new_fills(self, ctx: LadderContext, report: LadderReport) -> None:
        for eid, pos in self.positions.items():
            if pos.mode not in (None, "hold"):
                continue
            mapping = ctx.market_map.get(eid)
            if mapping is None:
                continue
            pm = ks = None
            if pos.mode is None:  # a fresh fill: refetch both oracles for this market now
                if mapping.polymarket_token:
                    pm = (await self._polymarket.get_quotes([mapping.polymarket_token])).get(mapping.polymarket_token)
                if mapping.kalshi_ticker:
                    ks = (await self._kalshi.get_quotes([mapping.kalshi_ticker])).get(mapping.kalshi_ticker)
            else:  # holding: re-check with this cycle's quotes
                pm, ks = ctx.quotes.get(eid, (None, None))
            fair, _ = self._fair_fn(pm, ks, ctx.max_quote_age_s)
            if fair is None:
                if pos.mode is None:
                    logger.warning("Ladder fill: both oracles stale, holding", extra={"exchange_id": eid})
                pos.mode = "hold"
                continue
            fair_side = fair.prob if pos.side == "yes" else 1.0 - fair.prob
            u = uncertainty_band(pm, ks, ctx.max_quote_age_s)
            kind = classify_exit(pos.fair_entry, fair_side, pos.last_fill_price, u)
            pos.u = u
            if kind == "information":
                pos.mode = "information"
                self._blocked_until[eid] = ctx.now + LADDER_INFO_BLOCK_S
                await self._cancel_market(ctx, eid, report)  # stop catching more of a falling knife
            else:
                pos.mode = "noise"
                pos.exit_started_at = ctx.now
                book = ctx.books.get(eid)
                ask = book.buy_price(pos.side) if book is not None else None  # best ask in side terms
                pos.exit_anchor = floor_tick(ask - PRICE_TICK) if ask else floor_tick(fair_side)
            self._journal.write("exit_intent", exchange_id=eid, side=pos.side, price=None, quantity=pos.shares, fair=round(fair_side, 4), u=round(u, 4), level=None, mode=pos.mode, kind=kind)

    async def _cancel_market(self, ctx: LadderContext, eid: str, report: LadderReport) -> bool:
        """Scoped cancel-all; True only when it came back clean (nothing left uncancelled)."""
        result = await self._exchange.cancel_all_orders(exchange_id=eid, tournament_id=ctx.tournament_id)
        report.cancels += 1
        clean = result is None or not result.get("errors")  # None = DRY_RUN
        if clean:
            for order_id in [oid for oid, o in self.orders.items() if o.exchange_id == eid]:
                del self.orders[order_id]
        return clean

    async def _quote(self, ctx: LadderContext, report: LadderReport) -> None:
        candidates = {eid: fair for eid, fair in ctx.fair.items() if eid not in ctx.exclude_markets}
        markets = [eid for eid in choose_markets(candidates, self._holes) if ctx.market_map.get(eid) is not None]
        report.markets = len(markets)

        # Pass 1 - requotes: a scoped cancel-all, and the market is re-posted below only if
        # that cancel came back clean (200). Untouched markets keep their resting levels.
        to_fill: list[str] = []
        for eid in markets:
            pos = self.positions.get(eid)
            if ctx.now < self._blocked_until.get(eid, 0.0) or (pos is not None and pos.mode == "hold"):
                continue  # information block, or holding on stale oracles: add nothing
            side, fair_side = favourite_side(ctx.fair[eid])
            resting = [o for o in self.orders.values() if o.exchange_id == eid]
            if resting:
                moved = max(abs(fair_side - o.fair_side) for o in resting)
                near_expiry = min(o.expires_at for o in resting) - ctx.now < LADDER_EXPIRY_REQUOTE_S
                if moved <= LADDER_REQUOTE_MOVE and not near_expiry:
                    continue
                self._journal.write("requote", exchange_id=eid, side=side, price=None, quantity=None, fair=round(fair_side, 4), u=None, level=None, moved=round(moved, 4), near_expiry=near_expiry)
                if not await self._cancel_market(ctx, eid, report):
                    continue
            to_fill.append(eid)

        # Pass 2 - sizing from the state after the cancels: LADDER_BOOK_FRAC budget, the
        # global gross cap and the directional cap (taker book + ladder), and the race
        # worst case incl. ALL holdings and every resting ladder order as if filled.
        race_legs = race_legs_from_holdings(ctx.market_map, ctx.holdings)
        for source in (ctx.other_race_legs, self.resting_race_legs(ctx.market_map)):
            for race, legs in source.items():
                race_legs.setdefault(race, []).extend(legs)
        race_parties = {key: [m.party for m in group] for key, group in ctx.market_map.by_race().items()}
        race_cap = ctx.limits.max_race_worst_loss_fraction * ctx.bankroll
        budget = LADDER_BOOK_FRAC * ctx.bankroll - self.ladder_cost()
        total_room = ctx.limits.max_total_gross_exposure_fraction * ctx.bankroll - ctx.other_gross - self.ladder_cost()
        direction_cap = ctx.limits.max_directional_fraction * ctx.bankroll
        direction_used: dict[Direction, float] = dict(ctx.other_directional)
        for direction, cost in self.directional_cost(ctx.market_map).items():
            direction_used[direction] = direction_used.get(direction, 0.0) + cost

        to_place: list[tuple[OrderRequest, dict[str, Any]]] = []
        for eid in to_fill:
            side, fair_side = favourite_side(ctx.fair[eid])
            pm, ks = ctx.quotes.get(eid, (None, None))
            u = uncertainty_band(pm, ks, ctx.max_quote_age_s)
            mapping = ctx.market_map.get(eid)
            legs = race_legs.setdefault(mapping.race.key, [])
            parties = race_parties.get(mapping.race.key, [mapping.party])
            direction = direction_of(ctx.market_map, eid, side)
            for level in ladder_levels(fair_side, u):
                quantity = min(level.quantity, math.floor(max(budget, 0.0) / level.price + _EPS))
                # A level that does not fit whole under the global gross cap or the directional
                # cap (taker + ladder) is skipped, never shrunk into the last bit of room.
                if quantity > 0 and quantity * level.price > total_room + _EPS:
                    report.blocked_global += 1
                    continue
                if quantity > 0 and direction is not Direction.NEUTRAL and direction_used.get(direction, 0.0) + quantity * level.price > direction_cap + _EPS:
                    report.blocked_direction += 1
                    continue
                quantity = min(quantity, max_shares_within_race_cap(parties, legs, mapping.party, side, level.price, quantity, race_cap))
                if quantity <= 0:
                    continue
                cost = quantity * level.price
                legs.append(RaceLeg(mapping.party, side, quantity, cost))
                budget -= cost
                total_room -= cost
                direction_used[direction] = direction_used.get(direction, 0.0) + cost
                meta = {"exchange_id": eid, "side": side, "price": level.price, "quantity": quantity, "fair": round(fair_side, 4), "u": round(u, 4), "level": level.index}
                self._journal.write("intent", **meta)
                to_place.append((self._buy(eid, side, level.price, quantity, ctx), meta))
        report.placed += await self._send_batches(ctx, to_place)

    def _buy(self, eid: str, side: Side, price: float, quantity: int, ctx: LadderContext) -> OrderRequest:
        expires = datetime.fromtimestamp(ctx.now + LADDER_ORDER_TTL_S, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return OrderRequest(exchange_id=eid, side=side, action="buy", quantity=quantity, price=price, tournament_id=ctx.tournament_id, expiration_date=expires)

    async def _send_batches(self, ctx: LadderContext, items: list[tuple[OrderRequest, dict[str, Any]]], exit_eid: str | None = None) -> int:
        """Batches of <= 50, one key each; on 207 the failed items once more under a new key."""
        placed = 0
        for start in range(0, len(items), MAX_BATCH_ORDERS):
            chunk = items[start : start + MAX_BATCH_ORDERS]
            for attempt in range(2):
                key = new_idempotency_key("ladder")
                results = await self._exchange.place_batch([order for order, _ in chunk], key, write_rank=WRITE_RANK_LADDER)
                if results is None:  # DRY_RUN (or halted): remember virtual orders, place nothing
                    if self._exchange.dry_run and not self._exchange.halted:
                        for order, meta in chunk:
                            self._virtual_ids += 1
                            self._register(f"dry-{self._virtual_ids}", order, meta, ctx, exit_eid)
                    break
                failed: list[tuple[OrderRequest, dict[str, Any]]] = []
                for (order, meta), result in zip(chunk, results):
                    if result.get("ok"):
                        data = result.get("data") or {}
                        self._register(str(data.get("orderId")), order, meta, ctx, exit_eid, data)
                        placed += 1
                    elif result.get("status") != 403:  # terms problems are not retried
                        failed.append((order, meta))
                if not failed or attempt == 1:
                    for order, meta in failed:
                        self._journal.write("placement_failed", **meta)
                    break
                chunk = failed
        return placed

    def _register(self, order_id: str, order: OrderRequest, meta: dict[str, Any], ctx: LadderContext, exit_eid: str | None, data: dict | None = None) -> None:
        self._journal.write("placement" if exit_eid is None else "exit_placement", order_id=order_id, **meta)
        if exit_eid is not None:
            self.exit_orders[order_id] = exit_eid
            self._exit_ids.add(order_id)
            pos = self.positions.get(exit_eid)
            if pos is not None:
                pos.exit_order_id, pos.exit_price = order_id, order.price
            return
        remaining = int(abs((data or {}).get("remainingQuantity", order.quantity) or 0)) if data else order.quantity
        ladder_order = LadderOrder(order_id, order.exchange_id, order.side, order.price, remaining, meta["level"], meta["fair"], meta["u"], ctx.now + LADDER_ORDER_TTL_S)
        self._order_meta[order_id] = ladder_order
        if remaining > 0:
            self.orders[order_id] = ladder_order

    async def _manage_exits(self, ctx: LadderContext, report: LadderReport) -> None:
        for eid, pos in list(self.positions.items()):
            if pos.shares <= 0 or pos.mode in (None, "hold"):
                continue
            # Never sell more than is really held: an oversized SELL is rewritten by the
            # engine into the complement BUY (sell yes q@p = buy no q@(1 - p)).
            held = ctx.holdings.get(eid)
            sellable = min(pos.shares, held.shares) if held is not None and held.side == pos.side else 0
            if sellable <= 0:
                continue
            book = ctx.books.get(eid)
            ttl = LADDER_ORDER_TTL_S
            if pos.mode == "information":
                price = book.sell_price(pos.side) if book is not None else None  # best bid, side terms
                if price is None or pos.exit_order_id is not None:
                    continue
                ttl = INFO_EXIT_TTL_S  # marketable; an unfilled rest is re-posted at the new bid
            else:
                fair_yes = ctx.fair.get(eid)
                if fair_yes is None:
                    continue  # no fresh fair this cycle: keep the current exit order
                fair_side = fair_yes if pos.side == "yes" else 1.0 - fair_yes
                steps = int((ctx.now - (pos.exit_started_at or ctx.now)) // LADDER_EXIT_STEP_S)
                price = noise_exit_price(fair_side, pos.u, pos.exit_anchor or floor_tick(fair_side), steps)
                if pos.exit_order_id is not None:
                    if pos.exit_price == price:
                        continue
                    if not pos.exit_order_id.startswith("dry-"):
                        await self._exchange.cancel_order(pos.exit_order_id)
                        report.cancels += 1
                    self.exit_orders.pop(pos.exit_order_id, None)
                    pos.exit_order_id = None
            expires = datetime.fromtimestamp(ctx.now + ttl, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            order = OrderRequest(exchange_id=eid, side=pos.side, action="sell", quantity=sellable, price=price, tournament_id=ctx.tournament_id, expiration_date=expires)
            meta = {"exchange_id": eid, "side": pos.side, "price": price, "quantity": sellable, "fair": round(pos.fair_entry, 4), "u": round(pos.u, 4), "level": None, "mode": pos.mode}
            self._journal.write("exit_intent", **meta)
            report.exits_placed += await self._send_batches(ctx, [(order, meta)], exit_eid=eid)


def _unix(raw: str | None) -> float:
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
