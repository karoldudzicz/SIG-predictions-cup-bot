"""Trading logic: one `run_cycle()` = read state -> price -> size -> plan -> execute.

Strategy #2 (cross-market oracle arb) is live here. Strategy #1 (SIG relationship
constraints) is OBSERVE-ONLY for now: SIG has published no relationships yet, so
the payload shape of real violations is unverified - we log them instead of
trading blind. Market making (#3) is not implemented.

Cycle, in order:
  1. STATE    - account value, positions and open orders from the API (source of truth)
  2. PRICES   - SIG top of book, then oracle quotes (Polymarket, Kalshi) last, so they
                are as fresh as possible for the staleness check
  3. FAIR     - fair P(YES) from fresh, two-sided oracle quotes; no fair price -> skip market
                (reason counted: missing / stale / one-sided-or-crossed / too wide)
  4. SIZE     - risk.size_portfolio: Kelly per cluster, THEN every risk limit
                (per contract, per cluster, per direction, portfolio) - see risk.py
  5. PLAN     - diff targets vs holdings -> explicit INCREASE (buy) / REDUCE (sell) orders,
                shrink each to the liquidity resting in its SIG order book, then the
                exposure gates (apply_exposure_caps): race worst-case loss, gross
                exposure, new notional per cycle
  6. EXECUTE  - journal the idempotencyKey, place a marketable limit order, journal the outcome

Orders are marketable limits (buy at the ask / sell at the bid) with a short expiry,
so an unfilled remainder never rests stale for long. Exchanges with an open order
are skipped until it fills or expires, so orders never stack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import aiohttp

from config import (
    BOOK_CACHE_MAX_AGE_S,
    BOOK_READ_WINDOW_S,
    LADDER_ENABLED,
    MARKOUT_ENABLED,
    MIN_CYCLE_GAP_S,
    QUOTER_ENABLED,
    RELATIONSHIPS_CHECK_ENABLED,
    SIG_READS_PER_MIN,
    MIN_ORACLE_MARGIN,
    ORDER_MAX_QUOTE_AGE_S,
    SELL_ABOVE_TARGET,
    UNKNOWN_ORDER_BLOCK_S,
    Settings,
)
from exchange_api import (
    ApiError,
    AuthError,
    ExchangeClient,
    READ_BOOK,
    OrderRequest,
    ReadDeferredError,
    ReadPausedError,
    RequestTimeoutError,
    SelfCrossError,
    TermsNotAcknowledgedError,
    new_idempotency_key,
)
from ladder import LadderContext, LadderManager
from quoter import QuoterContext, QuoterManager
from markout import FillFeed, MarkoutTracker
from market_mapper import DATA_DIR, MarketMap
from oracle_api import KalshiClient, OracleError, OracleQuote, PolymarketClient
from risk import (
    Cluster,
    Direction,
    Opportunity,
    RaceLeg,
    RiskLimits,
    SizedBet,
    build_clusters,
    check_buy,
    directional_exposure,
    leaderboard_standing,
    max_shares_within_race_cap,
    race_legs_from_holdings,
    race_worst_case_loss,
    size_portfolio,
    tournament_kelly_fraction,
)

logger = logging.getLogger(__name__)

JOURNAL_PATH = DATA_DIR / "order_journal.jsonl"

# Oracle sanity: a quote wider than this has no meaningful mid; two venues further
# apart than this means bad data or a bad mapping - either way, do not trade on it.
MAX_ORACLE_SPREAD = 0.10
MAX_ORACLE_DISAGREEMENT = 0.10

MIN_ORDER_SHARES = 10  # smaller diffs are not worth an order
REBALANCE_BAND = 0.25  # only trade when holding differs from target by > 25% of target
ORDER_TTL_S = 60  # unfilled remainders expire instead of resting stale

LEADERBOARD_REFRESH_S = 300
RELATIONSHIPS_REFRESH_S = 300
# Bulk prices failed: at most this many per-market order books are read instead.
FALLBACK_MAX_BOOKS = 10

Side = Literal["yes", "no"]


# ---------------------------------------------------------------------- order journal


class OrderStatus(str, Enum):
    PENDING = "pending"  # written BEFORE sending; if it is the last state, the outcome is unknown
    ACCEPTED = "accepted"  # API confirmed; the canonical order is recorded
    REJECTED = "rejected"  # API refused terminally (4xx) - definitely not placed
    UNKNOWN = "unknown"  # retries exhausted / ORDER_STATUS_UNKNOWN - may or may not exist
    DRY_RUN = "dry_run"  # never sent
    BLOCKED = "blocked"  # never sent: kill switch engaged (shutdown in progress)
    RESOLVED = "resolved"  # PENDING/UNKNOWN later settled by reconciliation (main.py)


_SETTLED = {OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.DRY_RUN, OrderStatus.BLOCKED, OrderStatus.RESOLVED}


class OrderJournal:
    """Append-only JSONL log of every placement, keyed by idempotencyKey.

    Lifecycle: PENDING is fsync'ed before the request leaves, then exactly one of
    ACCEPTED / REJECTED / UNKNOWN once we know. An entry whose latest state is
    PENDING or UNKNOWN is unresolved: the order may or may not exist. main.py must
    settle those at startup (see `unresolved()` / `mark_resolved()`); resending the
    stored payload with the same key is always safe - the API never double-places.
    """

    def __init__(self, path: Path = JOURNAL_PATH) -> None:
        self._path = path
        self._latest: dict[str, dict[str, Any]] = {}
        self._order_actions: dict[str, str] = {}  # engine order id -> canonical action
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._latest[entry["key"]] = {**self._latest.get(entry["key"], {}), **entry}
                    self._index_order(entry)

    def _index_order(self, entry: dict[str, Any]) -> None:
        result = entry.get("result")
        if isinstance(result, dict) and result.get("orderId") is not None and result.get("action"):
            self._order_actions[str(result["orderId"])] = result["action"]

    def action_for_order(self, order_id: str) -> str | None:
        """Canonical "buy" / "sell" of an accepted taker order, from its engine order id."""
        return self._order_actions.get(str(order_id))

    def record(self, key: str, status: OrderStatus, **fields: Any) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "key": key, "status": status.value, **fields}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # must survive a crash right after this line
        self._latest[key] = {**self._latest.get(key, {}), **entry}
        self._index_order(entry)

    def status(self, key: str) -> OrderStatus | None:
        entry = self._latest.get(key)
        return OrderStatus(entry["status"]) if entry else None

    def is_settled(self, key: str) -> bool:
        """True once we know for sure whether the order exists (or it was never sent)."""
        return self.status(key) in _SETTLED

    def unresolved(self) -> list[dict[str, Any]]:
        """Entries whose order may or may not exist, each with the original `payload`."""
        return [e for e in self._latest.values() if OrderStatus(e["status"]) not in _SETTLED]

    def mark_resolved(self, key: str, note: str) -> None:
        self.record(key, OrderStatus.RESOLVED, note=note)

    def recent_unknown(self, max_age_s: float, now: datetime | None = None) -> dict[tuple[str, str], dict[str, Any]]:
        """(exchange_id, side) -> {key, age_s} for UNKNOWN entries younger than max_age_s.

        Such an order may still be accepted late by the engine; a new order on the same
        market and side could double it. Multi-leg payloads (no top-level exchangeId)
        are not covered.
        """
        now = now or datetime.now(timezone.utc)
        blocked: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in self._latest.values():
            if entry.get("status") != OrderStatus.UNKNOWN.value:
                continue
            payload = entry.get("payload") or {}
            exchange_id, side = payload.get("exchangeId"), payload.get("side")
            try:
                age = (now - datetime.fromisoformat(entry["ts"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                continue
            if exchange_id and side and age <= max_age_s:
                blocked[(str(exchange_id), side)] = {"key": entry["key"], "age_s": round(age, 1)}
        return blocked


# ---------------------------------------------------------------------- pricing helpers


@dataclass(frozen=True)
class FairPrice:
    prob: float  # fair P(YES)
    sources: tuple[str, ...]
    half_spread: float = 0.0  # widest half bid-ask spread among the quotes used
    fetched_at: float | None = None  # unix time of the OLDEST quote used

    @property
    def margin(self) -> float:
        """Uncertainty haircut for sizing: max(MIN_ORACLE_MARGIN, half the oracle spread)."""
        return max(MIN_ORACLE_MARGIN, self.half_spread)


# Why a market got no fair price - one per market, counted in CycleReport.no_fair_reasons.
REASON_MISSING = "missing price"
REASON_STALE = "stale quote"
REASON_ONE_SIDED = "one-sided/crossed book"
REASON_WIDE = f"spread > {MAX_ORACLE_SPREAD:.2f}"
# When neither oracle is usable, report the most actionable problem: staleness is the
# only one the bot itself can fix (fetch timing); a missing mapping is the least informative.
_REASON_PRIORITY = (REASON_STALE, REASON_WIDE, REASON_ONE_SIDED, REASON_MISSING)


def _quote_problem(quote: OracleQuote | None, max_age_s: float) -> str | None:
    """Why one oracle quote is unusable, or None if it is usable."""
    if quote is None:
        return REASON_MISSING
    if quote.age_seconds > max_age_s:
        return REASON_STALE
    if quote.mid is None or quote.spread is None or quote.spread < 0:
        return REASON_ONE_SIDED
    if quote.spread > MAX_ORACLE_SPREAD:
        return REASON_WIDE
    return None


def compute_fair_price(
    polymarket: OracleQuote | None, kalshi: OracleQuote | None, max_age_s: float
) -> tuple[FairPrice | None, str]:
    """Average the mids of fresh, tight oracle quotes. Returns (None, reason) if unusable.

    Never falls back to the SIG price - that would be arbitraging against ourselves.
    A crossed oracle book (bid > ask) is treated like a one-sided one: its mid means nothing.
    """
    usable = []
    problems = []
    for quote in (polymarket, kalshi):
        problem = _quote_problem(quote, max_age_s)
        if problem is None:
            usable.append(quote)
        else:
            problems.append(problem)

    if not usable:
        return None, next(r for r in _REASON_PRIORITY if r in problems)
    if len(usable) == 2 and abs(usable[0].mid - usable[1].mid) > MAX_ORACLE_DISAGREEMENT:
        return None, f"oracles disagree ({usable[0].mid:.3f} vs {usable[1].mid:.3f})"
    prob = sum(q.mid for q in usable) / len(usable)
    half_spread = max(q.spread for q in usable) / 2
    fetched_at = min(q.fetched_at for q in usable)
    return FairPrice(prob, tuple(q.source for q in usable), half_spread, fetched_at), "ok"


@dataclass(frozen=True)
class SigBook:
    """SIG top of book, YES-normalised. Converts to the price paid/received per side."""

    yes_bid: float | None
    yes_ask: float | None
    fetched_at: float | None = None  # unix time the response carrying this top of book arrived
    last: float | None = None  # latest trade price (bulk read only); read by the virtual quoter's fill log

    def buy_price(self, side: Side) -> float | None:
        if side == "yes":
            return self.yes_ask
        return round(1.0 - self.yes_bid, 3) if self.yes_bid is not None else None

    def sell_price(self, side: Side) -> float | None:
        if side == "yes":
            return self.yes_bid
        return round(1.0 - self.yes_ask, 3) if self.yes_ask is not None else None


@dataclass(frozen=True)
class Holding:
    side: Side
    shares: int
    market_value: float
    cost_basis: float = 0.0  # cash paid for the open shares - what we lose if they settle at 0


def holdings_from_positions(positions: list[dict[str, Any]]) -> dict[str, Holding]:
    """/portfolio/positions: quantity > 0 = YES shares, < 0 = NO shares."""
    holdings = {}
    for position in positions:
        quantity = position.get("quantity") or 0
        if position.get("settled") or quantity == 0:
            continue
        holdings[position["exchangeId"]] = Holding(
            side="yes" if quantity > 0 else "no",
            shares=int(abs(quantity)),
            market_value=abs(position.get("marketValue") or 0.0),
            cost_basis=abs(position.get("costBasis") or 0.0),
        )
    return holdings


# ---------------------------------------------------------------------- planning


@dataclass(frozen=True)
class PlannedOrder:
    kind: Literal["increase", "reduce"]
    exchange_id: str
    side: Side
    quantity: int
    price: float
    reason: str
    # Filled in by apply_exposure_caps - for the "Order intent" log line.
    race_key: str = ""
    fair_side: float | None = None  # fair probability of the side being traded
    margin: float = 0.0
    edge: float | None = None  # buys: fair_side - margin - price; sells: price - fair_side
    race_loss_after: float | None = None
    gross_after: float | None = None
    quote_fetched_at: float | None = None  # oracle fetch time behind fair_side (unix seconds)

    @property
    def action(self) -> Literal["buy", "sell"]:
        # INCREASE buys more of `side`; REDUCE sells `side` we already hold. We never
        # sell more than we hold, so the engine's SELL canonicalization (flat/oversized
        # sell -> complement buy) should not trigger - but the response is still what
        # we record as the truth.
        return "buy" if self.kind == "increase" else "sell"


def quote_too_old(order: PlannedOrder, now: float | None = None, max_age_s: float = ORDER_MAX_QUOTE_AGE_S) -> bool:
    """True (and one INFO line) if the oracle snapshot behind this order is too old to send it.

    An order without a recorded fetch time is treated as too old: never trade on a fair
    price we cannot date.
    """
    now = time.time() if now is None else now
    age = None if order.quote_fetched_at is None else now - order.quote_fetched_at
    if age is not None and age <= max_age_s:
        return False
    logger.info(
        "Order skipped: quote too old",
        extra={
            "exchange_id": order.exchange_id,
            "kind": order.kind,
            "side": order.side,
            "quantity": order.quantity,
            "quote_age_s": None if age is None else round(age, 2),
            "max_age_s": max_age_s,
        },
    )
    return True


def _absorb_non_essential(read: str, exc: BaseException, fallback: str) -> None:
    """Swallow a transient failure of a NON-essential read (one WARNING), or re-raise.

    Auth / terms errors and programming errors are never swallowed.
    """
    if isinstance(exc, (AuthError, TermsNotAcknowledgedError)):
        raise exc
    if not isinstance(exc, (ApiError, OracleError, aiohttp.ClientError, TimeoutError)):
        raise exc
    logger.warning("Non-essential read failed, continuing", extra={"read": read, "error": repr(exc), "fallback": fallback})


def _older(oracle_time: float | None, book_time: Any) -> float | None:
    """The older of two fetch times; an undated oracle snapshot stays undated (too old)."""
    if oracle_time is None or not isinstance(book_time, (int, float)):
        return oracle_time
    return min(oracle_time, float(book_time))


def book_read_order(
    orders: list[PlannedOrder], fair: dict[str, float], margins: dict[str, float], deferred_before: set[str]
) -> list[str]:
    """Exchanges whose book is needed, in the order the read budget serves them: reduces first
    (they only free risk), then exchanges deferred last cycle, then by expected profit
    (quantity * net edge: fair - margin - price for a buy, price - fair for a sell)."""
    best: dict[str, tuple[int, int, float]] = {}
    for order in orders:
        fair_yes = fair.get(order.exchange_id)
        fair_side = None if fair_yes is None else (fair_yes if order.side == "yes" else 1.0 - fair_yes)
        if fair_side is None:
            edge = 0.0
        elif order.kind == "increase":
            edge = fair_side - margins.get(order.exchange_id, MIN_ORACLE_MARGIN) - order.price
        else:
            edge = order.price - fair_side
        key = (0 if order.kind == "reduce" else 1, 0 if order.exchange_id in deferred_before else 1, -order.quantity * edge)
        best[order.exchange_id] = min(best.get(order.exchange_id, key), key)
    return sorted(best, key=lambda eid: (*best[eid], eid))


def drop_blocked_by_unknown(
    planned: list[PlannedOrder], blocked: dict[tuple[str, str], dict[str, Any]]
) -> tuple[list[PlannedOrder], int]:
    """Remove orders on an (exchange_id, side) with a recent UNKNOWN order; one INFO line each."""
    kept: list[PlannedOrder] = []
    dropped = 0
    for order in planned:
        unknown = blocked.get((order.exchange_id, order.side))
        if unknown is None:
            kept.append(order)
            continue
        dropped += 1
        logger.info(
            "Order blocked: recent UNKNOWN order on same market and side",
            extra={
                "exchange_id": order.exchange_id,
                "side": order.side,
                "kind": order.kind,
                "quantity": order.quantity,
                "unknown_key": unknown["key"],
                "unknown_age_s": unknown["age_s"],
            },
        )
    return kept, dropped


def plan_orders(
    evaluated: dict[str, float],
    targets: dict[str, SizedBet],
    holdings: dict[str, Holding],
    books: dict[str, SigBook],
    skip: set[str],
    market_map: MarketMap | None = None,
    sell_above_target: bool = SELL_ABOVE_TARGET,
) -> list[PlannedOrder]:
    """Diff target positions against holdings for every exchange priced this cycle.

    evaluated: exchange_id -> fair P(YES), only exchanges with a usable fair price.
    sell_above_target: if False, a holding larger than its (cap-reduced) target is
    held and logged, never sold down. The other two sell paths are EV-driven, not
    cap-driven, and are unchanged: "flip" and "no edge" only sell at a bid at or
    above the held side's fair value.
    """
    race_legs = race_legs_from_holdings(market_map, holdings) if market_map is not None else {}
    planned: list[PlannedOrder] = []
    for exchange_id, fair_yes in evaluated.items():
        if exchange_id in skip:
            continue
        target = targets.get(exchange_id)
        held = holdings.get(exchange_id)
        book = books[exchange_id]

        if held and target and held.side != target.side:
            # Flip: close the wrong side first; the new side is bought next cycle,
            # after the close shows up in positions.
            price = book.sell_price(held.side)
            if price is not None:
                planned.append(PlannedOrder("reduce", exchange_id, held.side, held.shares, price, "flip: close opposite side"))
            continue

        if target:
            held_shares = held.shares if held else 0
            delta = target.quantity - held_shares
            threshold = max(MIN_ORDER_SHARES, REBALANCE_BAND * target.quantity)
            if delta >= threshold:
                planned.append(PlannedOrder("increase", exchange_id, target.side, delta, target.price, f"edge {target.edge:.3f}"))
            elif -delta >= threshold:
                if not sell_above_target:
                    mapping = market_map.get(exchange_id) if market_map is not None else None
                    parties = [m.party for m in market_map.by_race().get(mapping.race.key, [])] if mapping else []
                    logger.info(
                        "Position above target, holding",
                        extra={
                            "exchange_id": exchange_id,
                            "side": held.side,
                            "held_shares": held_shares,
                            "held_cost": round(held.cost_basis, 2),
                            "target_shares": target.quantity,
                            "target_stake": round(target.stake, 2),
                            "race": mapping.race.key if mapping else None,
                            "race_worst_loss": round(race_worst_case_loss(parties, race_legs[mapping.race.key]), 2) if mapping else None,
                        },
                    )
                    continue
                price = book.sell_price(target.side)
                if price is not None:
                    planned.append(PlannedOrder("reduce", exchange_id, target.side, -delta, price, "above target (risk limits / Kelly)"))
            continue

        if held:
            # No target: edge is gone. Selling below fair value gives away EV, so only
            # sell once the bid pays at least what the position is worth.
            fair_side = fair_yes if held.side == "yes" else 1.0 - fair_yes
            price = book.sell_price(held.side)
            if price is not None and price >= fair_side and held.shares >= MIN_ORDER_SHARES:
                planned.append(PlannedOrder("reduce", exchange_id, held.side, held.shares, price, f"no edge, bid {price} >= fair {fair_side:.3f}"))
    return planned


def _side_levels(orderbook: dict[str, Any], side: Side, action: Literal["buy", "sell"]) -> list[tuple[float, float]]:
    """(price, quantity) levels in `side`-relative prices, best first for that action.

    The SIG book is YES-normalised: buying YES lifts YES asks, buying NO hits YES bids
    (NO price = 1 - YES bid); selling YES hits YES bids, selling NO lifts YES asks.
    """
    asks = [(float(l["price"]), float(l["quantity"])) for l in orderbook.get("asks") or []]
    bids = [(float(l["price"]), float(l["quantity"])) for l in orderbook.get("bids") or []]
    if side == "no":
        asks, bids = [(round(1.0 - p, 3), q) for p, q in bids], [(round(1.0 - p, 3), q) for p, q in asks]
    if action == "buy":
        return sorted(asks)  # cheapest first
    return sorted(bids, reverse=True)  # highest first


def fit_to_book(
    order: PlannedOrder, orderbook: dict[str, Any], fair_yes: float, limits: RiskLimits, margin: float
) -> PlannedOrder | None:
    """Shrink an order to the liquidity actually resting in the book, so it fills now.

    INCREASE walks down the book while every level still passes the same price-aware
    buy filter as sizing (shrunk edge >= min_edge, high-price rule); the limit becomes
    the worst level used. REDUCE takes the best level only (never sells into a thinner,
    worse bid). Returns None if the fitted order is below MIN_ORDER_SHARES.
    """
    levels = _side_levels(orderbook, order.side, order.action)
    fair_side = fair_yes if order.side == "yes" else 1.0 - fair_yes
    remaining = order.quantity
    filled = 0
    limit: float | None = None
    for price, quantity in levels:
        if order.kind == "increase" and not check_buy(price, fair_side, margin, limits)[0]:
            break
        take = min(remaining, int(quantity))
        if take <= 0:
            break
        filled += take
        remaining -= take
        limit = price
        if order.kind == "reduce" or remaining == 0:
            break

    if limit is None or filled < MIN_ORDER_SHARES:
        return None
    if filled == order.quantity and limit == order.price:
        return order
    return replace(order, quantity=filled, price=limit, reason=f"{order.reason}; fitted to book {filled}/{order.quantity}")


def apply_exposure_caps(
    planned: list[PlannedOrder],
    holdings: dict[str, Holding],
    market_map: MarketMap,
    fair: dict[str, float],
    margins: dict[str, float],
    bankroll: float,
    limits: RiskLimits,
    race_holdings: dict[str, Holding] | None = None,
    extra_race_legs: dict[str, list[RaceLeg]] | None = None,
    books_gross: float = 0.0,
    log: bool = True,
) -> list[PlannedOrder]:
    """Last gate before execution: race worst-case loss, gross exposure, per-cycle notional.

    Works on running state, so every order accepted earlier in THIS cycle counts
    against the caps of the orders after it - not only positions read from the API.
    Reduces go first (they free risk and capital), then increases by shrunk edge, best
    first, so the budget goes to the best trades. Reduces are never blocked: they only
    exit positions we hold. Every increase gets one INFO "Race cap check" line.

    `holdings` feed the gross cap (the taker's "hold book"). The race worst case uses
    `race_holdings` (default: `holdings`) plus `extra_race_legs` - with the ladder on, ALL
    positions incl. ladder inventory, plus every resting ladder order as if filled.
    `books_gross` (ladder + quoter: inventory cost + resting orders at notional) counts with
    the hold book against the global max_total_gross_exposure_fraction; that cap shows up
    in `limited_by` only when it is tighter than the hold-book gross cap.
    `log=False`: no log lines (the per-order pre-check before order books are read).
    """
    race_parties = {key: [m.party for m in group] for key, group in market_map.by_race().items()}
    race_legs = race_legs_from_holdings(market_map, holdings if race_holdings is None else race_holdings)
    for race_key, legs in (extra_race_legs or {}).items():
        race_legs[race_key].extend(legs)

    gross = sum(h.cost_basis for h in holdings.values())
    race_cap = limits.max_race_worst_loss_fraction * bankroll
    gross_cap = limits.max_gross_exposure_fraction * bankroll
    total_cap = limits.max_total_gross_exposure_fraction * bankroll - books_gross
    cycle_cap = limits.max_new_notional_per_cycle_fraction * bankroll
    cycle_notional = 0.0

    def annotate(order: PlannedOrder) -> tuple[PlannedOrder, Any]:
        mapping = market_map.get(order.exchange_id)
        fair_yes = fair.get(order.exchange_id)
        fair_side = None if fair_yes is None else (fair_yes if order.side == "yes" else 1.0 - fair_yes)
        margin = margins.get(order.exchange_id, MIN_ORACLE_MARGIN)
        if fair_side is None:
            edge = None
        elif order.kind == "increase":
            edge = fair_side - margin - order.price
        else:
            edge = order.price - fair_side
        order = replace(order, race_key=mapping.race.key if mapping else "", fair_side=fair_side, margin=margin, edge=edge)
        return order, mapping

    reduces = [annotate(o) for o in planned if o.kind == "reduce"]
    increases = sorted((annotate(o) for o in planned if o.kind == "increase"), key=lambda item: -(item[0].edge or 0.0))

    accepted: list[PlannedOrder] = []
    for order, mapping in reduces:
        notional = order.quantity * order.price
        if mapping is not None:
            race_legs[order.race_key].append(RaceLeg(mapping.party, order.side, -order.quantity, -notional))
        gross = max(0.0, gross - notional)  # approximation: frees roughly the sale value
        loss_after = race_worst_case_loss(race_parties.get(order.race_key, []), race_legs[order.race_key]) if mapping else None
        accepted.append(replace(order, race_loss_after=loss_after, gross_after=gross))

    for order, mapping in increases:
        if mapping is None:
            if log:
                logger.warning("Order dropped: exchange not in market map, race risk unknown", extra={"order": order.__dict__})
            continue
        parties, legs = race_parties.get(order.race_key, [mapping.party]), race_legs[order.race_key]
        loss_before = race_worst_case_loss(parties, legs)
        limits_by_cap = {
            "race_worst_loss": max_shares_within_race_cap(parties, legs, mapping.party, order.side, order.price, order.quantity, race_cap),
            "gross_exposure": math.floor(max(0.0, gross_cap - gross) / order.price),
            "cycle_notional": math.floor(max(0.0, cycle_cap - cycle_notional) / order.price),
        }
        total_room = math.floor(max(0.0, total_cap - gross) / order.price)
        if total_room < limits_by_cap["gross_exposure"]:
            limits_by_cap["total_gross"] = total_room
        quantity = min(order.quantity, *limits_by_cap.values())
        binding = [name for name, q in limits_by_cap.items() if q < order.quantity]

        if quantity < MIN_ORDER_SHARES:
            if not log:
                continue
            logger.info(
                "Race cap check: order dropped",
                extra={
                    "race": order.race_key, "exchange_id": order.exchange_id, "side": order.side,
                    "requested": order.quantity, "allowed": quantity, "limited_by": binding,
                    "race_loss_before": round(loss_before, 2), "race_loss_after": round(loss_before, 2),
                    "race_cap": round(race_cap, 2), "gross": round(gross, 2), "cycle_notional": round(cycle_notional, 2),
                },
            )
            continue

        notional = quantity * order.price
        legs.append(RaceLeg(mapping.party, order.side, quantity, notional))
        loss_after = race_worst_case_loss(parties, legs)
        gross += notional
        cycle_notional += notional
        if not log:
            accepted.append(order)
            continue
        logger.info(
            "Race cap check: order accepted" + (" (shrunk)" if quantity < order.quantity else ""),
            extra={
                "race": order.race_key, "exchange_id": order.exchange_id, "side": order.side,
                "requested": order.quantity, "allowed": quantity, "limited_by": binding,
                "race_loss_before": round(loss_before, 2), "race_loss_after": round(loss_after, 2),
                "race_cap": round(race_cap, 2), "gross": round(gross, 2), "cycle_notional": round(cycle_notional, 2),
            },
        )
        reason = order.reason if quantity == order.quantity else f"{order.reason}; capped {quantity}/{order.quantity} by {','.join(binding)}"
        accepted.append(replace(order, quantity=quantity, reason=reason, race_loss_after=loss_after, gross_after=gross))
    return accepted


# ---------------------------------------------------------------------- strategy


@dataclass
class CycleReport:
    markets: int = 0
    priced: int = 0
    no_fair_price: int = 0
    targets: int = 0
    planned: int = 0
    accepted: int = 0
    rejected: int = 0
    unknown: int = 0
    dry_run: int = 0
    blocked: int = 0
    blocked_by_unknown: int = 0  # orders not planned: recent UNKNOWN order on same exchange/side
    skipped_stale_quote: int = 0  # orders not sent: oracle snapshot older than ORDER_MAX_QUOTE_AGE_S
    skipped_no_book: int = 0  # orders dropped: their order-book read timed out
    skipped_by_caps_precheck: int = 0  # planned orders whose book was not read: no room under the caps
    sig_prices: str = "bulk"  # "bulk" (/exchanges/prices) or "fallback" (per-market order books)
    sig_prices_age_s: float | None = None  # age of the oldest SIG top of book when planning ended
    books_read: int = 0  # /exchanges/{id}/orderbook reads by the taker this cycle
    books_deferred: int = 0  # planned orders whose book read was left to the next cycle (read budget)
    books_cache_hits: int = 0  # planned orders fitted to a book read within BOOK_CACHE_MAX_AGE_S
    get_retries: int = 0  # retried SIG GET attempts this cycle
    slowest_get_s: float = 0.0  # slowest successful logical SIG GET this cycle
    # SIG requests in the last 60 s at cycle end (budget: 100 reads + 30 writes per minute).
    sig_reads_last_min: int = 0
    sig_writes_last_min: int = 0
    sig_429_last_min: int = 0
    skipped_open_orders: int = 0
    relationship_violations: int | None = 0  # None: the constraints read failed or was skipped
    # Deep ladder (zeros while LADDER_ENABLED is off).
    ladder_markets: int = 0
    ladder_placed: int = 0
    ladder_cancels: int = 0
    ladder_fills: int = 0
    ladder_exits_placed: int = 0
    ladder_rounds_closed: int = 0
    ladder_realized_pnl: float = 0.0  # closed rounds this cycle
    ladder_realized_pnl_total: float = 0.0
    ladder_paused: bool = False
    ladder_blocked_global: int = 0  # ladder levels skipped: over TOTAL_GROSS_EXPOSURE_FRAC
    ladder_blocked_direction: int = 0  # ... over the directional cap (all books)
    # Quoter (zeros while QUOTER_ENABLED is off).
    quoter_markets: int = 0
    quoter_placed: int = 0
    quoter_cancels: int = 0
    quoter_requotes: int = 0
    quoter_fills: int = 0
    quoter_killed: int = 0
    quoter_blocked: dict[str, int] = field(default_factory=dict)  # guard / cap -> count, non-zero only
    # Fill markouts (MARKOUT_ENABLED): new fills seen, markout lines written; None = read failed.
    fills_new: int | None = 0
    markouts: int = 0
    kelly_fraction: float = 0.0
    bankroll: float = 0.0
    duration_s: float = 0.0
    no_fair_reasons: dict[str, int] = field(default_factory=dict)


class Strategy:
    def __init__(
        self,
        settings: Settings,
        exchange: ExchangeClient,
        polymarket: PolymarketClient,
        kalshi: KalshiClient,
        market_map: MarketMap,
        journal: OrderJournal,
        limits: RiskLimits = RiskLimits(),
        ladder: LadderManager | None = None,
        quoter: QuoterManager | None = None,
        markout: MarkoutTracker | None = None,
        fill_feed: FillFeed | None = None,
    ) -> None:
        self._settings = settings
        self._exchange = exchange
        self._polymarket = polymarket
        self._kalshi = kalshi
        self._market_map = market_map
        self._journal = journal
        self._base_limits = limits
        # Deep ladder: None (the default, LADDER_ENABLED = False) leaves the cycle unchanged.
        if ladder is None and LADDER_ENABLED:
            ladder = LadderManager(exchange, polymarket, kalshi, compute_fair_price)
        self._ladder = ladder
        # Quoter: None (the default, QUOTER_ENABLED = False) leaves the cycle unchanged.
        if quoter is None and QUOTER_ENABLED:
            quoter = QuoterManager(exchange)  # QUOTER_DRY_RUN / QUOTER_MARKET_ALLOWLIST from config
        self._quoter = quoter
        # Fill markouts: on by default (MARKOUT_ENABLED), and always with the quoter (its
        # per-market kill needs them). One fills read per cycle, none in DRY_RUN.
        if markout is None and (MARKOUT_ENABLED or quoter is not None):
            markout = MarkoutTracker()
        if fill_feed is None and markout is not None:
            fill_feed = FillFeed(exchange, settings.tournament_slug)
        self._markout = markout
        self._fill_feed = fill_feed

        self._tournament: dict[str, Any] | None = None
        self._clusters: list[Cluster] = build_clusters(market_map)
        self._relationships_loaded_at = 0.0
        self._relationships_count: int | None = None  # from the last good /relationships read
        self._constraints_checked = False  # the start-up constraints read has been made
        self._standing: float | None = None
        self._standing_loaded_at = 0.0
        self._last_bankroll: float | None = None
        self._last_books: dict[str, SigBook] = {}  # last good bulk top of book (fallback candidates)
        self._recent_books: dict[str, dict[str, Any]] = {}  # exchange_id -> last order book read (has receivedAt)
        self._deferred_books: set[str] = set()  # exchanges deferred last cycle: read first this cycle

    @property
    def journal(self) -> OrderJournal:
        return self._journal

    # ------------------------------------------------------------ slow-changing state
    #
    # Reads are ESSENTIAL or NON-ESSENTIAL. Essential (positions, open orders, SIG prices,
    # oracle quotes, order books): any failure propagates and the cycle is skipped - it
    # never runs on empty or stale holdings. Non-essential (tournament, leaderboard,
    # relationships, constraints, and pnl once a bankroll is known): on a transient
    # failure log one WARNING, use the last known value / neutral default, carry on, and
    # retry next cycle. Auth / terms errors are never absorbed.

    async def _refresh_tournament(self) -> dict[str, Any]:
        if self._tournament is None:
            try:
                self._tournament = await self._exchange.get_tournament(self._settings.tournament_slug)
            except Exception as exc:
                _absorb_non_essential("tournament", exc, "no time-remaining data: tournament Kelly modifier off")
                return {}
        return self._tournament

    async def _refresh_clusters(self) -> None:
        if time.monotonic() - self._relationships_loaded_at < RELATIONSHIPS_REFRESH_S:
            return
        try:
            relationships = [r async for r in self._exchange.iter_relationships()]
        except Exception as exc:
            _absorb_non_essential("relationships", exc, f"keep previous {len(self._clusters)} clusters")
            return
        self._clusters = build_clusters(self._market_map, relationships)
        self._relationships_loaded_at = time.monotonic()
        self._relationships_count = len(relationships)
        logger.info("Risk clusters rebuilt", extra={"relationships": len(relationships), "clusters": len(self._clusters)})

    async def _refresh_standing(self) -> float | None:
        if time.monotonic() - self._standing_loaded_at >= LEADERBOARD_REFRESH_S:
            try:
                board = await self._exchange.get_leaderboard(self._settings.tournament_slug, limit=1)
            except Exception as exc:
                fallback = "neutral (base kelly_fraction)" if self._standing is None else f"last known standing {self._standing:.3f}"
                _absorb_non_essential("leaderboard", exc, fallback)
                return self._standing
            self._standing = leaderboard_standing(board.get("myRank"), board.get("total"))
            self._standing_loaded_at = time.monotonic()
        return self._standing

    async def _read_bankroll(self) -> float:
        """Account value from /portfolio/pnl; last known value if that read fails.

        Essential until the first success: with no known bankroll there is nothing safe
        to size against, so the failure propagates and the cycle is skipped.
        """
        try:
            pnl = await self._exchange.get_pnl()
        except Exception as exc:
            if self._last_bankroll is None:
                raise
            _absorb_non_essential("portfolio/pnl", exc, f"last known bankroll {self._last_bankroll:.2f}")
            return self._last_bankroll
        self._last_bankroll = float(pnl.get("totalAccountValue") or 0.0)
        return self._last_bankroll

    def _time_remaining(self, tournament: dict[str, Any]) -> float:
        start, end = tournament.get("startDate"), tournament.get("endDate")
        if not start or not end:
            return 1.0
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        total = (end_dt - start_dt).total_seconds()
        left = (end_dt - datetime.now(timezone.utc)).total_seconds()
        return min(max(left / total, 0.0), 1.0) if total > 0 else 1.0

    # ------------------------------------------------------------ the cycle

    async def run_cycle(self) -> CycleReport:
        started = time.monotonic()
        report = CycleReport(markets=len(self._market_map))
        self._exchange.reset_get_stats()  # get_retries / slowest_get_s count this cycle only
        mappings = list(self._market_map)
        exchange_ids = [m.sig_exchange_id for m in mappings]

        # 1. STATE - always from the API, never from memory. Positions and open orders are
        #    essential: if either fails, the exception skips the whole cycle.
        bankroll, positions, open_orders = await asyncio.gather(
            self._read_bankroll(),
            self._exchange.get_positions(),
            self._collect_open_orders(),
        )
        all_holdings = holdings_from_positions(positions)
        self._exchange.note_open_orders(open_orders)  # self-cross guard: our resting orders
        # With the ladder / quoter on, the taker sees only its "hold book": their inventory
        # and resting orders are their business (and their own budgets).
        holdings = all_holdings
        book_ids: set[str] = set()
        for book in self._books():
            holdings = book.hold_book(holdings)
            book_ids |= book.order_ids()
        open_order_exchanges = {o["exchangeId"] for o in open_orders if str(o.get("id")) not in book_ids}
        report.bankroll = bankroll

        # 2. PRICES - SIG first: ONE bulk read of every mapped exchange's top of book
        #    (ceil(N / 100) requests), scoped by the tournament id (read once, cached).
        #    Oracles LAST and together, right before the fair-price check: a slow SIG
        #    fetch can no longer age the oracle quotes past the staleness threshold.
        tournament = await self._refresh_tournament()
        books = await self._read_bulk_books(exchange_ids, tournament.get("id"))
        pm_quotes, ks_quotes = await asyncio.gather(
            self._polymarket.get_quotes(self._market_map.polymarket_tokens()),
            self._kalshi.get_quotes(self._market_map.kalshi_tickers()),
        )

        # 3. FAIR - skip (and count) every market without a trustworthy oracle price.
        fair: dict[str, float] = {}
        margins: dict[str, float] = {}
        quote_times: dict[str, float | None] = {}
        reasons: dict[str, int] = {}
        for mapping in mappings:
            eid = mapping.sig_exchange_id
            price, reason = compute_fair_price(
                pm_quotes.get(mapping.polymarket_token) if mapping.polymarket_token else None,
                ks_quotes.get(mapping.kalshi_ticker) if mapping.kalshi_ticker else None,
                self._settings.oracle_staleness_threshold_s,
            )
            if price is None:
                reasons[reason.split(" (")[0]] = reasons.get(reason.split(" (")[0], 0) + 1
                logger.debug("Skipping market this cycle", extra={"exchange_id": eid, "title": mapping.title, "reason": reason})
                continue
            fair[eid] = price.prob
            margins[eid] = price.margin
            quote_times[eid] = price.fetched_at
        book_cache: dict[str, dict[str, Any]] = {}  # order books read this cycle, reused by the fit
        if books is None:  # the bulk read failed: books only for markets that may trade
            report.sig_prices = "fallback"
            books = await self._fallback_books(fair, margins, holdings, book_cache)
            report.books_read += len(book_cache)
        for eid in [eid for eid in fair if eid not in books]:
            reasons["no SIG price"] = reasons.get("no SIG price", 0) + 1
            del fair[eid]
        report.priced = len(fair)
        report.no_fair_price = len(mappings) - len(fair)
        report.no_fair_reasons = reasons

        opportunities = {
            eid: Opportunity(
                eid,
                fair_prob=prob,
                yes_price=books[eid].buy_price("yes"),
                no_price=books[eid].buy_price("no"),
                margin=margins[eid],
            )
            for eid, prob in fair.items()
        }

        # 4. SIZE - Kelly per cluster; ALL risk limits are applied inside size_portfolio
        #    (per contract -> per cluster -> per direction -> portfolio).
        await self._refresh_clusters()
        standing = await self._refresh_standing()
        kelly_fraction = tournament_kelly_fraction(
            self._base_limits.kelly_fraction, standing, self._time_remaining(tournament)
        )
        limits = replace(self._base_limits, kelly_fraction=kelly_fraction)
        report.kelly_fraction = kelly_fraction

        # Every held position goes in: ones without a same-side target this cycle still
        # use up the cluster, directional and portfolio caps (see size_portfolio).
        held = {eid: (holding.side, holding.market_value) for eid, holding in holdings.items()}
        held_shares = {eid: holding.shares for eid, holding in holdings.items()}  # stress directional mode only
        sized = size_portfolio(self._clusters, opportunities, bankroll, self._market_map, limits, held, held_shares)
        targets = {bet.exchange_id: bet for bet in sized}
        report.targets = len(targets)

        # 5. PLAN - explicit increase / reduce orders, then the exposure gates: race
        #    worst-case loss, gross exposure, per-cycle notional. Order books are read only
        #    for orders that can still be sent: not blocked by a recent UNKNOWN, and with
        #    room under the caps on their own (a necessary condition - fitting to the book
        #    only shrinks an order or worsens its price). The caps then run once more, in
        #    full, on the fitted orders, exactly as before.
        skip = open_order_exchanges & set(fair)
        report.skipped_open_orders = len(skip)
        planned = plan_orders(fair, targets, holdings, books, skip, self._market_map)
        # Before the caps, so a blocked order does not use up race / gross / cycle budget.
        planned, report.blocked_by_unknown = drop_blocked_by_unknown(
            planned, self._journal.recent_unknown(UNKNOWN_ORDER_BLOCK_S)
        )
        caps = dict(
            race_holdings=all_holdings,
            extra_race_legs=self._resting_race_legs() or None,
            books_gross=sum(book.book_cost() for book in self._books()),
        )
        precheck = [o for o in planned if apply_exposure_caps([o], holdings, self._market_map, fair, margins, bankroll, limits, log=False, **caps)]
        report.skipped_by_caps_precheck = len(planned) - len(precheck)
        planned, report.skipped_no_book = await self._fit_to_books(precheck, fair, limits, margins, book_cache, report)
        planned = apply_exposure_caps(planned, holdings, self._market_map, fair, margins, bankroll, limits, **caps)
        # Quote age = the older of the oracle snapshot and the order book the order was fitted
        # to (a freshly read book is newer than the oracles, so this only bites for a cached one).
        planned = [replace(o, quote_fetched_at=_older(quote_times.get(o.exchange_id), (book_cache.get(o.exchange_id) or {}).get("receivedAt"))) for o in planned]
        report.planned = len(planned)
        fetched = [books[eid].fetched_at for eid in fair if books[eid].fetched_at is not None]
        report.sig_prices_age_s = round(time.time() - min(fetched), 2) if fetched else None

        # 6. EXECUTE - one failure never aborts the rest of the cycle.
        sent = await self._execute_planned(planned, report)

        # Our new fills (taker, ladder, quoter): markout journal, and handed to the ladder
        # and the quoter; then the markouts due now (the quoter's kill switch reads them).
        new_fills = await self._read_new_fills(report)
        if new_fills:
            for fill in new_fills:
                source, action = self._fill_source(str(fill.get("orderId")))
                self._markout.record(source, fill, action, fair.get(str(fill.get("exchangeId"))))
        markouts = self._markout.update(time.time(), fair) if self._markout is not None else []
        report.markouts = len(markouts)

        # Quoter, then deep ladder (each only when enabled). Each sees the taker's book -
        # hold-book cost plus every buy sent this cycle (accepted, unknown or dry-run) - and
        # the other one's book for the global gross and directional caps. A transient
        # failure in either never fails the taker cycle.
        if self._quoter is not None or self._ladder is not None:
            taker_book = [(eid, h.side, h.cost_basis) for eid, h in holdings.items()]
            taker_book += [(o.exchange_id, o.side, o.quantity * o.price) for o in sent if o.kind == "increase"]
            shared = (report, bankroll, tournament, fair, pm_quotes, ks_quotes, books, all_holdings, open_orders, limits)
            if self._quoter is not None:
                await self._run_quoter(*shared, taker_book=taker_book, fills=new_fills, markouts=markouts)
            if self._ladder is not None:
                await self._run_ladder(*shared, taker_book=taker_book, fills=new_fills)

        # Strategy #1 - observe only.
        report.relationship_violations = await self.observe_relationship_violations()

        report.duration_s = round(time.monotonic() - started, 2)
        stats = self._exchange.get_stats()
        report.get_retries, report.slowest_get_s = stats["get_retries"], stats["slowest_get_s"]
        report.sig_reads_last_min = stats.get("reads_last_min", 0)
        report.sig_writes_last_min = stats.get("writes_last_min", 0)
        report.sig_429_last_min = stats.get("rate_limited_last_min", 0)
        logger.info("Cycle done", extra={k: v for k, v in report.__dict__.items()})
        return report

    def _books(self) -> list[Any]:
        """The non-taker books that are on: ladder and / or quoter."""
        return [book for book in (self._ladder, self._quoter) if book is not None]

    def _resting_race_legs(self, exclude: Any = None) -> dict[str, list[RaceLeg]]:
        """Resting ladder and quoter orders as if filled, by race (optionally without one book)."""
        legs: dict[str, list[RaceLeg]] = {}
        for book in self._books():
            if book is exclude:
                continue
            for race, race_legs in book.resting_race_legs(self._market_map).items():
                legs.setdefault(race, []).extend(race_legs)
        return legs

    def _other_books(self, taker_book: list[tuple[str, str, float]], me: Any) -> tuple[float, dict[Direction, float], dict[str, list[RaceLeg]], frozenset[str]]:
        """For one of ladder / quoter: gross and directional exposure of the taker and the
        OTHER book, the other book's resting race legs, and its markets in use."""
        gross = sum(cost for _, _, cost in taker_book)
        directional = dict(directional_exposure(self._market_map, taker_book))
        exclude: frozenset[str] = frozenset()
        for book in self._books():
            if book is me:
                continue
            gross += book.book_cost()
            for direction, cost in book.directional_cost(self._market_map).items():
                directional[direction] = directional.get(direction, 0.0) + cost
            exclude |= book.markets_in_use()
        return gross, directional, self._resting_race_legs(exclude=me), exclude

    def _quotes_by_exchange(self, pm_quotes, ks_quotes) -> dict[str, tuple[Any, Any]]:
        return {
            m.sig_exchange_id: (
                pm_quotes.get(m.polymarket_token) if m.polymarket_token else None,
                ks_quotes.get(m.kalshi_ticker) if m.kalshi_ticker else None,
            )
            for m in self._market_map
        }

    async def _run_quoter(
        self, report, bankroll, tournament, fair, pm_quotes, ks_quotes, books, all_holdings, open_orders, limits,
        taker_book: list[tuple[str, str, float]], fills: list[dict[str, Any]] | None, markouts: list[Any],
    ) -> None:
        other_gross, other_directional, other_legs, exclude = self._other_books(taker_book, self._quoter)
        ctx = QuoterContext(
            now=time.time(),
            bankroll=bankroll,
            tournament_id=(tournament or {}).get("id"),
            market_map=self._market_map,
            fair=fair,
            quotes=self._quotes_by_exchange(pm_quotes, ks_quotes),
            books=books,
            holdings=all_holdings,
            open_orders=open_orders,
            limits=limits,
            max_quote_age_s=self._settings.oracle_staleness_threshold_s,
            other_gross=other_gross,
            other_directional=other_directional,
            other_race_legs=other_legs,
            exclude_markets=exclude,
            fills=fills,
            markouts=list(markouts),
        )
        try:
            quoter = await self._quoter.run_cycle(ctx)
        except Exception as exc:
            _absorb_non_essential("quoter", exc, "quoter skipped this cycle")
            return
        report.quoter_markets, report.quoter_placed, report.quoter_cancels = quoter.markets, quoter.placed, quoter.cancels
        report.quoter_requotes, report.quoter_fills, report.quoter_killed = quoter.requotes, quoter.fills, quoter.killed
        report.quoter_blocked = {
            name.removeprefix("blocked_"): value for name, value in quoter.__dict__.items() if name.startswith("blocked_") and value
        }

    async def _run_ladder(
        self, report, bankroll, tournament, fair, pm_quotes, ks_quotes, books, all_holdings, open_orders, limits,
        taker_book: list[tuple[str, str, float]] | None = None,
        fills: list[dict[str, Any]] | None = None,
    ) -> None:
        other_gross, other_directional, other_legs, exclude = self._other_books(taker_book or [], self._ladder)
        quotes = self._quotes_by_exchange(pm_quotes, ks_quotes)
        ctx = LadderContext(
            now=time.time(),
            bankroll=bankroll,
            tournament_id=(tournament or {}).get("id"),
            tournament_slug=self._settings.tournament_slug,
            market_map=self._market_map,
            fair=fair,
            quotes=quotes,
            books=books,
            holdings=all_holdings,
            open_orders=open_orders,
            limits=limits,
            max_quote_age_s=self._settings.oracle_staleness_threshold_s,
            other_gross=other_gross,
            other_directional=other_directional,
            other_race_legs=other_legs,
            exclude_markets=exclude,
            fills=fills,
        )
        try:
            ladder = await self._ladder.run_cycle(ctx)
        except Exception as exc:
            _absorb_non_essential("ladder", exc, "ladder skipped this cycle")
            return
        report.ladder_markets, report.ladder_placed, report.ladder_cancels = ladder.markets, ladder.placed, ladder.cancels
        report.ladder_fills, report.ladder_exits_placed = ladder.fills, ladder.exits_placed
        report.ladder_rounds_closed, report.ladder_realized_pnl = ladder.rounds_closed, ladder.realized_pnl
        report.ladder_realized_pnl_total, report.ladder_paused = ladder.realized_pnl_total, ladder.paused
        report.ladder_blocked_global, report.ladder_blocked_direction = ladder.blocked_global, ladder.blocked_direction

    async def _read_new_fills(self, report: CycleReport) -> list[dict[str, Any]] | None:
        """This cycle's new fills, or None (markouts off, or the read failed - non-essential)."""
        if self._fill_feed is None:
            report.fills_new = None
            return None
        try:
            fills = await self._fill_feed.read()
        except Exception as exc:
            _absorb_non_essential("portfolio/fills", exc, "fills read next cycle")
            report.fills_new = None
            return None
        report.fills_new = len(fills)
        return fills

    def _fill_source(self, order_id: str) -> tuple[str, str | None]:
        """(source, canonical action) of one of our fills, by its order id."""
        for name, book in (("ladder", self._ladder), ("quoter", self._quoter)):
            action = book.order_action(order_id) if book is not None else None
            if action is not None:
                return name, action
        return "taker", self._journal.action_for_order(order_id)

    async def _read_bulk_books(self, exchange_ids: list[str], tournament_id: str | None) -> dict[str, SigBook] | None:
        """Every mapped exchange's top of book from /exchanges/prices, or None if that read
        failed transiently (the cycle then falls back to per-market order books)."""
        try:
            prices = await self._exchange.get_prices(exchange_ids, tournament_id)
        except Exception as exc:
            _absorb_non_essential("exchanges/prices", exc, "order books for markets that may trade")
            return None
        self._last_books = {
            eid: SigBook(yes_bid=row.get("bestBid"), yes_ask=row.get("bestAsk"), fetched_at=row.get("receivedAt"), last=row.get("latestPrice"))
            for eid, row in prices.items()
        }
        return self._last_books

    async def _fallback_books(
        self, fair: dict[str, float], margins: dict[str, float], holdings: dict[str, Holding], cache: dict[str, dict[str, Any]]
    ) -> dict[str, SigBook]:
        """Top of book from per-market order books, for at most FALLBACK_MAX_BOOKS markets
        that may trade: held positions first, then markets whose LAST good bulk price
        showed an edge against this cycle's fair. The books land in `cache` for the fit."""
        def edge(eid: str) -> float:
            last = self._last_books.get(eid)
            if last is None:
                return -1.0
            fair_yes, margin = fair[eid], margins.get(eid, MIN_ORACLE_MARGIN)
            edges = [fair_yes - margin - last.yes_ask if last.yes_ask is not None else -1.0]
            edges.append((1.0 - fair_yes) - margin - (1.0 - last.yes_bid) if last.yes_bid is not None else -1.0)
            return max(edges)

        held = sorted(eid for eid in fair if eid in holdings)
        rest = sorted((eid for eid in fair if eid not in holdings and edge(eid) > 0), key=lambda eid: -edge(eid))
        candidates = (held + rest)[:FALLBACK_MAX_BOOKS]
        logger.warning("SIG bulk prices unavailable: reading order books instead", extra={"markets": len(candidates), "held": len(held)})
        results = await asyncio.gather(*(self._exchange.get_orderbook(eid) for eid in candidates), return_exceptions=True)
        books: dict[str, SigBook] = {}
        for eid, orderbook in zip(candidates, results):
            if isinstance(orderbook, (AuthError, TermsNotAcknowledgedError)):
                raise orderbook
            if isinstance(orderbook, BaseException):
                continue  # this market is simply not priced this cycle
            cache[eid] = orderbook
            bids, asks = orderbook.get("bids") or [], orderbook.get("asks") or []
            books[eid] = SigBook(
                yes_bid=float(bids[0]["price"]) if bids else None,
                yes_ask=float(asks[0]["price"]) if asks else None,
                fetched_at=orderbook.get("receivedAt"),
            )
        return books

    async def _fit_to_books(
        self,
        planned: list[PlannedOrder],
        fair: dict[str, float],
        limits: RiskLimits,
        margins: dict[str, float],
        cache: dict[str, dict[str, Any]] | None = None,
        report: CycleReport | None = None,
    ) -> tuple[list[PlannedOrder], int]:
        """Resize every planned order to the depth resting in its SIG order book.

        Returns (fitted orders, number dropped because their book read timed out). A
        timed-out book drops only ITS order - an order is never sent without a fresh book.
        Any other failure (auth, terms, non-timeout API error, cancellation) still fails
        the whole cycle, as before.

        Read budget (clients with a read bucket): a book read within BOOK_CACHE_MAX_AGE_S
        is reused; of the rest only as many are read as ExchangeClient.read_budget allows
        (book_read_order decides which), each waiting at most BOOK_READ_WINDOW_S for its
        token. The others are deferred - INFO "Order deferred: read budget", no request,
        no retry - and served first next cycle.
        """
        if not planned:
            return [], 0
        cache = {} if cache is None else cache
        now = time.time()
        for eid in sorted({o.exchange_id for o in planned} - set(cache)):
            recent = self._recent_books.get(eid)
            if recent is not None and now - float(recent.get("receivedAt") or 0.0) <= BOOK_CACHE_MAX_AGE_S:
                cache[eid] = recent
                if report is not None:
                    report.books_cache_hits += 1
        need = [o for o in planned if o.exchange_id not in cache]
        to_read = sorted({o.exchange_id for o in need})
        read_budget = getattr(self._exchange, "read_budget", None)  # None: a client without a read bucket
        deferred: set[str] = set()
        if read_budget is not None and to_read:
            budget = read_budget(READ_BOOK, BOOK_READ_WINDOW_S, self._book_read_reserve())
            if len(to_read) > budget:
                ranked = book_read_order(need, fair, margins, self._deferred_books)
                to_read, deferred = sorted(ranked[:budget]), set(ranked[budget:])
        if read_budget is not None:
            read = await asyncio.gather(*(self._exchange.get_orderbook(eid, token_wait_s=BOOK_READ_WINDOW_S) for eid in to_read), return_exceptions=True)
        else:
            read = await asyncio.gather(*(self._exchange.get_orderbook(eid) for eid in to_read), return_exceptions=True)
        for eid, orderbook in zip(to_read, read):
            if isinstance(orderbook, ReadDeferredError):
                deferred.add(eid)  # no token within the window: nothing was sent
                continue
            cache[eid] = orderbook
            if report is not None:
                report.books_read += 1
            if isinstance(orderbook, dict):
                self._recent_books[eid] = orderbook
        self._deferred_books = deferred
        fitted = []
        skipped = 0
        for order in planned:
            if order.exchange_id in deferred:
                if report is not None:
                    report.books_deferred += 1
                logger.info(
                    "Order deferred: read budget",
                    extra={"exchange_id": order.exchange_id, "side": order.side, "kind": order.kind, "quantity": order.quantity, "price": order.price},
                )
                continue
            orderbook = cache[order.exchange_id]
            if isinstance(orderbook, (RequestTimeoutError, ReadPausedError)):
                skipped += 1
                logger.warning(
                    "Order skipped: order book unavailable",
                    extra={"exchange_id": order.exchange_id, "side": order.side, "kind": order.kind, "error": repr(orderbook)},
                )
                continue
            if isinstance(orderbook, BaseException):
                raise orderbook
            margin = margins.get(order.exchange_id, MIN_ORACLE_MARGIN)
            result = fit_to_book(order, orderbook, fair[order.exchange_id], limits, margin)
            if result is None:
                logger.info("Planned order dropped: not enough liquidity at an acceptable price", extra={"order": order.__dict__})
            elif result is not order:
                logger.info("Planned order fitted to book", extra={"planned": order.__dict__, "fitted": result.__dict__})
            if result is not None:
                fitted.append(result)
        return fitted, skipped

    def _book_read_reserve(self) -> int:
        """Read tokens to keep for the next cycle's essential and bulk reads (positions,
        orders, pnl, ceil(N / 100) price pages, fills) beyond what the bucket refills during
        the shortest gap between cycles."""
        needed = 3 + math.ceil(len(self._market_map) / 100) + (1 if self._fill_feed is not None and not self._exchange.dry_run else 0)
        return max(0, math.ceil(needed - SIG_READS_PER_MIN / 60.0 * MIN_CYCLE_GAP_S))

    async def _collect_open_orders(self) -> list[dict[str, Any]]:
        return [order async for order in self._exchange.iter_orders(status="open")]

    async def _execute_planned(self, planned: list[PlannedOrder], report: CycleReport) -> list[PlannedOrder]:
        """Execute in order; returns the orders that may have reached the book (accepted,
        unknown, or dry-run) - the ladder counts them as taker exposure."""
        sent: list[PlannedOrder] = []
        for order in planned:
            outcome = await self._execute(order)
            if outcome is None:
                report.skipped_stale_quote += 1
                continue
            setattr(report, outcome.value, getattr(report, outcome.value) + 1)
            if outcome in (OrderStatus.ACCEPTED, OrderStatus.UNKNOWN, OrderStatus.DRY_RUN):
                sent.append(order)
        return sent

    async def _execute(self, planned: PlannedOrder) -> OrderStatus | None:
        """Send one planned order. None: not sent, its oracle snapshot was too old.

        The write token is taken BEFORE the quote-age check, so the check runs after any
        wait for the bucket: slow order-book fetches, earlier POSTs in this cycle and the
        bucket itself can age the snapshot well past the start of the cycle. A skipped
        order gives its token back.
        """
        prepaid = False if self._exchange.dry_run else await self._exchange.acquire_order_token()
        if quote_too_old(planned):
            if prepaid:
                self._exchange.refund_order_token()
            return None

        expires = (datetime.now(timezone.utc) + timedelta(seconds=ORDER_TTL_S)).isoformat(timespec="milliseconds")
        try:
            order = OrderRequest(
                exchange_id=planned.exchange_id,
                side=planned.side,
                action=planned.action,
                quantity=planned.quantity,
                price=planned.price,
                expiration_date=expires.replace("+00:00", "Z"),
            )
        except ValueError as exc:
            if prepaid:
                self._exchange.refund_order_token()
            logger.warning("Planned order is invalid, skipped", extra={"order": planned.__dict__, "error": str(exc)})
            return OrderStatus.REJECTED

        key = new_idempotency_key(f"s2-{planned.kind}")
        payload = order.to_payload()
        context = {"kind": planned.kind, "reason": planned.reason, "payload": payload}

        # One measurement line per order, real or DRY_RUN, logged before anything is sent.
        logger.info(
            "Order intent",
            extra={
                "key": key,
                "exchange_id": planned.exchange_id,
                "race": planned.race_key,
                "kind": planned.kind,
                "side": planned.side,
                "action": planned.action,
                "price": planned.price,
                "quantity": planned.quantity,
                "fair_side": None if planned.fair_side is None else round(planned.fair_side, 4),
                "margin": round(planned.margin, 4),
                "shrunk_edge": None if planned.edge is None else round(planned.edge, 4),
                "race_worst_loss_after": None if planned.race_loss_after is None else round(planned.race_loss_after, 2),
                "gross_exposure_after": None if planned.gross_after is None else round(planned.gross_after, 2),
                "dry_run": self._exchange.dry_run,
            },
        )

        if not self._exchange.dry_run:
            self._journal.record(key, OrderStatus.PENDING, **context)
        try:
            result = await self._exchange.place_order(order, key, prepaid=prepaid)
        except asyncio.CancelledError:
            # The cycle was cancelled (e.g. main.py's cycle timeout) with this POST in
            # flight: the engine may still accept it. Journal UNKNOWN so the next cycles
            # block this market/side, then let the cancellation propagate.
            if not self._exchange.dry_run:
                self._journal.record(key, OrderStatus.UNKNOWN, error="cycle cancelled while POST in flight", **context)
                logger.warning("Order outcome unknown (cycle cancelled mid-POST)", extra={"key": key, "exchange_id": planned.exchange_id})
            raise
        except SelfCrossError as exc:
            # Never sent (exchange_api logged "self_cross_blocked"): a settled rejection.
            self._journal.record(key, OrderStatus.REJECTED, code=exc.code, resting_order_id=exc.resting_order_id, **context)
            return OrderStatus.REJECTED
        except (TermsNotAcknowledgedError, AuthError) as exc:
            # Every other order would fail the same way - stop the cycle, a human must act.
            self._journal.record(key, OrderStatus.REJECTED, code=exc.code, **context)
            raise
        except ApiError as exc:
            status = OrderStatus.UNKNOWN if exc.retryable or exc.code == "ORDER_STATUS_UNKNOWN" else OrderStatus.REJECTED
            self._journal.record(key, status, code=exc.code, error=exc.message, **context)
            logger.error("Order failed", extra={"key": key, "status": status.value, "code": exc.code, "detail": exc.message})
            return status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._journal.record(key, OrderStatus.UNKNOWN, error=repr(exc), **context)
            logger.error("Order outcome unknown (network)", extra={"key": key, "error": repr(exc)})
            return OrderStatus.UNKNOWN

        if result is None:
            # Nothing was sent: either DRY_RUN or the kill switch. Never record it as a
            # fill - positions are re-read from the API every cycle anyway.
            status = OrderStatus.BLOCKED if self._exchange.halted else OrderStatus.DRY_RUN
            self._journal.record(key, status, **context)
            return status

        # The response is the truth: record the CANONICAL order, not what we sent.
        canonical = {k: result.get(k) for k in ("orderId", "action", "side", "price", "quantity", "quantityTraded", "remainingQuantity", "open", "terminalReasonCode")}
        self._journal.record(key, OrderStatus.ACCEPTED, result=canonical, **context)
        if (result.get("action"), result.get("side")) != (order.action, order.side):
            logger.warning("Engine canonicalized the order", extra={"key": key, "sent": payload, "canonical": canonical})
        return OrderStatus.ACCEPTED

    def _constraints_due(self) -> bool:
        """RELATIONSHIPS_CHECK_ENABLED: "on" every cycle, "off" never, "auto" once at start
        and afterwards only while the last relationships read was not empty."""
        if RELATIONSHIPS_CHECK_ENABLED != "auto":
            return RELATIONSHIPS_CHECK_ENABLED == "on"
        if not self._constraints_checked:
            return True
        return bool(self._relationships_count)

    async def observe_relationship_violations(self) -> int | None:
        """Strategy #1, observe-only: log engine-reported violations and suggested trades.

        Non-essential and runs AFTER orders were sent: a failure here must not turn a
        cycle that already traded into a "failed" one. Returns None when unknown - or when
        the read is skipped (see _constraints_due), which costs no request.
        """
        if not self._constraints_due():
            return None
        first = not self._constraints_checked
        self._constraints_checked = True
        if first and RELATIONSHIPS_CHECK_ENABLED == "auto":
            logger.info(
                "Relationship constraints check",
                extra={"relationships": self._relationships_count, "every_cycle": bool(self._relationships_count)},
            )
        try:
            body = await self._exchange.get_constraints(violations_only=True)
        except Exception as exc:
            _absorb_non_essential("relationships/constraints", exc, "violations unknown this cycle")
            return None
        for constraint in body.get("data") or []:
            logger.warning(
                "Relationship violation (observe-only, not traded)",
                extra={
                    "relationship_id": constraint.get("relationshipId"),
                    "type": constraint.get("type"),
                    "violation": constraint.get("violationAmount"),
                    "rule": (constraint.get("constraint") or {}).get("priceRule"),
                    "suggested": [
                        (t.get("action"), t.get("outcomeSide"), t.get("exchangeId"), t.get("currentPrice"))
                        for t in constraint.get("suggestedCorrectiveTrades") or []
                    ],
                },
            )
        return int(body.get("violationsCount") or 0)


# ---------------------------------------------------------------------- sanity check


async def _smoke_test(allow_live: bool) -> int:
    from config import load_settings
    from logging_config import setup_logging

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    log = logging.getLogger("strategy.smoke_test")
    if not settings.dry_run and not allow_live:
        log.error("DRY_RUN is off - refusing to run a cycle from the smoke test without --live")
        return 2

    market_map = MarketMap.load()
    async with ExchangeClient(settings) as exchange, PolymarketClient() as poly, KalshiClient() as kalshi:
        strategy = Strategy(settings, exchange, poly, kalshi, market_map, OrderJournal())
        await strategy.run_cycle()
    return 0


if __name__ == "__main__":
    #   python strategy.py         one full cycle; with DRY_RUN=true orders are only logged
    #   python strategy.py --live  same, but allowed to trade when DRY_RUN=false
    import sys

    raise SystemExit(asyncio.run(_smoke_test("--live" in sys.argv[1:])))
