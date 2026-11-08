"""Position sizing: cluster-aware fractional Kelly.

No price history exists on the platform, so correlation is never estimated
statistically. Instead contracts are grouped into risk clusters by LOGIC:

- same race (D / R / I of "who wins the Rhode Island Senate") - from market_map;
- SIG relationship graph edges (/relationships) - when SIG publishes them.

Two sizing rules, depending on what we know about a cluster:

1. Exclusive cluster (outcomes are mutually exclusive, e.g. one race): the joint
   distribution is known - exactly one outcome happens (or "someone else" wins) -
   so we solve the exact multi-outcome Kelly problem numerically:
       maximise  sum_s P(s) * log(1 + sum_j f_j * r_j(s))
   over stake fractions f_j >= 0, for YES and NO bets together. This correctly
   treats "YES on R" and "NO on D" as largely the same bet, instead of doubling up.

2. Linked cluster (logically related but joint distribution unknown, e.g. an
   implication): each bet is Kelly-sized alone, then the cluster total is capped
   at the single best bet's size - i.e. treated as perfectly correlated. Conservative
   on purpose.

Then: x kelly_fraction (fractional Kelly), per-contract cap, per-cluster cap,
per-direction cap (total bet on one party across all races - the "national wave"
factor that race clusters cannot see), portfolio cap. Outputs are TARGET positions;
strategy.py diffs them against what is already held.

Prices are what we would actually pay per share (YES ask, or NO ask = 1 - YES bid),
so the spread is already inside the edge. The oracle fair probability is shrunk by
an uncertainty margin before any bet is considered (`check_buy`), and contracts above
MAX_BUY_PRICE need a larger shrunk edge.

`race_worst_case_loss` / `max_shares_within_race_cap` price a whole race across every
"who wins" scenario; strategy.py uses them as the last gate before execution.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

import numpy as np

from config import (
    HIGH_PRICE_MIN_EDGE,
    MAX_BUY_PRICE,
    DIRECTIONAL_LIMIT_MODE,
    DIRECTIONAL_STRESS_FRAC,
    DIRECTIONAL_STRESS_S,
    MAX_GROSS_EXPOSURE_FRAC,
    TAKER_GROSS_FRAC,
    TOTAL_GROSS_EXPOSURE_FRAC,
    MAX_MARKET_FRAC,
    MAX_NEW_NOTIONAL_PER_CYCLE_FRAC,
    MAX_RACE_WORST_LOSS_FRAC,
)
from market_mapper import MarketMap

logger = logging.getLogger(__name__)

Side = Literal["yes", "no"]

# SIG relationship types whose members are mutually exclusive outcomes.
_EXCLUSIVE_RELATIONSHIP_TYPES = {"mutually_exclusive", "complementary"}

_SOLVER_MAX_ITERATIONS = 2000
_SOLVER_TOLERANCE = 1e-12
_MAX_TOTAL_STAKE = 0.999  # keep 1 + f.r strictly positive in every scenario


class Direction(str, Enum):
    """Which party a bet profits from - the shared 'national wave' risk factor."""

    PRO_REPUBLICAN = "R"
    PRO_DEMOCRAT = "D"
    NEUTRAL = "N"


# Thresholds were first set before any live data and tightened after day 1 (see below).
@dataclass(frozen=True)
class RiskLimits:
    kelly_fraction: float = 0.25  # fraction of full Kelly actually bet
    min_edge: float = 0.02  # ignore bets with less SHRUNK edge per share than this
    # Tightened after day 1 (2026-10-01): 5% / 10% let ~9.9% of bankroll into one thin
    # race (Rhode Island Senate, D YES + R NO at 0.95-0.97).
    max_fraction_per_contract: float = MAX_MARKET_FRAC  # of bankroll, per exchange (0.02)
    max_fraction_per_cluster: float = 0.03  # of bankroll, per risk cluster (was 0.10)
    max_total_fraction: float = TAKER_GROSS_FRAC  # of bankroll, across everything (default 0.30; was 0.60)
    # Races are sized independently, but a midterm "wave" moves them together, so cap
    # the total bet on each party across ALL races (tighter than the portfolio cap).
    max_directional_fraction: float = 0.15  # of bankroll, per Direction (R / D) (was 0.35) - "nominal" mode
    # "nominal" | "stress" (DIRECTIONAL_LIMIT_MODE): in "stress" mode the directional cap is
    # stress_directional_cap's - adverse-wave loss at directional_stress_s <= directional_stress_frac.
    directional_mode: str = DIRECTIONAL_LIMIT_MODE
    directional_stress_s: float = DIRECTIONAL_STRESS_S
    directional_stress_frac: float = DIRECTIONAL_STRESS_FRAC
    # Price-aware sizing and exposure gates - see config.py for the reasoning.
    max_buy_price: float = MAX_BUY_PRICE
    high_price_min_edge: float = HIGH_PRICE_MIN_EDGE
    max_race_worst_loss_fraction: float = MAX_RACE_WORST_LOSS_FRAC
    max_gross_exposure_fraction: float = MAX_GROSS_EXPOSURE_FRAC
    max_total_gross_exposure_fraction: float = TOTAL_GROSS_EXPOSURE_FRAC  # taker + ladder
    max_new_notional_per_cycle_fraction: float = MAX_NEW_NOTIONAL_PER_CYCLE_FRAC


@dataclass(frozen=True)
class Opportunity:
    """Input: our fair probability for YES and the prices we could buy at right now."""

    exchange_id: str
    fair_prob: float  # our estimate of P(YES), e.g. from oracle mids
    yes_price: float | None  # cost to buy 1 YES share now (best YES ask), None if no liquidity
    no_price: float | None  # cost to buy 1 NO share now (1 - best YES bid), None if no liquidity
    margin: float = 0.0  # uncertainty haircut on fair_prob (and on 1 - fair_prob for NO)


@dataclass(frozen=True)
class SizedBet:
    """Output: target position for one contract."""

    exchange_id: str
    side: Side
    price: float
    quantity: int  # shares
    stake: float  # quantity * price, in bankroll currency
    edge: float  # expected profit per share at `price`
    full_kelly: float  # full-Kelly bankroll fraction before any scaling or caps
    cluster: str
    direction: Direction = Direction.NEUTRAL  # filled in by size_portfolio


@dataclass(frozen=True)
class Cluster:
    key: str
    exchange_ids: frozenset[str]
    exclusive: bool  # True -> exact multi-outcome Kelly; False -> linked, capped


@dataclass(frozen=True)
class _Bet:
    exchange_id: str
    side: Side
    price: float
    win_prob: float  # fair probability this bet pays out (before the margin)
    margin: float = 0.0

    @property
    def edge(self) -> float:
        """Shrunk edge: (win_prob - margin) - price."""
        return self.win_prob - self.margin - self.price


def check_buy(price: float, win_prob: float, margin: float, limits: RiskLimits) -> tuple[bool, float, str]:
    """Price-aware buy filter. Returns (accepted, shrunk_edge, reason).

    p_used = win_prob - margin. No trade unless p_used - price >= min_edge, and above
    `max_buy_price` the shrunk edge must also reach `high_price_min_edge` - near-certain
    contracts have tiny upside and Kelly would otherwise max them out.
    """
    edge = win_prob - margin - price
    if not 0 < price < 1:
        return False, edge, "price outside (0, 1)"
    if edge <= 0:
        return False, edge, "no edge after margin"
    if edge < limits.min_edge:
        return False, edge, "shrunk edge below min_edge"
    if price > limits.max_buy_price and edge < limits.high_price_min_edge:
        return False, edge, "price above max_buy_price without high_price_min_edge"
    return True, edge, "ok"


# ---------------------------------------------------------------------- clusters


def build_clusters(market_map: MarketMap, relationships: Iterable[Mapping[str, Any]] = ()) -> list[Cluster]:
    """Union-find over race groups and SIG relationship edges.

    A cluster stays `exclusive` only if every link inside it says "mutually
    exclusive"; any other relationship type downgrades it to `linked`.
    """
    parent: dict[str, str] = {}
    exclusive_root: dict[str, bool] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        exclusive_root.setdefault(x, True)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(members: list[str], exclusive: bool) -> None:
        roots = {find(m) for m in members}
        combined = exclusive and all(exclusive_root[r] for r in roots)
        first, *rest = roots
        for root in rest:
            parent[root] = first
        exclusive_root[first] = combined

    for race_key, mappings in market_map.by_race().items():
        union([m.sig_exchange_id for m in mappings], exclusive=True)

    for relationship in relationships:
        if relationship.get("status", "active") != "active":
            continue
        members = [str(node["exchangeId"]) for node in relationship.get("nodes") or []]
        if len(members) >= 2:
            union(members, exclusive=relationship.get("type") in _EXCLUSIVE_RELATIONSHIP_TYPES)

    groups: dict[str, set[str]] = {}
    for exchange_id in list(parent):
        groups.setdefault(find(exchange_id), set()).add(exchange_id)

    race_key_of = {m.sig_exchange_id: m.race.key for m in market_map}
    clusters = []
    for root, members in groups.items():
        race_keys = sorted({race_key_of.get(m, f"exchange:{m}") for m in members})
        clusters.append(Cluster(key="+".join(race_keys), exchange_ids=frozenset(members), exclusive=exclusive_root[root]))
    return sorted(clusters, key=lambda c: c.key)


# ---------------------------------------------------------------------- Kelly maths


def kelly_single(win_prob: float, price: float) -> float:
    """Full-Kelly bankroll fraction for one binary bet paying 1 per share, bought at `price`."""
    if not 0 < price < 1 or win_prob <= price:
        return 0.0
    return (win_prob - price) / (1 - price)


def _project_to_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {f >= 0, sum(f) <= cap}."""
    clipped = np.clip(v, 0.0, None)
    if clipped.sum() <= cap:
        return clipped
    # Standard sort-based projection onto {f >= 0, sum(f) == cap}.
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u) - cap
    rho = np.nonzero(u - cumulative / np.arange(1, len(u) + 1) > 0)[0][-1]
    theta = cumulative[rho] / (rho + 1)
    return np.clip(v - theta, 0.0, None)


def kelly_scenarios(returns: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Maximise E[log(1 + f . r)] over f >= 0, sum(f) < 1.

    returns[j, s] = net return per unit staked on bet j if scenario s happens
    probs[s]      = probability of scenario s (sums to 1)
    Projected gradient ascent with backtracking; the objective is concave, so this
    converges to the global optimum. Problems here are tiny (<= ~10 bets).
    """
    n_bets = returns.shape[0]
    f = np.zeros(n_bets)

    def objective(x: np.ndarray) -> float:
        return float(probs @ np.log1p(x @ returns))

    value = objective(f)
    step = 1.0
    for _ in range(_SOLVER_MAX_ITERATIONS):
        wealth = 1.0 + f @ returns
        gradient = returns @ (probs / wealth)
        while step > 1e-12:
            candidate = _project_to_capped_simplex(f + step * gradient, _MAX_TOTAL_STAKE)
            candidate_value = objective(candidate)
            if candidate_value >= value:
                break
            step /= 2
        else:
            break
        improvement = candidate_value - value
        f, value = candidate, candidate_value
        step *= 2  # let the step grow again after a successful move
        if improvement < _SOLVER_TOLERANCE:
            break
    return f


def _candidate_bets(opportunity: Opportunity, limits: RiskLimits) -> list[_Bet]:
    q = min(max(opportunity.fair_prob, 0.0), 1.0)
    margin = max(opportunity.margin, 0.0)
    bets = []
    if opportunity.yes_price is not None:
        bets.append(_Bet(opportunity.exchange_id, "yes", opportunity.yes_price, q, margin))
    if opportunity.no_price is not None:
        bets.append(_Bet(opportunity.exchange_id, "no", opportunity.no_price, 1.0 - q, margin))
    return [b for b in bets if check_buy(b.price, b.win_prob, b.margin, limits)[0]]


def _full_kelly_exclusive(members: list[str], opportunities: Mapping[str, Opportunity], bets: list[_Bet]) -> list[float]:
    # Scenarios: "member i's YES happens" for each member, plus "none of them" (e.g. a
    # third party wins a race where SIG only lists D and R).
    probs = np.array([min(max(opportunities[m].fair_prob, 0.0), 1.0) for m in members])
    if probs.sum() > 1.0:
        probs = probs / probs.sum()  # oracle noise can push the sum slightly over 1
    probs = np.append(probs, max(0.0, 1.0 - probs.sum()))

    index = {m: i for i, m in enumerate(members)}
    returns = np.empty((len(bets), len(probs)))
    for j, bet in enumerate(bets):
        # The scenario probabilities are shared by all bets, so the margin cannot shrink
        # each bet's own win probability here. Charge it as extra price instead: same
        # shrunk edge (p - margin - price), slightly different Kelly denominator.
        cost = min(bet.price + bet.margin, 0.999)
        payout = (1.0 - cost) / cost
        wins = np.zeros(len(probs), dtype=bool)
        wins[index[bet.exchange_id]] = True
        if bet.side == "no":
            wins = ~wins
        returns[j] = np.where(wins, payout, -1.0)

    return list(kelly_scenarios(returns, probs))


def _full_kelly_linked(bets: list[_Bet]) -> list[float]:
    # Unknown joint distribution: size each bet alone, then treat the whole cluster as
    # one perfectly-correlated bet - the total never exceeds the best single bet.
    fractions = [kelly_single(b.win_prob - b.margin, b.price) for b in bets]
    total, best = sum(fractions), max(fractions, default=0.0)
    if total > best > 0:
        fractions = [f * best / total for f in fractions]
    return fractions


def size_cluster(
    cluster: Cluster, opportunities: Mapping[str, Opportunity], bankroll: float, limits: RiskLimits
) -> list[SizedBet]:
    members = sorted(m for m in cluster.exchange_ids if m in opportunities)
    bets = [bet for m in members for bet in _candidate_bets(opportunities[m], limits)]
    if not bets:
        return []

    if cluster.exclusive:
        full = _full_kelly_exclusive(members, opportunities, bets)
    else:
        full = _full_kelly_linked(bets)

    fractions = [min(f * limits.kelly_fraction, limits.max_fraction_per_contract) for f in full]
    cluster_total = sum(fractions)
    if cluster_total > limits.max_fraction_per_cluster:
        fractions = [f * limits.max_fraction_per_cluster / cluster_total for f in fractions]

    sized = []
    for bet, f_full, fraction in zip(bets, full, fractions):
        quantity = math.floor(fraction * bankroll / bet.price)
        if quantity <= 0:
            continue
        sized.append(
            SizedBet(
                exchange_id=bet.exchange_id,
                side=bet.side,
                price=bet.price,
                quantity=quantity,
                stake=round(quantity * bet.price, 6),
                edge=round(bet.edge, 6),
                full_kelly=round(float(f_full), 6),
                cluster=cluster.key,
            )
        )
    return sized


def _scale_bets(bets: list[SizedBet], scale: float) -> list[SizedBet]:
    scaled = []
    for bet in bets:
        quantity = math.floor(bet.quantity * scale)
        if quantity > 0:
            scaled.append(replace(bet, quantity=quantity, stake=round(quantity * bet.price, 6)))
    return scaled


def size_portfolio(
    clusters: Iterable[Cluster],
    opportunities: Mapping[str, Opportunity],
    bankroll: float,
    market_map: MarketMap,
    limits: RiskLimits = RiskLimits(),
    holdings: Mapping[str, tuple[Side, float]] | None = None,
    held_shares: Mapping[str, float] | None = None,
    report: dict[str, Any] | None = None,
) -> list[SizedBet]:
    """Size every cluster, then apply the cluster, directional and portfolio caps.

    Returned bets are TARGET positions for the exchanges in `opportunities`.
    `holdings`: EVERY held position, exchange_id -> (side, stake in bankroll currency).
    A held position whose exchange gets a target on the same side is replaced by that
    target (strategy.py only trades the difference). Every other held position is
    "uncovered": it stays on the books this cycle (no fair price, no edge, or about to
    be flipped), so it uses up room in the cluster, directional and portfolio caps.
    Opportunities for exchanges not in any cluster are sized as single-contract clusters.

    limits.directional_mode "nominal" (default): cluster -> directional (market value) ->
    portfolio. "stress": cluster -> portfolio -> stress_directional_cap, which needs
    `held_shares` (exchange_id -> shares held). `report` (optional) receives the stress
    cap's per-direction numbers.
    """
    holdings = holdings or {}
    clusters = list(clusters)
    clustered = {m for c in clusters for m in c.exchange_ids}
    clusters += [
        Cluster(key=f"exchange:{m}", exchange_ids=frozenset({m}), exclusive=True)
        for m in opportunities
        if m not in clustered
    ]

    sized: list[SizedBet] = []
    for cluster in clusters:
        bets = [
            replace(bet, direction=direction_of(market_map, bet.exchange_id, bet.side))
            for bet in size_cluster(cluster, opportunities, bankroll, limits)
        ]
        # Cluster cap, incl. held positions in this cluster that no target replaces.
        covered = {(b.exchange_id, b.side) for b in bets}
        uncovered = sum(
            stake
            for exchange_id, (side, stake) in holdings.items()
            if exchange_id in cluster.exchange_ids and (exchange_id, side) not in covered
        )
        room = max(0.0, limits.max_fraction_per_cluster * bankroll - uncovered)
        total = sum(b.stake for b in bets)
        if total > room:
            logger.info(
                "Cluster cap binding (held positions)",
                extra={"cluster": cluster.key, "wanted": round(total, 2), "held_uncovered": round(uncovered, 2), "allowed": round(room, 2)},
            )
            bets = _scale_bets(bets, room / total)
        sized.extend(bets)

    # Held positions no final target replaces count toward the directional and portfolio caps.
    covered = {(b.exchange_id, b.side) for b in sized}
    held_uncovered: dict[Direction, float] = defaultdict(float)
    for exchange_id, (side, stake) in holdings.items():
        if (exchange_id, side) not in covered:
            held_uncovered[direction_of(market_map, exchange_id, side)] += stake

    stress_mode = limits.directional_mode == "stress"
    if stress_mode:
        capped = list(sized)  # the stress directional cap runs after the portfolio cap
    else:
        # Directional cap: total bet on one party across all races.
        by_direction: dict[Direction, list[SizedBet]] = defaultdict(list)
        for bet in sized:
            by_direction[bet.direction].append(bet)
        capped = list(by_direction.pop(Direction.NEUTRAL, []))
        for direction, bets in by_direction.items():
            room = max(0.0, limits.max_directional_fraction * bankroll - held_uncovered.get(direction, 0.0))
            total = sum(b.stake for b in bets)
            if total > room:
                logger.info(
                    "Directional cap binding",
                    extra={"direction": direction.value, "wanted": round(total, 2), "allowed": round(room, 2)},
                )
                bets = _scale_bets(bets, room / total)
            capped.extend(bets)

    # Portfolio cap.
    room = max(0.0, limits.max_total_fraction * bankroll - sum(held_uncovered.values()))
    total = sum(b.stake for b in capped)
    if total > room:
        logger.info("Portfolio cap binding", extra={"wanted": round(total, 2), "allowed": round(room, 2)})
        capped = _scale_bets(capped, room / total)

    if stress_mode:
        capped, info = stress_directional_cap(capped, opportunities, market_map, bankroll, limits, holdings, held_shares or {})
        if report is not None:
            report.update(info)
    return capped


# ---------------------------------------------------------------------- national-wave stress
#
# One national wave moves every race together: a D-wave ("D wins" more often) shifts every D
# party's win-probability logit up by s and every R party's down by s; an R-wave the other way.
# A pro-D position (D YES, R NO) is hurt by the R-wave, a pro-R one by the D-wave. Shared by
# the "stress" directional limit below and scripts/directional_stress_preview.py /
# scripts/gross_cap_preview.py, so the preview and the bot cannot drift apart.

_P_CLAMP = 1e-6


def logit(p: float) -> float:
    p = min(max(p, _P_CLAMP), 1.0 - _P_CLAMP)
    return math.log(p / (1.0 - p))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def stressed_yes(p_yes: float, party: str, wave: str, s: float) -> float:
    """P(YES) of a `party` market after a `wave` wave ("D" or "R") of size s: the wave's party
    +s in logit, the other major party -s, anyone else unchanged."""
    if party not in ("D", "R"):
        return p_yes
    return logistic(logit(p_yes) + (s if party == wave else -s))


def side_loss_per_share(p_yes: float, party: str, side: Side, wave: str, s: float) -> float:
    """max(0, price now - price after the wave) of one share of `side`."""
    after = stressed_yes(p_yes, party, wave, s)
    now_side, after_side = (p_yes, after) if side == "yes" else (1.0 - p_yes, 1.0 - after)
    return max(0.0, now_side - after_side)


def adverse_wave(direction: str) -> str | None:
    """The wave that hurts a direction's positions: wave R (R wins) hurts D-directional ones."""
    return {"D": "R", "R": "D"}.get(direction)


def stress_directional_cap(
    bets: list[SizedBet],
    opportunities: Mapping[str, Opportunity],
    market_map: MarketMap,
    bankroll: float,
    limits: RiskLimits,
    holdings: Mapping[str, tuple[Side, float]],
    held_shares: Mapping[str, float],
) -> tuple[list[SizedBet], dict[str, Any]]:
    """Directional limit as stress: per direction, the adverse-wave loss (at
    limits.directional_stress_s) of the held positions no target replaces plus the
    direction's targets must stay within limits.directional_stress_frac of bankroll; else
    every target of that direction is scaled by one factor. Positions are priced at the
    oracle fair (`opportunities`), or at their mark (stake / shares) without one."""
    covered = {(b.exchange_id, b.side) for b in bets}
    s = limits.directional_stress_s
    budget = limits.directional_stress_frac * bankroll
    scaled = [b for b in bets if b.direction.value not in ("D", "R")]
    info: dict[str, Any] = {}
    for direction in ("D", "R"):
        wave = adverse_wave(direction)
        held_stress = 0.0
        for exchange_id, (side, stake) in holdings.items():
            mapping = market_map.get(exchange_id)
            shares = held_shares.get(exchange_id) or 0
            if (exchange_id, side) in covered or mapping is None or shares <= 0:
                continue
            if direction_of(market_map, exchange_id, side).value != direction:
                continue
            if exchange_id in opportunities:
                p_yes = opportunities[exchange_id].fair_prob
            else:
                mark = stake / shares
                p_yes = mark if side == "yes" else 1.0 - mark
            held_stress += shares * side_loss_per_share(p_yes, mapping.party, side, wave, s)
        mine = [b for b in bets if b.direction.value == direction]
        wanted = sum(
            b.quantity * side_loss_per_share(opportunities[b.exchange_id].fair_prob, market_map.get(b.exchange_id).party, b.side, wave, s)
            for b in mine
            if market_map.get(b.exchange_id) is not None
        )
        room = max(0.0, budget - held_stress)
        factor = 1.0 if wanted <= room or wanted == 0 else room / wanted
        if factor < 1.0:
            logger.info(
                "Directional cap binding (stress)",
                extra={
                    "positions": f"{direction}-directional", "adverse_wave": f"wave {wave} ({wave} wins)", "s": s,
                    "held_stress": round(held_stress, 2), "wanted_stress": round(wanted, 2), "budget": round(budget, 2), "scale": round(factor, 4),
                },
            )
        scaled.extend(_scale_bets(mine, factor) if factor < 1.0 else mine)
        info[direction] = {"held_uncovered_stress": round(held_stress, 2), "targets_stress": round(wanted, 2), "budget": round(budget, 2), "scale": round(factor, 4)}
    return scaled, info


# ---------------------------------------------------------------------- race worst case

NONE_OF_LISTED = "none of the listed parties"


@dataclass(frozen=True)
class RaceLeg:
    """One position (or planned order) inside a race, for scenario P&L."""

    party: str  # party of the SIG market the leg is on: D / R / I ...
    side: Side  # yes pays 1 if `party` wins the race; no pays 1 if it does not
    shares: float  # > 0 held or bought; < 0 sold
    cost: float  # cash paid (negative = sale proceeds)


def race_scenario_pnl(parties: Iterable[str], legs: Iterable[RaceLeg]) -> dict[str, float]:
    """Settlement P&L of all legs for each "winner" scenario: every listed party, plus none.

    Exactly one party wins a race, so D YES and R NO pay out in exactly the same
    scenarios - they are summed as the same bet. "None of the listed parties wins"
    is always included: the data cannot prove SIG lists every candidate, and an extra
    scenario can only make the worst case worse, never hide a loss.
    """
    legs = list(legs)
    winners = [*sorted(set(parties) | {leg.party for leg in legs}), NONE_OF_LISTED]
    pnl: dict[str, float] = {}
    for winner in winners:
        total = 0.0
        for leg in legs:
            pays = (leg.party == winner) == (leg.side == "yes")
            total += (leg.shares if pays else 0.0) - leg.cost
        pnl[winner] = total
    return pnl


def race_worst_case_loss(parties: Iterable[str], legs: Iterable[RaceLeg]) -> float:
    """Largest loss over all winner scenarios (0 if every scenario is profitable)."""
    return max(0.0, -min(race_scenario_pnl(parties, legs).values()))


def max_shares_within_race_cap(
    parties: Iterable[str],
    legs: Iterable[RaceLeg],
    party: str,
    side: Side,
    price: float,
    quantity: int,
    cap: float,
) -> int:
    """Largest q <= quantity such that buying q more shares keeps the race worst-case loss <= cap.

    Each scenario's P&L is linear in q (+(1 - price) per share where the new leg pays,
    -price where it does not), so the bound is exact, not a search. Returns 0 if no
    positive q fits. A hedging leg may be REQUIRED to reach a minimum size to pull an
    already-breaching scenario back under the cap; if that minimum exceeds the maximum,
    nothing fits.
    """
    if not 0 < price < 1 or quantity <= 0:
        return 0
    low, high = 0, quantity
    for winner, current in race_scenario_pnl(parties, legs).items():
        pays = (party == winner) == (side == "yes")
        per_share = (1.0 - price) if pays else -price
        if per_share < 0:
            high = min(high, math.floor((current + cap) / -per_share + 1e-9))
        elif current < -cap:
            low = max(low, math.ceil((-cap - current) / per_share - 1e-9))
    return high if high >= max(low, 1) else 0


def race_legs_from_holdings(market_map: MarketMap, holdings: Mapping[str, Any]) -> dict[str, list[RaceLeg]]:
    """Held positions (objects with side / shares / cost_basis) as RaceLegs at cost, grouped by
    race key. Unmapped exchanges are skipped."""
    race_legs: dict[str, list[RaceLeg]] = defaultdict(list)
    for exchange_id, holding in holdings.items():
        mapping = market_map.get(exchange_id)
        if mapping is None:
            logger.warning("Held position is not in the market map; ignored by race caps", extra={"exchange_id": exchange_id})
            continue
        race_legs[mapping.race.key].append(RaceLeg(mapping.party, holding.side, holding.shares, holding.cost_basis))
    return race_legs


# ---------------------------------------------------------------------- direction


def direction_of(market_map: MarketMap, exchange_id: str, side: Side) -> Direction:
    """YES on an R market or NO on a D market is pro-Republican, and vice versa.

    Independents (and anything unmapped) are NEUTRAL: they are not a bet on the
    national D-vs-R environment.
    """
    mapping = market_map.get(exchange_id)
    if mapping is None or mapping.party not in ("D", "R"):
        return Direction.NEUTRAL
    backs_party = side == "yes"
    if (mapping.party == "R") == backs_party:
        return Direction.PRO_REPUBLICAN
    return Direction.PRO_DEMOCRAT


def directional_exposure(
    market_map: MarketMap, holdings: Iterable[tuple[str, Side, float]]
) -> dict[Direction, float]:
    """Sum stake by direction. holdings: (exchange_id, side, stake) - e.g. from /portfolio/positions."""
    exposure: dict[Direction, float] = defaultdict(float)
    for exchange_id, side, stake in holdings:
        exposure[direction_of(market_map, exchange_id, side)] += stake
    return dict(exposure)


# ---------------------------------------------------------------------- tournament modifier


def leaderboard_standing(rank: int | None, total: int | None) -> float | None:
    """Fraction of the field at or ahead of us: rank 5 of 100 -> 0.05 (top 5%).

    From GET /tournaments/{slug}/leaderboard: `myRank` and `total`.
    """
    if not rank or not total:
        return None
    return min(rank / total, 1.0)


def variance_modifier(
    standing: float,
    target_standing: float = 0.10,
    *,
    max_boost: float = 1.5,
    min_scale: float = 0.5,
) -> float:
    """Kelly multiplier from leaderboard position; >1 below target, <1 at/above it.

    standing: fraction of the field at or ahead of us (0.05 = top 5%, 1.0 = last).
    Below target (standing > target): +1 per unit of gap, capped at max_boost -
        e.g. top 30% with a top-10% target -> 1.2x; bottom half -> 1.5x.
    At/above target: shrinks linearly to min_scale at 1st place -
        e.g. top 5% -> 0.75x, 1st place -> ~0.5x.
    """
    gap = standing - target_standing
    if gap > 0:
        return min(max_boost, 1.0 + gap)
    return max(min_scale, 1.0 + (1.0 - min_scale) * gap / target_standing)


def tournament_kelly_fraction(
    base: float,
    standing: float | None,
    time_remaining: float,
    *,
    target_standing: float = 0.10,  # top 10% of the field - realistic goal, not podium
    endgame: float = 0.2,
) -> float:
    """Apply `variance_modifier` gradually over the last `endgame` of the tournament.

    time_remaining: fraction of the trading window left (1.0 at start, 0.0 at end).
    Before the endgame the fraction is unchanged; inside it the modifier ramps in
    linearly, reaching full strength at the close. Never exceeds full Kelly.
    """
    if standing is None or time_remaining >= endgame:
        return base
    progress = 1.0 - max(time_remaining, 0.0) / endgame  # 0 at endgame start -> 1 at the end
    modifier = variance_modifier(standing, target_standing)
    return min(base * (1.0 + (modifier - 1.0) * progress), 1.0)


# ---------------------------------------------------------------------- sanity check


def _demo() -> None:
    import logging

    from config import load_settings
    from logging_config import setup_logging

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    log = logging.getLogger("risk.demo")
    limits = RiskLimits()
    bankroll = 100_000.0

    # 1. Solver agrees with the closed-form single-bet Kelly.
    q, p = 0.60, 0.50
    closed = kelly_single(q, p)
    numeric = kelly_scenarios(np.array([[(1 - p) / p, -1.0]]), np.array([q, 1 - q]))[0]
    log.info("Single bet: closed form vs solver", extra={"closed_form": round(closed, 4), "solver": round(float(numeric), 4)})

    # 2. Two-way race, SIG misprices D. YES-on-R and NO-on-D are nearly the same bet;
    #    the exclusive solver must NOT stack them as two independent full-size bets.
    race = Cluster("senate:XX", frozenset({"1", "2"}), exclusive=True)
    opportunities = {
        "1": Opportunity("1", fair_prob=0.62, yes_price=0.55, no_price=0.47),  # D: SIG too cheap
        "2": Opportunity("2", fair_prob=0.38, yes_price=0.47, no_price=0.55),  # R: SIG too dear
    }
    naive = sum(
        kelly_single(b.win_prob, b.price)
        for m in opportunities
        for b in _candidate_bets(opportunities[m], replace(limits, min_edge=0.0))
    )
    for bet in size_cluster(race, opportunities, bankroll, limits):
        log.info("Race bet", extra={k: v for k, v in bet.__dict__.items()})
    log.info("Naive independent full-Kelly total vs cluster-aware", extra={"naive_total_fraction": round(naive, 4)})

    # 3. Tournament modifier: below the top-10% target -> >1x, above it -> <1x.
    for rank, total, remaining in [(30, 100, 0.5), (30, 100, 0.0), (60, 100, 0.0), (10, 100, 0.0), (5, 100, 0.0), (1, 100, 0.0)]:
        standing = leaderboard_standing(rank, total)
        log.info(
            "Tournament Kelly fraction",
            extra={
                "rank": f"{rank}/{total}",
                "time_remaining": remaining,
                "modifier": round(variance_modifier(standing), 3),
                "fraction": round(tournament_kelly_fraction(0.25, standing, remaining), 4),
            },
        )

    # 4. Real clusters from the market map.
    market_map = MarketMap.load()
    clusters = build_clusters(market_map)
    log.info(
        "Clusters from market map",
        extra={
            "clusters": len(clusters),
            "exclusive": sum(c.exclusive for c in clusters),
            "largest": max(len(c.exchange_ids) for c in clusters),
            "sample": [(c.key, sorted(c.exchange_ids)) for c in clusters[:3]],
        },
    )

    # 5. Directional cap: pretend SIG underprices EVERY Democrat by 10c. Each race alone
    #    looks fine, but together it is one big bet on a blue wave - capped at 35%.
    wave = {
        m.sig_exchange_id: Opportunity(m.sig_exchange_id, fair_prob=0.60, yes_price=0.50, no_price=None)
        for m in market_map
        if m.party == "D"
    }
    bets = size_portfolio(clusters, wave, bankroll, market_map, limits)
    by_direction = directional_exposure(market_map, ((b.exchange_id, b.side, b.stake) for b in bets))
    log.info(
        "Blue-wave scenario",
        extra={
            "bets": len(bets),
            "stake_by_direction": {d.value: round(v, 2) for d, v in by_direction.items()},
            "cap": limits.max_directional_fraction * bankroll,
        },
    )


if __name__ == "__main__":
    # Sanity check: `python risk.py` - pure maths, no orders, no network.
    _demo()
