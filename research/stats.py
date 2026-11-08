"""Pure statistics for the research scripts: no I/O, no network, no clock.

Grows one script at a time; each function is unit-tested on synthetic data in
tests/test_research_stats.py.

Conventions:
- Prices are YES-normalised in [0, 1] (as SIG's tape and books report them).
- "Halves" always split observations by TIME order (first half vs second half), never at
  random, so the second half is out-of-sample for anything noticed in the first.
"""

from __future__ import annotations

import bisect
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

T = TypeVar("T")

PRICE_BUCKETS = ((0.15, "<0.15"), (0.85, "0.15-0.85"), (math.inf, ">0.85"))
SIZE_BUCKETS = ((10, "1-9"), (100, "10-99"), (500, "100-499"), (1000, "500-999"), (3000, "1000-2999"), (math.inf, "3000+"))
DEFAULT_PERCENTILES = (5, 10, 25, 50, 75, 90, 95, 99)


# ---------------------------------------------------------------------- distributions


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default) of an ascending sequence, q in [0, 100]."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def describe(values: Iterable[float], percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> dict[str, Any]:
    """count, sum, mean, min, max and the given percentiles (keys "p5", "p50", ...).
    An empty input gives count 0 and None everywhere else."""
    data = sorted(float(v) for v in values)
    out: dict[str, Any] = {"count": len(data)}
    if not data:
        return {**out, "sum": None, "mean": None, "min": None, "max": None, **{f"p{q:g}": None for q in percentiles}}
    total = math.fsum(data)
    return {
        **out,
        "sum": total,
        "mean": total / len(data),
        "min": data[0],
        "max": data[-1],
        **{f"p{q:g}": percentile(data, q) for q in percentiles},
    }


def bucket_label(value: float, buckets: Sequence[tuple[float, str]]) -> str:
    """Label of the first bucket whose upper bound (exclusive) is above `value`."""
    for upper, label in buckets:
        if value < upper:
            return label
    return buckets[-1][1]


def price_bucket(price: float) -> str:
    return bucket_label(price, PRICE_BUCKETS)


def size_bucket(size: float) -> str:
    return bucket_label(size, SIZE_BUCKETS)


def share_table(labels: Iterable[Hashable], order: Sequence[Hashable] | None = None) -> dict[str, dict[str, float]]:
    """{label: {"count", "share"}}; labels in `order` first (zeros kept), then the rest sorted."""
    counts = Counter(labels)
    total = sum(counts.values())
    keys = list(order or []) + sorted((k for k in counts if order is None or k not in order), key=str)
    return {str(k): {"count": counts.get(k, 0), "share": round(counts.get(k, 0) / total, 4) if total else 0.0} for k in keys}


# ---------------------------------------------------------------------- out-of-sample splits


def split_halves(rows: Sequence[T], key: Callable[[T], Any]) -> tuple[list[T], list[T]]:
    """Rows sorted by `key` (time), cut in two by count; an odd middle row goes to the first half."""
    ordered = sorted(rows, key=key)
    mid = (len(ordered) + 1) // 2
    return ordered[:mid], ordered[mid:]


def by_utc_day(rows: Iterable[T], ts: Callable[[T], datetime]) -> dict[str, list[T]]:
    """Rows grouped by the UTC date (YYYY-MM-DD) of `ts(row)`, days in ascending order."""
    groups: dict[str, list[T]] = defaultdict(list)
    for row in rows:
        groups[ts(row).strftime("%Y-%m-%d")].append(row)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------- leaderboard


def pnl_decomposition(pnl_all: float | None, pnl_7d: float | None, pnl_1d: float | None) -> dict[str, float | None]:
    """Split all-time P&L by calendar period (SIG periods are calendar-aligned UTC:
    7d = since the latest Sunday 00:00, 1d = since today 00:00):
    early = all - 7d, mid = 7d - 1d, today = 1d. A part is None when an input it needs is."""
    early = None if pnl_all is None or pnl_7d is None else pnl_all - pnl_7d
    mid = None if pnl_7d is None or pnl_1d is None else pnl_7d - pnl_1d
    return {"early": early, "mid": mid, "today": pnl_1d}


def zero_sum_check(pnls: Iterable[float]) -> dict[str, Any]:
    """Sum of P&L across participants and how it splits into winners / losers. A closed
    market valued consistently sums to ~0; a gap points at a counterparty outside the
    ranking (house liquidity, DMM) or at a valuation rule that is not consistent (e.g.
    every position marked at its bid)."""
    values = [float(v) for v in pnls]
    winners = [v for v in values if v > 0]
    losers = [v for v in values if v < 0]
    total = math.fsum(values)
    gross = math.fsum(winners) - math.fsum(losers)
    return {
        "participants": len(values),
        "sum_pnl": total,
        "sum_winners": math.fsum(winners),
        "sum_losers": math.fsum(losers),
        "winners": len(winners),
        "losers": len(losers),
        "flat": len(values) - len(winners) - len(losers),
        "sum_over_gross": total / gross if gross else None,  # -1..1; ~0 = zero-sum
    }


# ---------------------------------------------------------------------- tape


def tick_rule(prices: Sequence[float], eps: float = 1e-9) -> list[str | None]:
    """Aggressor PROXY per trade (prices in time order): "buy" on an uptick, "sell" on a
    downtick, a zero tick repeats the last non-zero call, None until the first price change.
    YES-normalised prices -> "buy" = a YES buyer (or NO seller) hit the ask. Only a proxy:
    the SIG tape carries no aggressor field."""
    calls: list[str | None] = []
    last_price: float | None = None
    last_call: str | None = None
    for price in prices:
        if last_price is not None:
            if price > last_price + eps:
                last_call = "buy"
            elif price < last_price - eps:
                last_call = "sell"
        calls.append(last_call)
        last_price = price
    return calls


# ---------------------------------------------------------------------- order books


def large_levels(book: Mapping[str, Any], min_size: float) -> list[dict[str, Any]]:
    """Resting levels with quantity >= min_size, on both sides of a YES-normalised book
    (`bids` / `asks` as [{"price", "quantity"}]). Each: side, price, quantity, rank (0 = best
    level) and distance from that side's best price (>= 0)."""
    found = []
    for side in ("bids", "asks"):
        levels = book.get(side) or []
        if not levels:
            continue
        best = float(levels[0]["price"])
        for rank, level in enumerate(levels):
            quantity = abs(float(level.get("quantity") or 0.0))
            if quantity >= min_size:
                price = float(level["price"])
                found.append({"side": side[:-1], "price": price, "quantity": quantity, "rank": rank, "distance": round(abs(price - best), 6)})
    return found


# ---------------------------------------------------------------------- valuation


def position_value(quantity: float, yes_bid: float | None, yes_ask: float | None, rule: str) -> float | None:
    """Value of `quantity` shares (> 0 YES, < 0 NO) from YES-normalised quotes.
    rule "bid": what selling now would get (YES at the YES bid, NO at 1 - YES ask);
    rule "mid": YES at (bid + ask) / 2, NO at 1 - that. None when a needed quote is missing."""
    if quantity == 0:
        return 0.0
    if rule == "bid":
        if quantity > 0:
            return None if yes_bid is None else quantity * yes_bid
        return None if yes_ask is None else -quantity * (1.0 - yes_ask)
    if rule == "mid":
        if yes_bid is None or yes_ask is None:
            return None
        mid = (yes_bid + yes_ask) / 2.0
        return quantity * mid if quantity > 0 else -quantity * (1.0 - mid)
    raise ValueError(f"unknown valuation rule {rule!r}")


def mark_denomination(quantity: float, current_price: float | None, market_value: float | None, tol: float = 1e-4) -> str | None:
    """Is a position's `currentPrice` YES-normalised or in the held side's terms? Decided by
    which one reproduces marketValue / |quantity|. Only a NO position can tell them apart:
    a YES position gives "ambiguous"; None when nothing fits or data is missing."""
    if not quantity or current_price is None or market_value is None:
        return None
    unit = abs(market_value) / abs(quantity)
    yes_fit = abs(unit - (current_price if quantity > 0 else 1.0 - current_price)) <= tol
    side_fit = abs(unit - current_price) <= tol
    if quantity > 0:
        return "ambiguous" if yes_fit else None
    if yes_fit and side_fit:  # currentPrice = 0.5
        return "ambiguous"
    return "yes" if yes_fit else ("side" if side_fit else None)


def classify_mark(value: float, bid: float | None, ask: float | None, last: float | None, tol: float = 1e-4) -> dict[str, Any]:
    """Which observable YES price a valuation mark equals (within `tol`): any of "last",
    "mid", "bid", "ask" (several when they coincide), and where it sits vs the spread."""
    mid = None if bid is None or ask is None else (bid + ask) / 2.0
    refs = {"last": last, "mid": mid, "bid": bid, "ask": ask}
    matches = [name for name, ref in refs.items() if ref is not None and abs(value - ref) <= tol]
    if bid is None or ask is None:
        position = "no two-sided quote"
    elif value < bid - tol:
        position = "below bid"
    elif value > ask + tol:
        position = "above ask"
    else:
        position = "inside spread"
    return {"matches": matches, "vs_spread": position, **{f"diff_{name}": (None if ref is None else value - ref) for name, ref in refs.items()}}


def closest_rule(target: float, candidates: Mapping[str, float | None]) -> tuple[str | None, dict[str, float | None]]:
    """Which candidate value is closest to `target`, and every residual (candidate - target).
    Candidates that are None (could not be valued) never win."""
    residuals = {name: (None if value is None else value - target) for name, value in candidates.items()}
    ranked = sorted((abs(r), name) for name, r in residuals.items() if r is not None)
    return (ranked[0][1] if ranked else None), residuals


# ---------------------------------------------------------------------- episodes (relationship probe)


def median_or_none(values: Iterable[float | None]) -> float | None:
    data = sorted(float(v) for v in values if v is not None)
    return None if not data else percentile(data, 50)


def episodes(times: Sequence[float], flags: Sequence[bool], max_gap_s: float) -> list[dict[str, Any]]:
    """Runs of consecutive flagged samples. `times` ascending (seconds). A run breaks on an
    unflagged sample or a gap > max_gap_s (a missed sample). Duration is a LOWER bound:
    last - first sighting, so a single sighting lasts 0 s (it may have lasted up to about
    two sample intervals - we cannot tell)."""
    if len(times) != len(flags):
        raise ValueError("times and flags differ in length")
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous: float | None = None
    for t, flag in zip(times, flags):
        gap_ok = previous is not None and t - previous <= max_gap_s
        if flag and current is not None and gap_ok:
            current["end"], current["samples"] = t, current["samples"] + 1
        elif flag:
            current = {"start": t, "end": t, "samples": 1}
            runs.append(current)
        else:
            current = None
        previous = t
    for run in runs:
        run["duration_s"] = run["end"] - run["start"]
    return runs


def premium_deviation_shares(excess: Sequence[float | None], premium: float | None, thresholds: Sequence[float], direction: int) -> dict[str, float | None]:
    """Share of samples whose excess sits at least `t` past the persistent premium (the
    median), for each FIXED threshold t - no z-scores. direction +1: excess >= premium + t
    (bids unusually rich); -1: excess <= premium - t (asks unusually cheap)."""
    data = [x for x in excess if x is not None]
    out: dict[str, float | None] = {}
    for t in thresholds:
        key = f"{t:g}"
        if premium is None or not data:
            out[key] = None
            continue
        hits = sum(1 for x in data if (x - premium >= t - 1e-9 if direction > 0 else premium - x >= t - 1e-9))
        out[key] = round(hits / len(data), 4)
    return out


# ---------------------------------------------------------------------- tape buckets (opening analysis)


def vwap(trades: Iterable[tuple[float, float]]) -> float | None:
    """Volume-weighted average of (price, size) pairs; None without volume."""
    pairs = [(float(p), float(s)) for p, s in trades if p is not None and s]
    volume = math.fsum(s for _, s in pairs)
    return math.fsum(p * s for p, s in pairs) / volume if volume else None


def bucket_index(t: datetime, start: datetime, minutes: int, buckets: int) -> int | None:
    """Index of the `minutes`-wide bucket holding `t` in [start, start + buckets * minutes);
    None outside the window."""
    offset = (t - start).total_seconds()
    if offset < 0:
        return None
    index = int(offset // (minutes * 60))
    return index if index < buckets else None


def share_outside(prices: Iterable[float], low: float, high: float) -> float | None:
    """Share of prices strictly below `low` or strictly above `high`; None when empty."""
    data = [float(p) for p in prices]
    return round(sum(1 for p in data if p < low or p > high) / len(data), 4) if data else None


# ---------------------------------------------------------------------- tape patterns (D)


def size_concentration(sizes: Iterable[float], min_count: int = 20, min_share: float = 0.05) -> list[dict[str, Any]]:
    """Exact sizes with count >= min_count AND share >= min_share, most frequent first."""
    data = [float(s) for s in sizes]
    counts = Counter(data)
    total = len(data)
    found = [{"size": s, "count": n, "share": round(n / total, 4)} for s, n in counts.items() if n >= min_count and n / total >= min_share]
    return sorted(found, key=lambda r: (-r["count"], r["size"]))


def is_round_notional(notional: float, multiple: float, tol: float = 0.005) -> bool:
    """Within tol (relative) of a non-zero multiple of `multiple`."""
    k = round(notional / multiple)
    return k >= 1 and abs(notional - k * multiple) <= tol * k * multiple


def expected_round_share(notionals: Iterable[float], multiple: float, tol: float = 0.005) -> float | None:
    """Share of `notionals` expected to look round by chance if notionals were locally
    uniform: near k * multiple the tolerance band is 2 * tol * k * multiple wide out of
    every `multiple`, so the chance is min(1, 2 * tol * k). Large notionals look round
    more often by construction - compare observed shares with this, not with 0."""
    data = [float(n) for n in notionals]
    if not data:
        return None
    return math.fsum(min(1.0, 2 * tol * round(n / multiple)) if round(n / multiple) >= 1 else 0.0 for n in data) / len(data)


def gap_spikes(gaps_s: Iterable[int], min_count: int = 20, ratio: float = 3.0, window: int = 5) -> list[dict[str, Any]]:
    """Inter-arrival gaps (whole seconds, >= 1: same-second prints are one sweep) that are
    abnormally frequent: count >= min_count and >= ratio * a baseline = the larger of the
    median count within +-window seconds (zeros included, the gap itself excluded) and the
    mean of the two adjacent seconds - so a smooth 4 / 5 / 6 s cluster is not a spike."""
    counts = Counter(int(g) for g in gaps_s if g >= 1)
    found = []
    for gap, n in counts.items():
        if n < min_count:
            continue
        neighbours = [counts.get(gap + d, 0) for d in range(-window, window + 1) if d != 0 and gap + d >= 1]
        adjacent = [counts.get(gap + d, 0) for d in (-1, 1) if gap + d >= 1]
        base = max(statistics.median(neighbours) if neighbours else 0.0, statistics.fmean(adjacent) if adjacent else 0.0)
        if n >= ratio * max(base, 1.0):
            found.append({"gap_s": gap, "count": n, "neighbour_median": base})
    return sorted(found, key=lambda r: -r["count"])


def poisson_binomial_tail(probs: Sequence[float], k: int) -> float:
    """P(X >= k) for X = a sum of independent Bernoulli(probs)."""
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            nxt[i] += q * (1 - p)
            nxt[i + 1] += q * p
        dist = nxt
    return math.fsum(dist[k:]) if k < len(dist) else 0.0


def recurring_minutes(times: Sequence[datetime], alpha: float = 0.05) -> dict[str, Any]:
    """Minutes of the UTC day that see trades on more distinct days than chance allows.
    Null (diurnal-aware): a day's n_d trades fall into hour h with that hour's pooled share
    f_h and uniformly over its 60 minutes, so minute m of hour h is hit on day d with
    p_d = 1 - (1 - f_h / 60)^n_d. A minute is flagged when P(days hit >= observed) * 1440
    < alpha (Bonferroni over the minutes of the day) and it is hit on >= 2 days."""
    per_day: dict[str, set[int]] = defaultdict(set)
    counts: Counter[str] = Counter()
    hours: Counter[int] = Counter()
    for t in times:
        day = t.strftime("%Y-%m-%d")
        per_day[day].add(t.hour * 60 + t.minute)
        counts[day] += 1
        hours[t.hour] += 1
    total = sum(hours.values())
    days_hit: Counter[int] = Counter(m for minutes in per_day.values() for m in minutes)
    found = []
    for minute, hit in days_hit.items():
        if hit < 2:
            continue
        share = hours[minute // 60] / total / 60
        probs = [1 - (1 - share) ** n for n in counts.values()]
        p_value = poisson_binomial_tail(probs, hit)
        if p_value * 1440 < alpha:
            found.append({"minute": f"{minute // 60:02d}:{minute % 60:02d}", "days": hit, "p_value": p_value})
    return {"days": len(counts), "minutes": sorted(found, key=lambda r: (r["p_value"], r["minute"]))}


def near_running_day_extreme(times: Sequence[datetime], prices: Sequence[float], tick: float = 0.005) -> list[bool | None]:
    """Per trade (time order): within one tick of the running UTC-day min or max price,
    the trade itself included. None for each day's first trade (trivially both)."""
    flags: list[bool | None] = []
    day = None
    lo = hi = 0.0
    for t, p in zip(times, prices):
        d = t.strftime("%Y-%m-%d")
        if d != day:
            day, lo, hi = d, p, p
            flags.append(None)
            continue
        lo, hi = min(lo, p), max(hi, p)
        flags.append(p <= lo + tick + 1e-9 or p >= hi - tick - 1e-9)
    return flags


def drift_after(times: Sequence[float], prices: Sequence[float], index: int, horizon_s: float) -> float | None:
    """Tape-price PROXY: last trade price at or before times[index] + horizon_s minus the
    trade's own price. None when the data ends before the horizon (cannot be known)."""
    target = times[index] + horizon_s
    if target > times[-1]:
        return None
    j = index
    while j + 1 < len(times) and times[j + 1] <= target:
        j += 1
    return prices[j] - prices[index]


def t_stat(values: Iterable[float]) -> float | None:
    """One-sample t of the mean against 0; None for n < 2 or zero spread."""
    data = [float(v) for v in values]
    if len(data) < 2:
        return None
    sd = statistics.stdev(data)
    return None if sd <= 1e-12 else statistics.fmean(data) / (sd / math.sqrt(len(data)))  # float noise is no spread


def welch_t(a: Iterable[float], b: Iterable[float]) -> float | None:
    """Welch t of mean(a) - mean(b); None when either side has < 2 values or no spread."""
    xa, xb = [float(v) for v in a], [float(v) for v in b]
    if len(xa) < 2 or len(xb) < 2:
        return None
    se = math.sqrt(statistics.variance(xa) / len(xa) + statistics.variance(xb) / len(xb))
    return None if se <= 1e-12 else (statistics.fmean(xa) - statistics.fmean(xb)) / se


# ---------------------------------------------------------------------- dry-run replay


def dedupe_repeats(rows: Sequence[T], key: Callable[[T], Hashable], when: Callable[[T], float], window_s: float, chained: bool = False) -> list[dict[str, Any]]:
    """Collapse repeats of the same intent. Per key, in time order, the first row opens a
    cluster; a later row joins it while it is within `window_s` of the cluster START
    (anchored) or of the cluster's LAST row (chained: an intent repeated every few seconds
    for hours stays one cluster). -> [{"row": first row, "repeats": n, "last": t}] by time."""
    clusters: list[dict[str, Any]] = []
    open_by_key: dict[Hashable, dict[str, Any]] = {}
    for row in sorted(rows, key=when):
        k, t = key(row), when(row)
        cluster = open_by_key.get(k)
        ref = None if cluster is None else (cluster["last"] if chained else cluster["start"])
        if cluster is not None and t - ref < window_s:
            cluster["repeats"] += 1
            cluster["last"] = t
            continue
        cluster = {"row": row, "repeats": 1, "start": t, "last": t}
        open_by_key[k] = cluster
        clusters.append(cluster)
    return clusters


def pnl_vs_mark(action: str, quantity: float, price_side: float, mark_side: float | None) -> float | None:
    """P&L of a fill marked at `mark_side`, both in the traded side's terms: a buy gains
    mark - price per share, a sell price - mark. None without a mark."""
    if mark_side is None:
        return None
    per_share = mark_side - price_side if action == "buy" else price_side - mark_side
    return abs(quantity) * per_share


# ---------------------------------------------------------------------- opening opportunity (proxies)


def deviation_volume(trades: Iterable[tuple[float, float]], reference: float | None) -> float | None:
    """PROXY of what liquidity provision could have earned: sum of size * |price - reference|
    over (price, size) prints. Not a P&L - the tape has no aggressor and no account id, so
    it cannot say who was on either side or whether a quote would have been hit."""
    if reference is None:
        return None
    return math.fsum(abs(float(s)) * abs(float(p) - reference) for p, s in trades if p is not None)


def nearest_pairs(a: Sequence[tuple[Any, ...]], b: Sequence[tuple[Any, ...]], max_gap_s: float) -> list[dict[str, Any]]:
    """For every print in `a` - a tuple (time_s, price, ...extra) - the nearest print in `b`
    within max_gap_s (b prints may be reused; ties -> the earlier one).
    -> [{t_a, t_b, price_a, price_b, gap_s, sum, item_a, item_b}] in `a`'s time order."""
    bs = sorted(b, key=lambda x: x[0])
    times = [x[0] for x in bs]
    out = []
    for item in sorted(a, key=lambda x: x[0]):
        t, p = item[0], item[1]
        k = bisect.bisect_left(times, t)
        best = None
        for j in (k - 1, k):
            if 0 <= j < len(bs) and abs(bs[j][0] - t) <= max_gap_s and (best is None or abs(bs[j][0] - t) < abs(bs[best][0] - t)):
                best = j
        if best is not None:
            other = bs[best]
            out.append({"t_a": t, "t_b": other[0], "price_a": p, "price_b": other[1], "gap_s": abs(other[0] - t), "sum": p + other[1],
                        "item_a": item, "item_b": other})
    return out


# ---------------------------------------------------------------------- SIG vs oracle wording (keyword check, no NLP)

PARTY_WORDS = {"D": ("democrat", "democrats", "democratic"), "R": ("republican", "republicans", "gop"), "I": ("independent", "independents")}
OFFICE_WORDS = {"senate": ("senate", "senator"), "governor": ("governor", "gubernatorial"), "house": ("house", "congressional", "representatives")}
CRITERION_PHRASES = ("popular vote", "margin", "more than", "at least", "seats", "majority", "nominee", "nomination", "primary", "runoff", "special", "control", "recount")
US_STATES = (
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
)
_NAME_STOPWORDS = {
    "will", "the", "who", "which", "what", "when", "win", "wins", "party", "parties", "race", "election", "elections", "midterm", "midterms", "us", "u", "s",
    "united", "states", "state", "district", "seat", "candidate", "yes", "no", "general", "november", "october", "january", "market", "contract",
    "freeform", "election outcome", "outcome", "for", "in", "of", "and", "or", "a", "an", "be", "by", "to", "on", "at", "is",
}


def _norm_text(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def text_keywords(text: str) -> dict[str, set[str]]:
    """Parties, offices, states, criterion phrases and capitalised name-like words of `text`
    (slugs and tickers are split on non-letters first). Simple keyword matching, no NLP."""
    norm = _norm_text(text)
    has = lambda word: f" {word} " in norm  # noqa: E731
    known = {w for words in (*PARTY_WORDS.values(), *OFFICE_WORDS.values()) for w in words} | {w for s in US_STATES for w in s.split()} | _NAME_STOPWORDS
    names = {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text) if w.lower() not in known}
    return {
        "parties": {p for p, words in PARTY_WORDS.items() if any(has(w) for w in words)},
        "offices": {o for o, words in OFFICE_WORDS.items() if any(has(w) for w in words)},
        "states": {s for s in US_STATES if has(s)},
        "criteria": {c for c in CRITERION_PHRASES if has(c)},
        "names": names,
    }


def wording_mismatch(sig_text: str, oracle_text: str, sig_party: str | None = None) -> list[str]:
    """Reasons the SIG title / rules and the oracle's title may describe different bets:
    a different party (the mapped SIG party counts as SIG's), office or state when both
    sides name one; disjoint candidate names when both sides name someone; a criterion
    phrase on one side only. Empty list = nothing flagged."""
    sig, ora = text_keywords(sig_text), text_keywords(oracle_text)
    sig_parties = sig["parties"] | ({sig_party} if sig_party else set())
    reasons = []
    for field, label in (("parties", "party"), ("offices", "office"), ("states", "state")):
        left = sig_parties if field == "parties" else sig[field]
        if left and ora[field] and not left & ora[field]:
            reasons.append(f"{label}: SIG {sorted(left)} vs oracle {sorted(ora[field])}")
    if sig["names"] and ora["names"] and not sig["names"] & ora["names"]:
        reasons.append(f"candidate: SIG {sorted(sig['names'])} vs oracle {sorted(ora['names'])}")
    only_sig, only_ora = sig["criteria"] - ora["criteria"], ora["criteria"] - sig["criteria"]
    if only_sig or only_ora:
        reasons.append(f"criterion: SIG only {sorted(only_sig)}, oracle only {sorted(only_ora)}")
    return reasons


# ---------------------------------------------------------------------- calibration (dry-run replay)

EDGE_BINS = ((0.02, "<0.02"), (0.04, "0.02-0.04"), (0.07, "0.04-0.07"), (math.inf, ">=0.07"))
_CLAIMED_EDGE_RE = re.compile(r"\bedge (-?\d+(?:\.\d+)?)")


def parse_claimed_edge(reason: str | None) -> float | None:
    """The edge the bot claimed in a journal reason ("edge 0.069; fitted to book ...")."""
    match = _CLAIMED_EDGE_RE.search(reason or "")
    return float(match.group(1)) if match else None


def edge_bin(edge: float | None) -> str | None:
    return None if edge is None else bucket_label(edge, EDGE_BINS)


def clustered_t(values_by_cluster: Mapping[Hashable, float]) -> tuple[float | None, int]:
    """t of the mean of per-cluster values (one value per cluster, e.g. a race's average
    markout): fills inside one race share its outcome and are not independent, so the
    clusters - not the fills - are the observations. -> (t, number of clusters)."""
    values = [float(v) for v in values_by_cluster.values() if v is not None]
    return t_stat(values), len(values)


def markout_table(fills: Sequence[Mapping[str, Any]], group: Callable[[Mapping[str, Any]], str | None], order: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Per group of fills ({"shares", "pnl", "cluster"}; pnl None = unvalued, skipped):
    fills, shares, clusters, share-weighted mean markout per share (sum pnl / sum shares),
    share of winners (fills with pnl > 0) and the CLUSTER-level t (per-cluster share-weighted
    means). Groups in `order` first (empty ones kept), then the rest sorted; None skipped."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for f in fills:
        if f.get("pnl") is None or not f.get("shares"):
            continue
        key = group(f)
        if key is not None:
            groups[key].append(f)
    keys = list(order) + sorted(k for k in groups if k not in order)
    rows = []
    for key in keys:
        members = groups.get(key, [])
        shares = math.fsum(abs(float(f["shares"])) for f in members)
        by_cluster: dict[Hashable, list[Mapping[str, Any]]] = defaultdict(list)
        for f in members:
            by_cluster[f.get("cluster")].append(f)
        cluster_means = {c: math.fsum(float(f["pnl"]) for f in fs) / math.fsum(abs(float(f["shares"])) for f in fs) for c, fs in by_cluster.items()}
        t, n = clustered_t(cluster_means)
        rows.append({"group": key, "fills": len(members), "shares": shares, "clusters": n,
                     "mean_markout": None if not shares else math.fsum(float(f["pnl"]) for f in members) / shares,
                     "winners_share": round(sum(1 for f in members if float(f["pnl"]) > 0) / len(members), 4) if members else None,
                     "t_clustered": t})
    return rows


def rises(values: Sequence[float | None]) -> bool | None:
    """Do the non-missing values never fall from one to the next (in the given order)?
    None with fewer than two values."""
    data = [v for v in values if v is not None]
    if len(data) < 2:
        return None
    return all(b >= a for a, b in zip(data, data[1:]))


def loss_concentration(pnl_by_group: Mapping[str, float], worst: int = 5) -> dict[str, Any]:
    """How much of the loss the `worst` groups carry: their P&L sum, its share of the net
    P&L (when the net is a loss) and of the gross losses (sum of the losing groups)."""
    ranked = sorted(pnl_by_group.items(), key=lambda kv: kv[1])
    top = ranked[:worst]
    worst_sum = math.fsum(v for _, v in top)
    net = math.fsum(pnl_by_group.values())
    gross_losses = math.fsum(v for v in pnl_by_group.values() if v < 0)
    return {"worst": [k for k, _ in top], "worst_pnl": worst_sum, "net_pnl": net, "gross_losses": gross_losses,
            "share_of_net_loss": round(worst_sum / net, 4) if net < 0 else None,
            "share_of_gross_losses": round(math.fsum(v for _, v in top if v < 0) / gross_losses, 4) if gross_losses < 0 else None}


# ---------------------------------------------------------------------- party-winner wording, carry, direction

_PM_PARTY_SUFFIX_RE = re.compile(r"\(\s*([DRI])\s*\)\s*$")
_KALSHI_PARTY_SUFFIX_RE = re.compile(r"-([DRI])$")


def oracle_parties(polymarket_label: str | None, kalshi_ticker: str | None) -> dict[str, str | None]:
    """The party each oracle names: Polymarket label suffix "(D)" (or a party word in the
    label, e.g. "Democratic Party"), Kalshi ticker suffix "-D". None where absent."""
    pm = None
    if polymarket_label:
        match = _PM_PARTY_SUFFIX_RE.search(polymarket_label.strip())
        words = text_keywords(polymarket_label)["parties"]
        pm = match.group(1) if match else (next(iter(words)) if len(words) == 1 else None)
    ks = None
    if kalshi_ticker:
        match = _KALSHI_PARTY_SUFFIX_RE.search(kalshi_ticker.strip().upper())
        ks = match.group(1) if match else None
    return {"polymarket": pm, "kalshi": ks}


def party_winner_mismatch(sig_party: str | None, parties: Mapping[str, str | None], linked: Mapping[str, bool]) -> list[str]:
    """Party-winner contracts: compare ONLY the party. For each linked oracle (linked[source]
    True), flag a party that differs from SIG's or that cannot be read. Never names."""
    reasons = []
    for source in ("polymarket", "kalshi"):
        if not linked.get(source):
            continue
        party = parties.get(source)
        if party is None:
            reasons.append(f"party: {source} names no party")
        elif party != sig_party:
            reasons.append(f"party: SIG {sig_party} vs {source} {party}")
    return reasons


def carry_pnl(shares: float, p_side: float | None, avg_cost: float | None) -> float | None:
    """HYPOTHETICAL expected P&L at settlement: shares * (p_side - avg_cost), p_side the
    probability (an independent reference, not truth) that the held side pays 1."""
    if p_side is None or avg_cost is None:
        return None
    return abs(shares) * (p_side - avg_cost)


def leg_direction(party: str | None, side: str | None, race_parties: Iterable[str]) -> str | None:
    """Which winner a leg bets on: YES on X -> "X"; NO on X in a two-party race -> the other
    party (YES-D and NO-R are the same bet); NO on X with more parties -> "not X"."""
    if party is None or side is None:
        return None
    if side == "yes":
        return party
    others = sorted(set(race_parties) - {party})
    return others[0] if len(others) == 1 else f"not {party}"


def race_direction(directions: Iterable[str | None]) -> str:
    """"long X" when every leg bets on the same winner X, "mixed" otherwise, "flat" without legs."""
    found = {d for d in directions if d is not None}
    if not found:
        return "flat"
    return f"long {found.pop()}" if len(found) == 1 else "mixed"


# ---------------------------------------------------------------------- opening convergence (hourly closes, PROXY)


def candle_start(candle: Mapping[str, Any]) -> datetime:
    """A candle's bucket start as an aware UTC datetime. Accepts ISO-8601 text (with or
    without "Z" / milliseconds / offset; naive = UTC), epoch seconds or epoch milliseconds."""
    raw = candle["time"]
    if isinstance(raw, datetime):
        t = raw
    elif isinstance(raw, (int, float)) or (isinstance(raw, str) and re.fullmatch(r"\d+(\.\d+)?", raw.strip())):
        value = float(raw)
        t = datetime.fromtimestamp(value / 1000 if value > 1e11 else value, tz=timezone.utc)
    else:
        t = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


_candle_start = candle_start  # kept for the callers below


def hour_coverage(candles_by_exchange: Mapping[str, Sequence[Mapping[str, Any]]], hours: Sequence[datetime]) -> dict[str, int]:
    """For each hour: how many exchanges have a candle STARTING at it (sparse series:
    an hour without trades has no candle)."""
    starts = {eid: {candle_start(c) for c in cs} for eid, cs in candles_by_exchange.items()}
    return {h.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"): sum(1 for s in starts.values() if h in s) for h in hours}


def first_candle_from(candles: Sequence[Mapping[str, Any]], start: datetime) -> Mapping[str, Any] | None:
    """The earliest candle (with a close) whose bucket starts at or after `start`."""
    found = sorted((c for c in candles if c.get("close") is not None and _candle_start(c) >= start), key=_candle_start)
    return found[0] if found else None


def close_as_of(candles: Sequence[Mapping[str, Any]], at: datetime, width: timedelta = timedelta(hours=1)) -> tuple[float | None, datetime | None]:
    """The last close known at `at`: the close of the latest candle that ENDED at or before
    `at` (start + width <= at). Sparse series: an hour without trades keeps the previous
    close. -> (close, that candle's start)."""
    done = [c for c in candles if c.get("close") is not None and _candle_start(c) + width <= at]
    if not done:
        return None, None
    last = max(done, key=_candle_start)
    return float(last["close"]), _candle_start(last)


def mis_initialised(first_close: float | None, oracle: float | None, band: tuple[float, float] = (0.40, 0.60), extremes: tuple[float, float] = (0.15, 0.85)) -> bool:
    """Opened near a coin flip while the oracle said it was nearly decided."""
    if first_close is None or oracle is None:
        return False
    return band[0] <= first_close <= band[1] and (oracle < extremes[0] or oracle > extremes[1])


def move_toward(start: float | None, later: float | None, target: float | None) -> dict[str, float | None]:
    """Price move start -> later, its component toward `target` (+ toward, - away), and the
    share of the start gap |target - start| it closed (1 = all the way, < 0 = moved away)."""
    if start is None or later is None or target is None:
        return {"move": None, "toward": None, "gap": None, "gap_closed": None}
    move = later - start
    gap = target - start
    direction = 1.0 if gap > 0 else -1.0 if gap < 0 else 0.0
    toward = move * direction
    return {"move": move, "toward": toward, "gap": abs(gap), "gap_closed": None if gap == 0 else toward / abs(gap)}


# ---------------------------------------------------------------------- tail prints (PROXY)

TAIL_LEVELS = ((0.05, "<0.05"), (0.15, "0.05-0.15"), (0.85, "0.15-0.85"), (0.95, "0.85-0.95"), (math.inf, ">0.95"))


def tail_level(price: float) -> str:
    return bucket_label(price, TAIL_LEVELS)


def tail_prints(times_s: Sequence[float], prices: Sequence[float], half_window_s: float = 300.0, threshold: float = 0.25,
                min_neighbours: int = 3) -> list[dict[str, Any] | None]:
    """Per trade (times ascending): the median price of the OTHER trades within
    +-half_window_s, the deviation from it, and tail = |deviation| >= threshold. None when
    fewer than `min_neighbours` other trades are in the window (not judged)."""
    out: list[dict[str, Any] | None] = []
    for i, (t, p) in enumerate(zip(times_s, prices)):
        lo = bisect.bisect_left(times_s, t - half_window_s)
        hi = bisect.bisect_right(times_s, t + half_window_s)
        neighbours = list(prices[lo:i]) + list(prices[i + 1 : hi])
        if len(neighbours) < min_neighbours:
            out.append(None)
            continue
        median = statistics.median(neighbours)
        deviation = p - median
        out.append({"median": median, "deviation": deviation, "tail": abs(deviation) >= threshold - 1e-9})
    return out


# ---------------------------------------------------------------------- settlement carry scan (HYPOTHETICAL)


def side_buy_prices(yes_bid: float | None, yes_ask: float | None) -> dict[str, float | None]:
    """What buying each side costs now: YES at the YES ask, NO at 1 - YES bid."""
    return {"yes": yes_ask, "no": None if yes_bid is None else round(1.0 - yes_bid, 6)}


def carry_edge(p_side: float | None, price: float | None, haircut: float = 0.0) -> tuple[float | None, float | None]:
    """(edge, yield) of buying a side at `price` and holding to settlement, with the
    oracle probability reduced by `haircut`: edge = (p_side - haircut) - price per share,
    yield = edge / price. HYPOTHETICAL - the oracle is a reference, not the truth."""
    if p_side is None or price is None or price <= 0:
        return None, None
    edge = (p_side - haircut) - price
    return edge, edge / price


def combine_oracles(mids: Mapping[str, float | None], linked: Mapping[str, bool], max_gap: float = 0.03) -> dict[str, Any]:
    """One oracle YES probability from the linked sources. Every linked source must have a
    mid (else p None, reason given); with two, p is their average and `disagree` is set
    when they differ by more than max_gap."""
    sources = [s for s in ("polymarket", "kalshi") if linked.get(s)]
    if not sources:
        return {"p": None, "gap": None, "disagree": False, "reason": "no oracle linked"}
    missing = [s for s in sources if mids.get(s) is None]
    if missing:
        return {"p": None, "gap": None, "disagree": False, "reason": f"no mid from {', '.join(missing)}"}
    values = [float(mids[s]) for s in sources]
    gap = max(values) - min(values) if len(values) > 1 else None
    return {"p": sum(values) / len(values), "gap": gap, "disagree": gap is not None and gap > max_gap + 1e-9, "reason": None}


def side_levels(book: Mapping[str, Any], side: str) -> list[tuple[float, float]]:
    """(price, shares) to BUY `side`, cheapest first, from a YES-normalised book: YES lifts
    YES asks; NO hits YES bids at 1 - bid."""
    if side == "yes":
        levels = [(float(l["price"]), abs(float(l["quantity"]))) for l in book.get("asks") or []]
    else:
        levels = [(round(1.0 - float(l["price"]), 6), abs(float(l["quantity"]))) for l in book.get("bids") or []]
    return sorted(levels)


def walk_levels(levels: Sequence[tuple[float, float]], max_price: float, max_shares: float | None = None, max_cost: float | None = None) -> tuple[float, float]:
    """(shares, cost) of buying cheapest-first at prices <= max_price, at most max_shares
    shares and at most max_cost spent (a level is taken partly when a limit falls inside it)."""
    shares = cost = 0.0
    for price, qty in levels:
        if price > max_price + 1e-9:
            break
        take = qty
        if max_shares is not None:
            take = min(take, max_shares - shares)
        if max_cost is not None and price > 0:
            take = min(take, (max_cost - cost) / price)
        if take <= 1e-9:
            break
        shares += take
        cost += take * price
    return shares, cost


# ---------------------------------------------------------------------- party-only check, hold yield, swaps


def sig_title_party(title: str | None, mapped_party: str | None = None) -> str | None:
    """The party a SIG contract title names (exactly one party word), else the mapped party."""
    parties = text_keywords(title or "")["parties"]
    return next(iter(parties)) if len(parties) == 1 else mapped_party


def party_only_check(sig_party: str | None, oracle: Mapping[str, str | None]) -> tuple[str, list[str]]:
    """Compare ONLY the party between SIG and the oracle labels (Polymarket label / Kalshi
    ticker suffix, from oracle_parties); candidate names are never compared. -> ("match" |
    "mismatch" | "unknown", reasons). Only "mismatch" is a flag: an oracle label that names no
    party (e.g. a bare candidate name) cannot contradict SIG and gives "unknown"."""
    named = {source: party for source, party in oracle.items() if party}
    if sig_party is None or not named:
        return "unknown", []
    reasons = [f"party: SIG {sig_party} vs {source} {party}" for source, party in named.items() if party != sig_party]
    return ("mismatch", reasons) if reasons else ("match", [])


def hold_yield(p_side: float | None, exit_price: float | None, haircut: float = 0.0) -> float | None:
    """HYPOTHETICAL hold-to-settlement yield on the money a sale would free now:
    ((p_side - haircut) - exit_price) / exit_price; exit_price = the side's current bid."""
    if p_side is None or exit_price is None or exit_price <= 0:
        return None
    return ((p_side - haircut) - exit_price) / exit_price


def greedy_swaps(holds: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Match freed capital to candidate capacity, best yield improvement first, each unit of
    capital and of capacity used once. holds: {"id", "capital", "yield"}; candidates: {"id",
    "capacity", "yield"} (capacity in money), optional "exchange". A pair is used only when the
    candidate's yield beats the hold's and ids and exchanges differ. -> [{"hold", "candidate", "amount", "yield_gain", "gain"}]."""
    pairs = sorted(((c["yield"] - h["yield"], h, c) for h in holds for c in candidates
                    if h.get("yield") is not None and c.get("yield") is not None and c["id"] != h["id"]
                    and c.get("exchange", c["id"]) != h.get("exchange", h["id"]) and c["yield"] > h["yield"]),
                   key=lambda x: (-x[0], str(x[1]["id"]), str(x[2]["id"])))
    capital = {h["id"]: float(h["capital"]) for h in holds}
    room = {c["id"]: float(c["capacity"]) for c in candidates}
    swaps = []
    for gain, h, c in pairs:
        amount = min(capital[h["id"]], room[c["id"]])
        if amount <= 1e-9:
            continue
        capital[h["id"]] -= amount
        room[c["id"]] -= amount
        swaps.append({"hold": h["id"], "candidate": c["id"], "amount": amount, "yield_gain": gain, "gain": amount * gain})
    return swaps


# ---------------------------------------------------------------------- basket (constraint) arbitrage, HYPOTHETICAL


def basket_no_metrics(yes_bids: Sequence[float], tick: float = 0.005, slippage_ticks: float = 1.0, haircut: float = 0.005) -> dict[str, float]:
    """Buy NO on every leg of a race at 1 - YES bid. At most one listed party wins, so at
    least n - 1 NO legs pay 1: profit per basket >= sum(bids) - 1 (more if none wins).
    -> raw excess, excess after `slippage_ticks` per leg (in price and ticks), capital locked
    (sum of NO prices), profit per 1 SUSQ locked without / with `haircut` per leg, and the
    break-even fee (per leg per share, and as a share of capital) that eats the raw excess."""
    n = len(yes_bids)
    total = math.fsum(float(b) for b in yes_bids)
    raw = total - 1.0
    after_slip = raw - n * slippage_ticks * tick
    capital = n - total
    after_haircut = after_slip - n * haircut
    return {"legs": n, "sum_bid": total, "excess": raw, "excess_after_slippage": after_slip,
            "excess_ticks_after_slippage": after_slip / tick, "capital": capital,
            "profit_per_capital": after_slip / capital if capital > 0 else math.nan,
            "profit_per_capital_haircut": after_haircut / capital if capital > 0 else math.nan,
            "break_even_fee_per_leg": raw / n if n else math.nan,
            "break_even_fee_of_capital": raw / capital if capital > 0 else math.nan}


# ---------------------------------------------------------------------- which cap binds


def binding_caps(share_limits: Mapping[str, float | None], tol: float = 1e-6) -> str:
    """The cap(s) with the smallest share limit, joined by "+" on a tie, in the order given
    (None = cap not evaluated). "none" when nothing was evaluated."""
    known = {k: float(v) for k, v in share_limits.items() if v is not None}
    if not known:
        return "none"
    low = min(known.values())
    return "+".join(k for k, v in known.items() if v <= low + tol)


# ---------------------------------------------------------------------- passive maker estimate (HYPOTHETICAL, tape PROXY)


def maker_sign(aggressor: str | None) -> int | None:
    """+1 when the maker SOLD YES (the aggressor bought), -1 when the maker BOUGHT YES."""
    return {"buy": 1, "sell": -1}.get(aggressor or "")


def maker_markouts(times: Sequence[float], prices: Sequence[float], calls: Sequence[str | None], horizon_s: float) -> list[tuple[float | None, str]]:
    """Per print (time order): (maker markout in price units per share, status).
    markout = maker_sign * (p - ref), ref = mean of the LAST buy-aggressor and the LAST
    sell-aggressor print in (t, t + horizon] (one of each damps bid-ask bounce). Status:
    "ok", "no aggressor", "beyond data" (t + horizon past the last print) or "one side missing"."""
    out: list[tuple[float | None, str]] = []
    end = times[-1] if times else 0.0
    for i, (t, p) in enumerate(zip(times, prices)):
        sign = maker_sign(calls[i])
        if sign is None:
            out.append((None, "no aggressor"))
            continue
        if t + horizon_s > end:
            out.append((None, "beyond data"))
            continue
        last = {"buy": None, "sell": None}
        j = i + 1
        while j < len(times) and times[j] <= t + horizon_s:
            if calls[j] in last:
                last[calls[j]] = prices[j]
            j += 1
        if last["buy"] is None or last["sell"] is None:
            out.append((None, "one side missing"))
            continue
        out.append((sign * (p - (last["buy"] + last["sell"]) / 2.0), "ok"))
    return out


def simulate_maker_inventory(prints: Sequence[tuple[datetime, float, float, str | None]], fill_rate: float, max_fill: float, inv_cap: float,
                             fee_per_share: float = 0.0) -> dict[str, Any]:
    """Sequential passive maker on one market. Each print (time, YES price, size, aggressor):
    the maker on the other side fills min(max_fill, fill_rate * size), cut so |inventory| stays
    <= inv_cap (a side stops filling at the cap until inventory reverses). Open inventory is
    marked at the last print of each UTC day. -> {"daily": {day: {"pnl", "fills", "shares"}},
    "max_abs_inventory", "max_drawdown"} (drawdown of marked equity, peak to trough)."""
    cash = inv = 0.0
    daily: dict[str, dict[str, float]] = {}
    max_inv = 0.0
    prev_equity = equity_peak = 0.0
    max_dd = 0.0
    day = None
    last_price = None

    def close(d: str) -> None:
        nonlocal prev_equity, equity_peak, max_dd
        equity = cash + inv * (last_price or 0.0)
        daily[d]["pnl"] = equity - prev_equity
        prev_equity = equity
        equity_peak = max(equity_peak, equity)
        max_dd = max(max_dd, equity_peak - equity)

    for t, p, q, aggressor in prints:
        d = t.strftime("%Y-%m-%d")
        if day is not None and d != day:
            close(day)
        day = d
        daily.setdefault(d, {"pnl": 0.0, "fills": 0, "shares": 0.0})
        last_price = p
        sign = maker_sign(aggressor)
        if sign is None:
            continue
        want = min(max_fill, fill_rate * abs(q))
        room = inv_cap + inv if sign > 0 else inv_cap - inv  # selling lowers inventory, buying raises it
        fill = max(0.0, min(want, room))
        if fill <= 0:
            continue
        inv += -fill if sign > 0 else fill
        cash += (fill * p if sign > 0 else -fill * p) - fill * fee_per_share
        max_inv = max(max_inv, abs(inv))
        daily[d]["fills"] += 1
        daily[d]["shares"] += fill
    if day is not None:
        close(day)
    return {"daily": daily, "max_abs_inventory": max_inv, "max_drawdown": max_dd}


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[Hashable], n_boot: int = 1000, seed: int = 7, level: float = 0.95) -> tuple[float | None, float | None, float | None]:
    """(mean, CI low, CI high) of the per-observation mean, resampling whole CLUSTERS with
    replacement (observations in one market-day are not independent). Needs >= 2 clusters
    for a CI; a fixed seed keeps reports reproducible."""
    if not values:
        return None, None, None
    groups: dict[Hashable, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(float(v))
    mean = math.fsum(values) / len(values)
    return cluster_bootstrap_sums(mean, [(math.fsum(groups[k]), len(groups[k])) for k in groups], n_boot, seed, level)


def cluster_bootstrap_sums(mean: float, sums: Sequence[tuple[float, int]], n_boot: int = 1000, seed: int = 7, level: float = 0.95) -> tuple[float, float | None, float | None]:
    """The cluster bootstrap from per-cluster aggregates (sum, count) in first-appearance order -
    what cluster_bootstrap_mean resamples, without keeping individual observations."""
    if len(sums) < 2:
        return mean, None, None
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        total = count = 0.0
        for _k in range(len(sums)):
            s, n = sums[rng.randrange(len(sums))]
            total += s
            count += n
        boots.append(total / count)
    boots.sort()
    alpha = (1 - level) / 2
    return mean, percentile(boots, 100 * alpha), percentile(boots, 100 * (1 - alpha))


def msum_partials(values: Iterable[float], partials: list[float] | None = None) -> list[float]:
    """Shewchuk's exact partial sums (the algorithm behind math.fsum), extended in place:
    math.fsum(partials) equals math.fsum of every value ever added, whatever the grouping."""
    partials = [] if partials is None else partials
    for x in values:
        i = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]
    return partials


class TimeHalves:
    """split_halves without keeping the rows: first / second half of a population by time.
    Built from every time of the population; label() must then be called for its members in the
    original row order (ties at the cut go to the first half in that order, as a stable sort)."""

    def __init__(self, times: Iterable[float]) -> None:
        ordered = sorted(times)
        self.size = len(ordered)
        first = (self.size + 1) // 2
        self.cut = ordered[first - 1] if first else None
        self.ties_first = first - bisect.bisect_left(ordered, self.cut) if first else 0
        self.ties_seen = 0

    def label(self, t: float) -> str | None:
        if self.cut is None:
            return None
        if t < self.cut:
            return "first_half"
        if t > self.cut:
            return "second_half"
        self.ties_seen += 1
        return "first_half" if self.ties_seen <= self.ties_first else "second_half"


class CutAggregator:
    """Streaming cut statistics: prints, qualifying prints, drop rate, markets and the mean value /
    tick with its cluster-bootstrap CI (cluster = market-day), fed one market at a time. Keeps per
    group only exact partial sums and one (sum, count) per market-day - never the observations."""

    def __init__(self, tick: float) -> None:
        self.tick = tick
        self.groups: dict[Hashable, dict[str, Any]] = {}
        self._calls = 0

    def touch(self, key: Hashable) -> dict[str, Any]:
        return self.groups.setdefault(key, {"called": 0, "ok": 0, "markets": set(), "partials": [], "clusters": [], "firsts": None})

    def add_market(self, key: Hashable, market: str, items: Sequence[tuple[str, float | None]], times: Sequence[float] | None = None) -> None:
        """The CALLED prints of one market in this group, in time order: (UTC day, value or None).
        Clusters enter the bootstrap in first-appearance order: market order by default, or - with
        `times` (one per item) - the order of a stable time sort of all members, as split_halves gives."""
        g = self.touch(key)
        g["called"] += len(items)
        self._calls += 1
        by_day: dict[str, list[float]] = {}
        first: dict[str, tuple[float, int, int]] = {}
        for i, (day, value) in enumerate(items):
            if value is not None:
                by_day.setdefault(day, []).append(value / self.tick)
                if times is not None:
                    first.setdefault(day, (times[i], self._calls, i))
        if not by_day:
            return
        g["markets"].add(market)
        if times is not None:
            g["firsts"] = g["firsts"] if g["firsts"] is not None else []
        for day, values in by_day.items():  # first-appearance order = day order within a market
            g["ok"] += len(values)
            cluster = msum_partials(values)
            g["clusters"].append((math.fsum(cluster), len(values)))
            if times is not None:
                g["firsts"].append(first[day])
            msum_partials(cluster, g["partials"])

    def result(self, key: Hashable) -> dict[str, Any]:
        g = self.touch(key)
        called, ok = g["called"], g["ok"]
        mean = lo = hi = None
        if ok:
            clusters = g["clusters"]
            if g["firsts"] is not None:
                clusters = [c for _, c in sorted(zip(g["firsts"], clusters), key=lambda x: x[0])]
            mean, lo, hi = cluster_bootstrap_sums(math.fsum(g["partials"]) / ok, clusters)
        return {"prints": called, "qualifying": ok, "drop_rate": round(1 - ok / called, 4) if called else None, "markets": len(g["markets"]),
                "mean_ticks": None if mean is None else round(mean, 3), "ci_low_ticks": None if lo is None else round(lo, 3),
                "ci_high_ticks": None if hi is None else round(hi, 3)}


def quartile_label(value: float, cuts: Sequence[float]) -> str:
    """Q1..Q4 against three ascending cut points (25th, 50th, 75th percentiles)."""
    for i, cut in enumerate(cuts, start=1):
        if value <= cut:
            return f"Q{i}"
    return "Q4"


# ---------------------------------------------------------------------- quotes: prevailing quote, Lee-Ready, spread width


def prevailing_quote(times: Sequence[float], quotes: Sequence[T], t: float, max_age_s: float) -> T | None:
    """The latest quote at or before `t` (times ascending), if it is at most max_age_s old."""
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or t - times[i] > max_age_s:
        return None
    return quotes[i]


def lee_ready(price: float, bid: float | None, ask: float | None, eps: float = 1e-9) -> str | None:
    """Quote rule: a trade at or above the ask is a buy, at or below the bid a sell; inside the
    spread (or without a two-sided quote) None - the caller falls back to the tick rule."""
    if ask is not None and price >= ask - eps:
        return "buy"
    if bid is not None and price <= bid + eps:
        return "sell"
    return None


def spread_bucket(bid: float | None, ask: float | None, tick: float = 0.005) -> str | None:
    """"1", "2" or ">=3" ticks; None without a two-sided quote."""
    if bid is None or ask is None:
        return None
    ticks = round((ask - bid) / tick)
    return ">=3" if ticks >= 3 else str(max(ticks, 1))


# ---------------------------------------------------------------------- oracle fair and uncertainty (offline copies of the bot's rules)

ORACLE_MAX_SPREAD = 0.10  # = strategy.MAX_ORACLE_SPREAD
ORACLE_MAX_DISAGREEMENT = 0.10  # = strategy.MAX_ORACLE_DISAGREEMENT


def oracle_problem(quote: Any, max_age_s: float) -> str | None:
    """Why one oracle quote (mid / spread / age_seconds attributes) is unusable for the fair, or None."""
    if quote is None:
        return "missing"
    if quote.age_seconds > max_age_s:
        return "stale"
    if quote.mid is None or quote.spread is None or quote.spread < 0:
        return "one-sided"
    if quote.spread > ORACLE_MAX_SPREAD:
        return "wide"
    return None


def oracle_fair(quotes: Sequence[Any], max_age_s: float) -> tuple[float | None, str]:
    """(fair P(YES), "ok") or (None, reason): the same rule as strategy.compute_fair_price - the mean
    mid of the fresh, tight, two-sided quotes; None when the two usable mids differ by more than
    ORACLE_MAX_DISAGREEMENT. Research code must not import strategy, hence this copy."""
    usable, problems = [], []
    for q in quotes:
        problem = oracle_problem(q, max_age_s)
        if problem is None:
            usable.append(q)
        else:
            problems.append(problem)
    if not usable:
        return None, next(r for r in ("stale", "wide", "one-sided", "missing") if r in problems)
    if len(usable) == 2 and abs(usable[0].mid - usable[1].mid) > ORACLE_MAX_DISAGREEMENT:
        return None, "disagree"
    return sum(q.mid for q in usable) / len(usable), "ok"


def oracle_uncertainty(quotes: Sequence[Any], max_age_s: float, tick: float = 0.005) -> float:
    """u = max(tick, half the tightest usable spread, half the gap between two usable mids): the same
    rule as ladder.uncertainty_band (fresh two-sided quotes, no spread limit)."""
    usable = [q for q in quotes if q is not None and q.age_seconds <= max_age_s and q.mid is not None and q.spread is not None and q.spread >= 0]
    half_tightest = min(q.spread for q in usable) / 2 if usable else 0.0
    disagreement = abs(usable[0].mid - usable[1].mid) / 2 if len(usable) == 2 else 0.0
    return max(tick, half_tightest, disagreement)


# ---------------------------------------------------------------------- quoter reach (offline copies of the quoter's pricing)

QUOTER_MIN_EDGE = 0.01  # = config.QUOTER_MIN_EDGE default
QUOTER_EDGE_OVER_U = 0.01  # = config.QUOTER_EDGE_OVER_U


def quoter_edge(u: float, min_edge: float = QUOTER_MIN_EDGE, over_u: float = QUOTER_EDGE_OVER_U) -> float:
    """The quoter's minimum distance from fair: max(QUOTER_MIN_EDGE, u + QUOTER_EDGE_OVER_U)."""
    return max(min_edge, u + over_u)


def _floor_tick(price: float, tick: float) -> float:
    return round(math.floor(price / tick + 1e-9) * tick, 3)


def quoter_prices(fair_yes: float, u: float, other_yes_bid: float | None, other_no_bid: float | None, tick: float = 0.005,
                  low: float = 0.005, high: float = 0.995) -> tuple[float | None, float | None]:
    """(YES bid, NO bid) of quoter.plan_quotes with zero race delta (no skew, no delta block): a tick
    inside the other participants' best bid per side, capped at fair - edge in that side's terms,
    never marketable, YES + NO <= 1 - 2 ticks; None outside [low, high]. Research must not import
    the quoter, hence this copy."""
    edge = quoter_edge(u)
    targets = {}
    for side, fair_side, other in (("yes", fair_yes, other_yes_bid), ("no", 1.0 - fair_yes, other_no_bid)):
        target = fair_side - edge
        if other is not None:
            target = min(other + tick, target)
        targets[side] = target
    if other_no_bid is not None:
        targets["yes"] = min(targets["yes"], round(1.0 - other_no_bid - tick, 6))
    if other_yes_bid is not None:
        targets["no"] = min(targets["no"], round(1.0 - other_yes_bid - tick, 6))
    prices = {side: _floor_tick(target, tick) for side, target in targets.items()}
    while prices["yes"] + prices["no"] > 1.0 - 2 * tick + 1e-9:
        lower = "yes" if prices["yes"] >= prices["no"] else "no"
        prices[lower] = round(prices[lower] - tick, 3)
    yes, no = (p if low - 1e-9 <= p <= high + 1e-9 else None for p in (prices["yes"], prices["no"]))
    return yes, no


def touch_ticks(bid: float | None, ask: float | None, tick: float = 0.005) -> int | None:
    return None if bid is None or ask is None else round((ask - bid) / tick)


def join_quotes(bid: float | None, ask: float | None, fair: float, edge: float, s_min: int, tick: float = 0.005) -> tuple[float | None, float | None]:
    """JOIN: (YES bid, YES ask) at the touch, in YES terms. The bid only when bid <= fair - edge, the ask
    (= a NO bid) only when ask >= fair + edge; neither when the spread is below s_min ticks or one-sided."""
    spread = touch_ticks(bid, ask, tick)
    if spread is None or spread < s_min:
        return None, None
    return (bid if bid <= fair - edge + 1e-9 else None), (ask if ask >= fair + edge - 1e-9 else None)


def improve_quotes(bid: float | None, ask: float | None, fair: float, u: float, s_min: int, tick: float = 0.005) -> tuple[float | None, float | None]:
    """IMPROVE (the current quoter): (YES bid, YES ask) in YES terms from quoter_prices against this
    touch; neither when the spread is below s_min ticks or one-sided."""
    spread = touch_ticks(bid, ask, tick)
    if spread is None or spread < s_min:
        return None, None
    yes, no = quoter_prices(fair, u, bid, round(1.0 - ask, 3), tick)
    return yes, (None if no is None else round(1.0 - no, 3))


def reaches(aggressor: str | None, price: float, yes_bid: float | None, yes_ask: float | None, exact: bool) -> bool:
    """Would a print reach our quote on its maker side? An aggressor SELL meets our YES bid, a BUY our
    YES ask (= NO bid). exact: the print is AT our price (JOIN); otherwise at or through it (IMPROVE)."""
    quote = yes_bid if aggressor == "sell" else yes_ask if aggressor == "buy" else None
    if quote is None:
        return False
    if exact:
        return abs(price - quote) <= 1e-9
    return price <= quote + 1e-9 if aggressor == "sell" else price >= quote - 1e-9


def capped_fill(size: float, cap: float = 100.0, share: float = 0.25) -> float:
    """Optimistic no-queue fill of one print: min(cap, share x size)."""
    return min(cap, share * abs(size))


# ---------------------------------------------------------------------- reducing a held position (sell into the book)


def sell_levels(book: Mapping[str, Any], side: str) -> list[tuple[float, float]]:
    """(price, shares) to SELL `side`, best first, in that side's terms, from a YES-normalised book:
    YES hits YES bids; NO lifts YES asks (sell NO q @ p = buy YES q @ 1 - p), at 1 - ask."""
    if side == "yes":
        levels = [(float(l["price"]), abs(float(l["quantity"]))) for l in book.get("bids") or []]
    else:
        levels = [(round(1.0 - float(l["price"]), 6), abs(float(l["quantity"]))) for l in book.get("asks") or []]
    return sorted(levels, reverse=True)


def sweep_sell(levels: Sequence[tuple[float, float]], quantity: float) -> dict[str, Any]:
    """Sell `quantity` into `levels` (best first): filled shares, proceeds, VWAP, levels touched, the
    best and the last price hit, and whether the depth covered the whole quantity."""
    filled = proceeds = 0.0
    used, last = 0, None
    for price, qty in levels:
        take = min(qty, quantity - filled)
        if take <= 1e-9:
            break
        filled += take
        proceeds += take * price
        used, last = used + 1, price
    return {"filled": filled, "proceeds": proceeds, "vwap": proceeds / filled if filled else None, "levels": used,
            "best": levels[0][0] if levels else None, "last": last, "enough": filled >= quantity - 1e-9}


def shrink_toward_half(p: float, by: float) -> float:
    """p moved `by` toward 0.5, never past it."""
    return max(p - by, 0.5) if p > 0.5 else min(p + by, 0.5)


def race_scenario_pnl(legs: Iterable[tuple[str, str, float, float]], winners: Iterable[str]) -> dict[str, float]:
    """Settlement P&L per winner: legs (party, side, shares held, net cost = cost basis - sale proceeds);
    a YES leg pays its shares when its party wins, a NO leg when it does not."""
    legs = list(legs)
    return {w: math.fsum((shares if (side == "yes") == (party == w) else 0.0) - net_cost for party, side, shares, net_cost in legs) for w in winners}
