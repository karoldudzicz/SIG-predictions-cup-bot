"""How a fill's `price` is denominated - per data source - and the helpers every reader uses.

REST (GET /tournaments/{slug}/portfolio/fills, what FillFeed and the ladder read): `price`
is what was paid per share of the fill's OWN side. Verified on live data (2026-10-03): for
a NO fill, shares * price matched the position's costBasis in /portfolio/positions to the
cent - so a NO fill's price is a NO price, not a YES price.

Realtime (the fills[] of account batches): documented as YES-denominated ("for a NO
position the effective per-share cost is 1 - price"). Not read by the bot yet; the entry
is here so a realtime reader passes source="realtime" and gets the right conversion.

Every cost basis, exit price, P&L and markout computed from a fill goes through
fill_price_side / fill_price_yes - never through fill["price"] directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

FillSource = Literal["rest", "realtime"]
Side = Literal["yes", "no"]

# What `price` is quoted in, per source: the fill's own side, or YES.
PRICE_CONVENTION: dict[str, Literal["side", "yes"]] = {"rest": "side", "realtime": "yes"}


def _raw(fill: Mapping[str, Any], side: str | None) -> tuple[float, str] | None:
    price = fill.get("price")
    side = side or fill.get("side")
    if price is None or side not in ("yes", "no"):
        return None
    return float(price), side


def fill_price_side(fill: Mapping[str, Any], source: FillSource = "rest", side: Side | None = None) -> float | None:
    """Price per share in the fill's own side's terms (a NO fill -> a NO price).

    `side` overrides fill["side"] (only when the fill does not carry one).
    """
    raw = _raw(fill, side)
    if raw is None:
        return None
    price, side = raw
    if PRICE_CONVENTION[source] == "yes" and side == "no":
        return round(1.0 - price, 6)
    return price


def fill_price_yes(fill: Mapping[str, Any], source: FillSource = "rest", side: Side | None = None) -> float | None:
    """Price per share in YES terms (a NO fill at a NO price p -> 1 - p)."""
    raw = _raw(fill, side)
    if raw is None:
        return None
    price, side = raw
    if PRICE_CONVENTION[source] == "side" and side == "no":
        return round(1.0 - price, 6)
    return price


def fill_shares(fill: Mapping[str, Any]) -> float:
    """Shares filled (the API signs quantity by side: NO < 0)."""
    return abs(float(fill.get("quantity") or 0))


def fill_cost(fill: Mapping[str, Any], source: FillSource = "rest", side: Side | None = None) -> float | None:
    """Cash per fill in the fill's own side's terms: shares * side price."""
    price = fill_price_side(fill, source, side)
    return None if price is None else fill_shares(fill) * price
