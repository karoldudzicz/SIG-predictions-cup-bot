"""Fill markouts: how the fair price moved after each of our fills (taker, ladder, quoter).

Per cycle, strategy.py:
  1. FillFeed.read()          - our new tournament fills since this process started
                                (one GET per cycle; nothing in DRY_RUN)
  2. MarkoutTracker.record()  - one "fill" line per fill: source, exchange, side, action,
                                price (YES-denominated), size, fair at the time we saw it
  3. MarkoutTracker.update()  - one "markout" line per fill and horizon (1, 5, 30 min)
                                once it is due, with that cycle's oracle fair

markout (ticks) = direction * (fair_after - fill_price_yes) / tick, where direction is +1
for a fill that made us longer YES (buy YES, sell NO) and -1 otherwise: positive = the
price moved our way after the fill, negative = we were picked off.

The Fill schema has no buy / sell: the action comes from what we know about the order
(ladder / quoter metadata, the taker's order journal). A fill whose action is unknown
gets markout null rather than a guessed sign.

Prices: REST fills quote the fill's own side (see fills.py); markouts are computed in YES
terms on both sides (fill price converted, oracle fair is YES). "fill" lines written
before that was known carry no `price_source` and stored a NO fill's NO price as
price_yes: legacy_price_yes() corrects them on read (the replay here and
scripts/markout_report.py); the journal itself is never rewritten.

data/markout.jsonl is append-only; scripts/markout_report.py summarises it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exchange_api import PRICE_TICK, ExchangeClient
from fills import FillSource, fill_price_side, fill_price_yes, fill_shares
from market_mapper import DATA_DIR

logger = logging.getLogger(__name__)

MARKOUT_JOURNAL_PATH = DATA_DIR / "markout.jsonl"
MARKOUT_HORIZONS_S = (60, 300, 1800)
# A horizon still without a fresh fair this long after it was due is written with fair null.
MARKOUT_GRACE_S = 300
# FillFeed re-reads this far behind the newest fill it has seen (fills can be recorded late).
FILL_FEED_OVERLAP_S = 120


def unix_time(raw: str | None) -> float:
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def legacy_price_yes(entry: Mapping[str, Any]) -> float | None:
    """price_yes of a journalled "fill" line. Lines without `price_source` were written while
    REST prices were taken for YES prices: for a NO fill they hold the NO price -> 1 - it."""
    price = entry.get("price_yes")
    if price is None or entry.get("price_source") is not None or entry.get("side") != "no":
        return price
    return round(1.0 - float(price), 6)


def long_yes_direction(side: str, action: str | None) -> int | None:
    """+1 if the fill made us longer YES (buy YES / sell NO), -1 if shorter, None if unknown."""
    if action not in ("buy", "sell") or side not in ("yes", "no"):
        return None
    return 1 if (side == "yes") == (action == "buy") else -1


def markout_ticks(direction: int | None, price_yes: float | None, fair_after: float | None) -> float | None:
    if direction is None or price_yes is None or fair_after is None:
        return None
    return round(direction * (fair_after - price_yes) / PRICE_TICK, 2)


# ---------------------------------------------------------------------- fill feed


class FillFeed:
    """Our new tournament fills, oldest first, each returned once.

    Reads pages (newest first) until it reaches fills older than its watermark: process
    start at first, later the newest fill seen minus FILL_FEED_OVERLAP_S. Fills made before
    this process started are never returned. In DRY_RUN it sends nothing.
    """

    def __init__(self, exchange: ExchangeClient, tournament_slug: str, clock: Callable[[], float] = time.time) -> None:
        self._exchange = exchange
        self._slug = tournament_slug
        self._since = clock()
        self._watermark = self._since
        self._seen: set[Any] = set()

    async def read(self) -> list[dict[str, Any]]:
        if self._exchange.dry_run:
            return []
        newest_first: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = await self._exchange.get_tournament_fills(self._slug, cursor=cursor)
            data = page.get("data") or []
            newest_first.extend(data)
            pagination = page.get("pagination") or {}
            cursor = pagination.get("nextCursor")
            if any(unix_time(f.get("filledAt")) < self._watermark for f in data) or not cursor or not pagination.get("hasMore"):
                break
        new = []
        for fill in reversed(newest_first):
            filled_at = unix_time(fill.get("filledAt"))
            if fill.get("id") in self._seen or filled_at < self._since:
                continue
            self._seen.add(fill.get("id"))
            new.append(fill)
            self._watermark = max(self._watermark, filled_at - FILL_FEED_OVERLAP_S)
        return new


# ---------------------------------------------------------------------- tracker


@dataclass(frozen=True)
class Markout:
    source: str
    exchange_id: str
    fill_id: Any
    order_id: Any
    filled_at: float
    horizon_s: int
    fair: float | None
    ticks: float | None


@dataclass
class _Pending:
    source: str
    exchange_id: str
    fill_id: Any
    order_id: Any
    filled_at: float
    direction: int | None
    price_yes: float | None
    horizons: list[int]


class MarkoutTracker:
    def __init__(
        self,
        path: Path = MARKOUT_JOURNAL_PATH,
        horizons: tuple[int, ...] = MARKOUT_HORIZONS_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self._horizons = horizons
        self._pending: dict[Any, _Pending] = self._replay(clock())

    def _write(self, event: str, **fields: Any) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _replay(self, now: float) -> dict[Any, _Pending]:
        """Fills whose horizons are not all written yet and can still be measured."""
        pending: dict[Any, _Pending] = {}
        if not self.path.exists():
            return pending
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("event") == "fill":
                pending[entry["fill_id"]] = _Pending(
                    entry["source"], entry["exchange_id"], entry["fill_id"], entry.get("order_id"), entry["filled_at"],
                    long_yes_direction(entry["side"], entry.get("action")), legacy_price_yes(entry), list(self._horizons),
                )
            elif entry.get("event") == "markout" and entry.get("fill_id") in pending:
                horizons = pending[entry["fill_id"]].horizons
                if entry.get("horizon_s") in horizons:
                    horizons.remove(entry["horizon_s"])
        return {
            fill_id: p for fill_id, p in pending.items()
            if p.horizons and now <= p.filled_at + max(p.horizons) + MARKOUT_GRACE_S
        }

    @property
    def pending(self) -> int:
        return len(self._pending)

    def record(
        self, source: str, fill: Mapping[str, Any], action: str | None, fair_yes: float | None, price_source: FillSource = "rest"
    ) -> None:
        """One of our fills, seen this cycle. `fair_yes`: this cycle's fair for its exchange.
        `price_source`: where the fill came from - decides how its price is converted."""
        side = str(fill.get("side"))
        price_yes = fill_price_yes(fill, price_source)
        filled_at = unix_time(fill.get("filledAt"))
        direction = long_yes_direction(side, action)
        self._write(
            "fill", source=source, exchange_id=str(fill.get("exchangeId")), side=side, action=action,
            price_source=price_source, price_side=fill_price_side(fill, price_source), price_yes=price_yes,
            quantity=fill_shares(fill), fill_id=fill.get("id"), order_id=fill.get("orderId"), filled_at=filled_at,
            fair_at_fill=fair_yes, markout_at_fill=markout_ticks(direction, price_yes, fair_yes),
        )
        self._pending[fill.get("id")] = _Pending(
            source, str(fill.get("exchangeId")), fill.get("id"), fill.get("orderId"), filled_at, direction, price_yes, list(self._horizons)
        )

    def update(self, now: float, fair: Mapping[str, float]) -> list[Markout]:
        """Write every horizon that is due: with this cycle's fair if there is one, else
        (after MARKOUT_GRACE_S) with fair null. Returns what was written, oldest fill first."""
        written: list[Markout] = []
        for fill_id, p in sorted(self._pending.items(), key=lambda item: item[1].filled_at):
            for horizon in list(p.horizons):
                due = p.filled_at + horizon
                if now < due:
                    continue
                fair_after = fair.get(p.exchange_id)
                if fair_after is None and now < due + MARKOUT_GRACE_S:
                    continue  # no fresh fair this cycle; try again next cycle
                ticks = markout_ticks(p.direction, p.price_yes, fair_after)
                self._write(
                    "markout", source=p.source, exchange_id=p.exchange_id, fill_id=fill_id, order_id=p.order_id,
                    horizon_s=horizon, fair=fair_after, late_s=round(now - due, 1), markout_ticks=ticks,
                )
                written.append(Markout(p.source, p.exchange_id, fill_id, p.order_id, p.filled_at, horizon, fair_after, ticks))
                p.horizons.remove(horizon)
        self._pending = {fill_id: p for fill_id, p in self._pending.items() if p.horizons}
        return written
