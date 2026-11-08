"""Live order placement vs. timeouts and cycle cancellation. Fake session - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import exchange_api
from config import Settings
from market_mapper import MarketMap, MarketMapping, Race
from strategy import OrderJournal, OrderStatus, PlannedOrder, Strategy, drop_blocked_by_unknown
from test_exchange_http import _client, _FakeSession

_ORDER = dict(kind="increase", exchange_id="1075", side="no", quantity=10, price=0.92, reason="test")


def _strategy(session: _FakeSession, journal: OrderJournal) -> Strategy:
    """A live-mode (DRY_RUN off) Strategy whose exchange client talks to the fake session."""
    client = _client(session, dry_run=False)
    settings = Settings(
        sig_api_key="test-key",
        sig_base_url="https://sig.invalid/api/v1",
        dry_run=False,
        log_level="DEBUG",
        log_dir=Path("."),
        tournament_slug="t",
        oracle_staleness_threshold_s=10.0,
        cycle_interval_s=15.0,
    )
    market_map = MarketMap(
        {"1075": MarketMapping("1075", "386", "Will the Republican Party win the Delaware Senate?", Race("senate", "DE"), "R", None, None)}
    )
    return Strategy(settings, client, None, None, market_map, journal)


def test_post_that_never_answers_ends_unknown_at_the_deadline(monkeypatch, tmp_path):
    monkeypatch.setattr(exchange_api, "SIG_POST_TIMEOUT_S", 0.25)
    monkeypatch.setattr(exchange_api, "SIG_POST_DEADLINE_S", 0.6)
    session = _FakeSession(delays=[30.0] * 10)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    strategy = _strategy(session, journal)
    order = PlannedOrder(**_ORDER, quote_fetched_at=time.time())

    started = time.monotonic()
    status = asyncio.run(strategy._execute(order))
    elapsed = time.monotonic() - started

    assert status is OrderStatus.UNKNOWN
    assert elapsed < 0.6 + 0.4  # bounded by the deadline, not by 5 x 15 s
    assert len(session.calls) >= 2  # it did retry inside the deadline...
    keys = {body["idempotencyKey"] for body in session.bodies}
    assert len(keys) == 1  # ...always with the same idempotencyKey
    assert journal.status(keys.pop()) is OrderStatus.UNKNOWN


def test_cancelled_cycle_marks_the_order_unknown_and_propagates(tmp_path):
    session = _FakeSession(delays=[10.0])  # POST in flight when the cycle is cancelled
    journal = OrderJournal(tmp_path / "journal.jsonl")
    strategy = _strategy(session, journal)

    async def scenario():
        task = asyncio.create_task(strategy._execute(PlannedOrder(**_ORDER, quote_fetched_at=time.time())))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):  # the cancellation is NOT swallowed
            await task

    asyncio.run(scenario())
    key = session.bodies[0]["idempotencyKey"]
    assert journal.status(key) is OrderStatus.UNKNOWN
    assert journal._latest[key]["error"] == "cycle cancelled while POST in flight"
    assert not journal.is_settled(key)


def test_recent_unknown_blocks_same_exchange_and_side(tmp_path, caplog):
    journal = OrderJournal(tmp_path / "journal.jsonl")
    journal.record("k-old", OrderStatus.UNKNOWN, payload={"exchangeId": "1075", "side": "no", "action": "buy"})

    blocked = journal.recent_unknown(max_age_s=300)
    assert set(blocked) == {("1075", "no")}

    planned = [
        PlannedOrder("increase", "1075", "no", 10, 0.92, "same market, same side"),
        PlannedOrder("increase", "1075", "yes", 10, 0.08, "same market, other side"),
        PlannedOrder("increase", "1042", "no", 10, 0.05, "other market"),
    ]
    with caplog.at_level("INFO", logger="strategy"):
        kept, dropped = drop_blocked_by_unknown(planned, blocked)
    assert dropped == 1
    assert [(o.exchange_id, o.side) for o in kept] == [("1075", "yes"), ("1042", "no")]
    assert any(r.getMessage().startswith("Order blocked: recent UNKNOWN") and r.unknown_key == "k-old" for r in caplog.records)


def test_unknown_older_than_window_no_longer_blocks(tmp_path):
    journal = OrderJournal(tmp_path / "journal.jsonl")
    journal.record("k-old", OrderStatus.UNKNOWN, payload={"exchangeId": "1075", "side": "no"})
    later = datetime.now(timezone.utc) + timedelta(minutes=6)
    assert journal.recent_unknown(max_age_s=300, now=later) == {}


def test_resolved_or_accepted_entries_do_not_block(tmp_path):
    journal = OrderJournal(tmp_path / "journal.jsonl")
    journal.record("k1", OrderStatus.UNKNOWN, payload={"exchangeId": "1075", "side": "no"})
    journal.mark_resolved("k1", "startup reconciliation")
    journal.record("k2", OrderStatus.ACCEPTED, payload={"exchangeId": "1042", "side": "no"})
    assert journal.recent_unknown(max_age_s=300) == {}
