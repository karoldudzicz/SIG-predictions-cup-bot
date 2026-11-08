"""Write bucket priorities and the taker's pay-then-check quote age. Fake session - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import time

import strategy
from exchange_api import WRITE_RANK_CANCEL, WRITE_RANK_LADDER, WRITE_RANK_ORDER, WRITE_RANK_QUOTER, OrderRequest, _WriteBucket
from strategy import CycleReport, OrderJournal, PlannedOrder, Strategy
from test_cycle_resilience import SETTINGS, FakeKalshi, FakePolymarket, MARKET_MAP
from test_exchange_http import _client, _FakeSession

ORDER_BODY = {"orderId": 1, "exchangeId": "1", "action": "buy", "side": "yes", "price": 0.45, "quantity": 10,
              "quantityTraded": 10, "remainingQuantity": 0, "open": False}


class _TimedSession(_FakeSession):
    """Also records the wall-clock time each request goes out."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_at: list[float] = []

    def request(self, method, url, params=None, json=None, timeout=None):
        self.sent_at.append(time.time())
        return super().request(method, url, params=params, json=json, timeout=timeout)


def _planned(fetched_at: float | None) -> PlannedOrder:
    return PlannedOrder("increase", "1", "yes", 10, 0.45, "test", quote_fetched_at=fetched_at)


def _strategy(client, tmp_path) -> Strategy:
    return Strategy(SETTINGS, client, FakePolymarket(), FakeKalshi(), MARKET_MAP, OrderJournal(tmp_path / "j.jsonl"))


def test_full_bucket_cycle_never_sends_a_stale_quote(tmp_path, monkeypatch):
    """12 taker orders, bucket of 8 at 10 tokens/s, max quote age 0.25 s: orders 9+ wait
    for tokens; whatever is sent is no older than the limit at send time, the rest are
    counted as skipped_stale_quote."""
    max_age = 0.25
    real_check = strategy.quote_too_old
    monkeypatch.setattr(strategy, "quote_too_old", lambda order: real_check(order, max_age_s=max_age))
    session = _TimedSession(delays=[0.0] * 12, body=ORDER_BODY)
    client = _client(session, dry_run=False)
    client._write_bucket = _WriteBucket(per_minute=600, burst=8)
    fetched_at = time.time()
    report = CycleReport()

    asyncio.run(_strategy(client, tmp_path)._execute_planned([_planned(fetched_at) for _ in range(12)], report))

    ages = [sent - fetched_at for sent in session.sent_at]
    assert ages and max(ages) <= max_age + 0.02  # 0.02: the check-to-socket gap, no awaits in between
    assert report.skipped_stale_quote >= 1  # the last orders waited past the limit
    assert report.accepted + report.skipped_stale_quote == 12
    assert report.accepted == len(session.calls) >= 8  # the burst went out at once


def test_skipped_order_gives_its_token_back(tmp_path):
    session = _FakeSession(delays=[0.0], body=ORDER_BODY)
    client = _client(session, dry_run=False)
    client._write_bucket = _WriteBucket(per_minute=1e-4, burst=1)  # one token, practically no refill
    taker = _strategy(client, tmp_path)

    async def scenario():
        skipped = await taker._execute(_planned(None))  # undatable quote -> too old
        assert skipped is None and session.calls == []
        assert client._write_bucket._tokens >= 0.999  # the token is back
        # ... and usable at once by the next order.
        return await asyncio.wait_for(taker._execute(_planned(time.time())), timeout=1)

    assert asyncio.run(scenario()).value == "accepted"
    assert len(session.calls) == 1


def test_ladder_never_takes_the_reserve_while_taker_waits():
    async def scenario():
        bucket = _WriteBucket(per_minute=600, burst=8)  # 10 tokens/s
        bucket._tokens = 0.0
        taken: list[tuple[str, float]] = []

        async def take(name, **kwargs):
            await bucket.acquire(**kwargs)
            taken.append((name, bucket._tokens))

        ladder = asyncio.create_task(take("ladder", rank=WRITE_RANK_LADDER, reserve=3))
        await asyncio.sleep(0)
        taker = asyncio.create_task(take("taker"))
        await asyncio.gather(ladder, taker)
        return taken

    taken = asyncio.run(scenario())
    assert [name for name, _ in taken] == ["taker", "ladder"]  # the ladder asked first, still waited
    assert taken[1][1] >= 3.0 - 1e-6  # >= 3 tokens left after the ladder's take


def test_ladder_batch_waits_for_the_reserve_taker_order_does_not():
    session = _FakeSession(delays=[0.0] * 2, body={"results": []})
    client = _client(session, dry_run=False)
    client._write_bucket._tokens = 3.5  # >= 1 for the taker, < 1 + 3 for the ladder
    client._write_bucket._rate = 1e-6
    order = OrderRequest(exchange_id="1", side="yes", action="buy", quantity=10, price=0.45)

    async def scenario():
        try:
            await asyncio.wait_for(client.place_batch([order], "ladder-1", write_rank=WRITE_RANK_LADDER), timeout=0.2)
        except TimeoutError:
            pass
        else:
            raise AssertionError("ladder batch went out below the reserve")
        session.body = ORDER_BODY
        await asyncio.wait_for(client.place_order(order, "taker-1"), timeout=1)

    asyncio.run(scenario())
    assert [c[1] for c in session.calls] == ["/orders"]


def test_cancels_still_go_first():
    async def scenario():
        bucket = _WriteBucket(per_minute=600, burst=8)
        bucket._tokens = 0.0
        order: list[str] = []

        async def take(name, **kwargs):
            await bucket.acquire(**kwargs)
            order.append(name)

        tasks = []
        for name, kwargs in (
            ("ladder", {"rank": WRITE_RANK_LADDER, "reserve": 3}),
            ("quoter", {"rank": WRITE_RANK_QUOTER, "reserve": 3}),
            ("taker", {"rank": WRITE_RANK_ORDER}),
            ("cancel", {"rank": WRITE_RANK_CANCEL}),
        ):
            tasks.append(asyncio.create_task(take(name, **kwargs)))
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(scenario()) == ["cancel", "taker", "quoter", "ladder"]


def test_prepaid_order_takes_no_second_token_and_refunds_when_blocked():
    session = _FakeSession(delays=[0.0], body=ORDER_BODY)
    client = _client(session, dry_run=False)
    client._write_bucket = _WriteBucket(per_minute=1e-4, burst=1)
    order = OrderRequest(exchange_id="1", side="yes", action="buy", quantity=10, price=0.45)

    async def scenario():
        assert await client.acquire_order_token() is True  # bucket now empty
        await asyncio.wait_for(client.place_order(order, "k-1", prepaid=True), timeout=1)  # no second wait
        client._write_bucket._tokens = 1.0
        assert await client.acquire_order_token() is True
        client.halt("test")
        assert await client.place_order(order, "k-2", prepaid=True) is None
        assert client._write_bucket._tokens >= 0.999  # blocked -> token refunded
        assert await client.acquire_order_token() is False  # halted: nothing taken

    asyncio.run(scenario())
    assert len(session.calls) == 1
