"""Self-cross guard: no order is sent that could trade against our own resting order.
Fake session - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from exchange_api import OrderRequest, SelfCrossError, _WriteBucket, crosses, yes_terms
from strategy import OrderStatus
from test_exchange_http import _client, _FakeSession
from test_write_bucket import ORDER_BODY, _planned, _strategy


def _order(side, action, price, exchange_id="1"):
    return OrderRequest(exchange_id=exchange_id, side=side, action=action, quantity=10, price=price)


def _resting(order_id, side, action, price, exchange_id="1"):
    return {"id": order_id, "exchangeId": exchange_id, "side": side, "action": action, "priceLimit": price, "quantity": 10, "open": True}


def test_yes_terms():
    assert yes_terms("yes", "buy", 0.60) == ("bid", 0.60)
    assert yes_terms("no", "sell", 0.40) == ("bid", 0.60)  # sell no @p = buy yes @(1-p)
    assert yes_terms("no", "buy", 0.40) == ("ask", 0.60)
    assert yes_terms("yes", "sell", 0.60) == ("ask", 0.60)
    assert yes_terms("yes", "buy", None) == ("bid", 1.0)  # a market buy crosses any ask


@pytest.mark.parametrize(
    ("resting_no_price", "blocked"),
    [(0.40, True), (0.45, True), (0.395, False)],  # BUY YES @0.60 vs own BUY NO @(1-0.60) or higher
)
def test_buy_yes_against_own_buy_no(resting_no_price, blocked):
    client = _client(_FakeSession(delays=[]), dry_run=False)
    client.note_open_orders([_resting(7, "no", "buy", resting_no_price)])
    assert (client.self_cross(_order("yes", "buy", 0.60)) == "7") is blocked


def test_sell_yes_against_own_buy_yes_and_other_exchanges():
    client = _client(_FakeSession(delays=[]), dry_run=False)
    client.note_open_orders([_resting(7, "yes", "buy", 0.50), _resting(8, "no", "buy", 0.10, exchange_id="2")])
    assert client.self_cross(_order("yes", "sell", 0.50)) == "7"
    assert client.self_cross(_order("yes", "sell", 0.505)) is None
    assert client.self_cross(_order("yes", "buy", 0.99)) is None  # exchange 1 has no own ask
    assert crosses(("bid", 0.5), ("bid", 0.6)) is False


def test_place_order_is_blocked_before_any_request(caplog):
    session = _FakeSession(delays=[0.0], body=ORDER_BODY)
    client = _client(session, dry_run=False)
    client._write_bucket = _WriteBucket(per_minute=1e-4, burst=1)
    client.note_open_orders([_resting(7, "no", "buy", 0.40)])

    async def scenario():
        assert await client.acquire_order_token() is True
        with pytest.raises(SelfCrossError) as raised:
            await client.place_order(_order("yes", "buy", 0.60), "k-1", prepaid=True)
        return raised.value

    with caplog.at_level(logging.WARNING, logger="exchange_api"):
        error = asyncio.run(scenario())
    assert session.calls == []
    assert (error.code, error.resting_order_id, error.retryable) == ("SELF_CROSS_BLOCKED", "7", False)
    assert client._write_bucket._tokens >= 0.999  # the prepaid token came back
    record = next(r for r in caplog.records if r.getMessage() == "self_cross_blocked")
    assert record.levelno == logging.WARNING and record.resting_order_id == "7"


def test_dry_run_is_guarded_too():
    client = _client(_FakeSession(delays=[]), dry_run=True)
    client.note_open_orders([_resting(7, "no", "buy", 0.40)])
    with pytest.raises(SelfCrossError):
        asyncio.run(client.place_order(_order("yes", "buy", 0.60), "k-1"))


def test_batch_sends_only_non_crossing_items_and_keeps_indexes():
    body = {"results": [{"index": 0, "ok": True, "status": 200, "data": {**ORDER_BODY, "orderId": 11, "open": True, "remainingQuantity": 10}}]}
    session = _FakeSession(delays=[0.0], body=body)
    client = _client(session, dry_run=False)
    client.note_open_orders([_resting(7, "no", "buy", 0.40)])
    orders = [
        _order("yes", "buy", 0.60),  # crosses resting order 7
        _order("yes", "buy", 0.50),  # fine
        _order("no", "buy", 0.55),  # YES ask 0.45 <= the 0.50 bid just above -> crosses item 1
    ]
    results = asyncio.run(client.place_batch(orders, "b-1"))
    assert [r["index"] for r in results] == [0, 1, 2]
    assert [r["ok"] for r in results] == [False, True, False]
    assert results[0]["data"]["code"] == results[2]["data"]["code"] == "SELF_CROSS_BLOCKED"
    assert [o["price"] for o in session.bodies[0]["orders"]] == [0.50]  # one item sent
    assert client.self_cross(_order("no", "buy", 0.55)) == "11"  # the placed order is now resting


def test_registry_follows_cancels():
    session = _FakeSession(delays=[0.0, 0.0], body={"cancelled": 1, "errors": []})
    client = _client(session, dry_run=False)
    client.note_open_orders([_resting(7, "no", "buy", 0.40), _resting(8, "no", "buy", 0.40, exchange_id="2")])
    asyncio.run(client.cancel_all_orders(exchange_id="1"))
    assert client.self_cross(_order("yes", "buy", 0.60)) is None
    assert client.self_cross(_order("yes", "buy", 0.60, exchange_id="2")) == "8"
    session.body = {"orderId": 8}
    asyncio.run(client.cancel_order(8))
    assert client.self_cross(_order("yes", "buy", 0.60, exchange_id="2")) is None


def test_taker_order_that_would_self_cross_is_rejected_not_sent(tmp_path):
    session = _FakeSession(delays=[0.0], body=ORDER_BODY)
    client = _client(session, dry_run=False)
    client.note_open_orders([_resting(7, "no", "buy", 0.55)])  # own YES ask at 0.45
    taker = _strategy(client, tmp_path)
    outcome = asyncio.run(taker._execute(_planned(time.time())))  # BUY YES @0.45
    assert outcome is OrderStatus.REJECTED
    assert session.calls == []
    journal = (tmp_path / "j.jsonl").read_text(encoding="utf-8")
    assert '"code": "SELF_CROSS_BLOCKED"' in journal


# ---------------------------------------------------------------------- own orders out of the book


def test_book_without_own_orders():
    from exchange_api import book_without_own

    book = {
        "bids": [{"price": 0.50, "quantity": 100}, {"price": 0.495, "quantity": 40}],
        "asks": [{"price": 0.52, "quantity": 30}, {"price": 0.53, "quantity": 80}],
    }
    own = [("bid", 0.50, 100), ("ask", 0.53, 20), ("ask", 0.52, 10)]  # e.g. our BUY NO @0.47 is a YES ask at 0.53
    other = book_without_own(book, own)
    assert other["bids"] == [{"price": 0.495, "quantity": 40}]  # our whole 0.50 bid removed: best other bid 0.495
    assert other["asks"] == [{"price": 0.52, "quantity": 20}, {"price": 0.53, "quantity": 60}]
    assert book["bids"][0]["quantity"] == 100  # the input is not modified


def test_own_orders_feed_book_without_own():
    from exchange_api import book_without_own

    client = _client(_FakeSession(delays=[]), dry_run=False)
    client.note_open_orders([_resting(7, "no", "buy", 0.47), _resting(8, "yes", "buy", 0.40, exchange_id="2")])
    assert client.own_orders("1") == [("ask", 0.53, 10.0)]
    other = book_without_own({"bids": [], "asks": [{"price": 0.53, "quantity": 10}]}, client.own_orders("1"))
    assert other["asks"] == []
