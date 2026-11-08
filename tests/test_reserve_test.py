"""scripts/reserve_test.py with a mocked client - no network."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("reserve_test", Path(__file__).resolve().parent.parent / "scripts" / "reserve_test.py")
reserve_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reserve_test)


class FakeClient:
    def __init__(self, open_orders=(), asks=(0.40,), balances=(1000.0, 999.0, 1000.0), fail_on_balance_call=None, resting=True):
        self.open_orders = list(open_orders)
        self.asks = list(asks)
        self.balances = list(balances)
        self.balance_calls = 0
        self.fail_on_balance_call = fail_on_balance_call
        self.resting = resting
        self.placed = []
        self.cancelled = []
        self.cancel_alls = []

    async def iter_orders(self, status="open", exchange_id=None, **kw):
        if exchange_id is None:
            for order in self.open_orders:
                yield order
        elif self.placed and self.resting:
            yield {"id": 555, "exchangeId": exchange_id, "quantity": 200}

    async def get_orderbook(self, exchange_id, depth=20):
        return {"bids": [{"price": 0.30, "quantity": 10}], "asks": [{"price": p, "quantity": 10} for p in self.asks]}

    async def get_tournament(self, slug):
        self.balance_calls += 1
        if self.balance_calls == self.fail_on_balance_call:
            raise RuntimeError("SIG went away")
        return {"id": "tid", "myBalance": self.balances[min(self.balance_calls, len(self.balances)) - 1]}

    async def place_order(self, order, key):
        self.placed.append((order, key))
        return {"orderId": 555, "open": True, "quantityTraded": 0}

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    async def cancel_all_orders(self, **scope):
        self.cancel_alls.append(scope)


@pytest.mark.parametrize("argv", [["--exchange-id", "1077"], ["--live", "--exchange-id", "1077"], ["--i-stopped-the-bot", "--exchange-id", "1077"]])
def test_refuses_without_both_flags(argv, monkeypatch):
    monkeypatch.setattr(reserve_test, "ExchangeClient", lambda s: pytest.fail("no client may be created"))
    assert reserve_test.main(argv) == reserve_test.EXIT_REFUSED


def test_aborts_when_open_orders_exist():
    client = FakeClient(open_orders=[{"id": 1}])
    assert asyncio.run(reserve_test.run_test(client, "1077")) == reserve_test.EXIT_ABORTED
    assert client.placed == [] and client.cancel_alls == []


@pytest.mark.parametrize("asks", [(0.02,), (0.01, 0.50), ()])
def test_aborts_when_best_ask_too_low_or_missing(asks):
    client = FakeClient(asks=asks)
    assert asyncio.run(reserve_test.run_test(client, "1077")) == reserve_test.EXIT_ABORTED
    assert client.placed == []


def test_safety_cancel_all_runs_after_an_exception():
    client = FakeClient(fail_on_balance_call=2)  # the read right after placing blows up
    with pytest.raises(RuntimeError):
        asyncio.run(reserve_test.run_test(client, "1077"))
    assert len(client.placed) == 1
    assert client.cancel_alls == [{"exchange_id": "1077", "tournament_id": "tid"}]


def test_safety_cancel_all_runs_after_cancellation():
    client = FakeClient()
    original = client.get_tournament

    async def slow_second_read(slug):
        if client.balance_calls == 1:
            await asyncio.sleep(10)  # the "SIGINT" lands here
        return await original(slug)

    client.get_tournament = slow_second_read

    async def scenario():
        task = asyncio.create_task(reserve_test.run_test(client, "1077"))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert client.cancel_alls == [{"exchange_id": "1077", "tournament_id": "tid"}]


@pytest.mark.parametrize(
    "balances, resting, expected",
    [
        ((1000.0, 999.0, 1000.0), True, "Cash IS reserved"),
        ((1000.0, 1000.0, 1000.0), True, "Cash is NOT reserved"),
        ((1000.0, 999.5, 1000.0), True, "Inconclusive ("),
        ((1000.0, 999.0, 1000.0), False, "Inconclusive: the test order did not rest unfilled"),
    ],
)
def test_conclusions(balances, resting, expected, caplog):
    client = FakeClient(balances=balances, resting=resting)
    with caplog.at_level(logging.INFO, logger="reserve_test"):
        assert asyncio.run(reserve_test.run_test(client, "1077")) == reserve_test.EXIT_OK
    assert client.cancelled == [555] and len(client.cancel_alls) == 1
    order, key = client.placed[0]
    assert (order.side, order.action, order.quantity, order.price, order.tournament_id) == ("yes", "buy", 200, 0.005, "tid")
    assert key.startswith("reserve-test-")
    assert any(r.getMessage().startswith(expected) for r in caplog.records)
