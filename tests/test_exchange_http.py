"""SIG HTTP layer: short GET timeout with fast retry, unchanged POST timeout, GET-only
concurrency cap. A fake aiohttp session - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

import exchange_api
from config import SIG_POST_TIMEOUT_S, Settings
from exchange_api import ExchangeClient


class _FakeResponse:
    def __init__(self, body):
        self.status = 200
        self.headers = {}
        self.reason = "OK"
        self._body = body

    async def json(self, content_type=None):
        return self._body


class _FakeRequest:
    def __init__(self, session, delay, body):
        self._session, self._delay, self._body = session, delay, body

    async def __aenter__(self):
        self._session.in_flight += 1
        self._session.max_in_flight = max(self._session.max_in_flight, self._session.in_flight)
        try:
            await asyncio.sleep(self._delay)  # a "hang" is just a long delay, cut by our timeout
        finally:
            self._session.in_flight -= 1
        return _FakeResponse(self._body)

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    """Records (method, path, timeout) per attempt; delays[i] is the latency of attempt i."""

    def __init__(self, delays, body=None):
        self.delays = list(delays)
        self.body = body if body is not None else {"ok": True}
        self.calls: list[tuple[str, str, float | None]] = []
        self.bodies: list = []  # JSON body of each attempt
        self.in_flight = 0
        self.max_in_flight = 0

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append((method, url.split("/api/v1")[-1], timeout.total if timeout else None))
        self.bodies.append(json)
        delay = self.delays.pop(0) if self.delays else 0.0
        return _FakeRequest(self, delay, self.body)

    async def close(self):
        pass


def _client(session: _FakeSession, dry_run: bool = True) -> ExchangeClient:
    settings = Settings(
        sig_api_key="test-key",
        sig_base_url="https://sig.invalid/api/v1",
        dry_run=dry_run,
        log_level="DEBUG",
        log_dir=Path("."),
        tournament_slug="t",
        oracle_staleness_threshold_s=10.0,
        cycle_interval_s=15.0,
    )
    client = ExchangeClient(settings)
    client._session = session
    # Unthrottled reads by default; read-budget tests install their own bucket.
    client._read_bucket = exchange_api._WriteBucket(per_minute=600_000, burst=1_000)
    return client


def test_hung_get_is_abandoned_and_retried_fast(monkeypatch, caplog):
    """First GET attempt hangs, second answers at once: done in ~the short timeout, not 15 s."""
    monkeypatch.setattr(exchange_api, "SIG_GET_TIMEOUT_S", 0.3)
    session = _FakeSession(delays=[30.0, 0.01], body={"id": "me", "balance": 100000})
    client = _client(session)

    started = time.monotonic()
    with caplog.at_level(logging.DEBUG, logger="exchange_api"):
        account = asyncio.run(client.get_account())
    elapsed = time.monotonic() - started

    assert account["balance"] == 100000
    assert [c[0:2] for c in session.calls] == [("GET", "/account"), ("GET", "/account")]
    assert session.calls[0][2] == pytest.approx(0.3)  # short per-attempt timeout was used
    assert elapsed < 0.3 + exchange_api.GET_TIMEOUT_RETRY_WAIT_MAX_S + 0.5  # nowhere near 15 s
    retry = next(r for r in caplog.records if r.getMessage() == "Request failed, retrying")
    assert (retry.method, retry.path, retry.attempt) == ("GET", "/account", 1)
    assert "Authorization" not in str(retry.__dict__) and "test-key" not in str(retry.__dict__)
    done = next(r for r in caplog.records if r.getMessage() == "Request succeeded after retry")
    assert (done.method, done.path, done.attempts) == ("GET", "/account", 2)


def test_get_gives_up_within_the_deadline(monkeypatch, caplog):
    monkeypatch.setattr(exchange_api, "SIG_GET_TIMEOUT_S", 0.1)
    session = _FakeSession(delays=[30.0] * 10)
    client = _client(session)
    with caplog.at_level(logging.ERROR, logger="exchange_api"), pytest.raises(exchange_api.RequestTimeoutError) as raised:
        asyncio.run(client.get_account())
    assert not isinstance(raised.value, TimeoutError)  # main.py must not mistake it for its cycle timeout
    assert (raised.value.method, raised.value.path) == ("GET", "/account")
    assert len(session.calls) == exchange_api.SIG_GET_MAX_ATTEMPTS
    failure = next(r for r in caplog.records if r.getMessage() == "Request failed, giving up")
    assert (failure.method, failure.path, failure.attempts) == ("GET", "/account", exchange_api.SIG_GET_MAX_ATTEMPTS)


def test_post_order_keeps_the_long_timeout(monkeypatch):
    """A POST slower than the GET timeout still succeeds on its first attempt."""
    monkeypatch.setattr(exchange_api, "SIG_GET_TIMEOUT_S", 0.1)
    body = {"orderId": 1, "exchangeId": "1075", "action": "buy", "side": "no", "price": 0.92, "quantity": 10,
            "quantityTraded": 10, "remainingQuantity": 0, "open": False}
    session = _FakeSession(delays=[0.5], body=body)
    client = _client(session, dry_run=False)
    order = exchange_api.OrderRequest(exchange_id="1075", side="no", action="buy", quantity=10, price=0.92)

    result = asyncio.run(client.place_order(order, "key-1"))

    assert result["orderId"] == 1
    assert session.calls == [("POST", "/orders", SIG_POST_TIMEOUT_S)]


def test_get_concurrency_is_capped():
    session = _FakeSession(delays=[0.1] * 20)
    client = _client(session)

    async def burst():
        await asyncio.gather(*(client.get_orderbook(str(i)) for i in range(20)))

    asyncio.run(burst())
    assert session.max_in_flight <= exchange_api.SIG_MAX_CONCURRENCY
    assert len(session.calls) == 20


def test_cancel_all_never_waits_for_the_get_semaphore():
    """Kill switch path: even with every GET slot taken, cancel-all goes straight out."""
    session = _FakeSession(delays=[0.01], body={"cancelled": 0, "errors": []})
    client = _client(session, dry_run=False)

    async def scenario():
        for _ in range(exchange_api.SIG_MAX_CONCURRENCY):
            await client._get_semaphore.acquire()  # all GET slots busy
        return await asyncio.wait_for(client.cancel_all_orders(), timeout=2)

    assert asyncio.run(scenario()) == {"cancelled": 0, "errors": []}
    assert session.calls == [("POST", "/orders/cancel-all", SIG_POST_TIMEOUT_S)]


# ---------------------------------------------------------------------- request rate visibility


def test_rate_counters_and_headers(caplog):
    session = _FakeSession(delays=[0.0, 0.0, 0.0])
    client = _client(session, dry_run=False)

    async def scenario():
        await client.get_account()
        await client.get_account()
        await client.cancel_all_orders()

    with caplog.at_level(logging.DEBUG, logger="exchange_api"):
        asyncio.run(scenario())
    stats = client.get_stats()
    assert (stats["reads_last_min"], stats["writes_last_min"], stats["rate_limited_last_min"]) == (2, 1, 0)
    assert not any(r.getMessage() == "Rate-limit headers" for r in caplog.records)  # none sent, none assumed


def test_rate_limit_headers_logged_when_present(caplog):
    class _Resp:
        status = 429
        headers = {"X-RateLimit-Remaining": "0", "Retry-After": "60", "Content-Type": "application/json"}

    with caplog.at_level(logging.INFO, logger="exchange_api"):
        exchange_api._log_rate_headers("POST", "/orders", _Resp())
    record = next(r for r in caplog.records if r.getMessage() == "Rate-limit headers")
    assert record.levelno == logging.INFO and record.headers == {"X-RateLimit-Remaining": "0", "Retry-After": "60"}


def test_request_rate_logged_once_a_minute(monkeypatch, caplog):
    session = _FakeSession(delays=[0.0] * 3)
    client = _client(session)
    client._rate_logged_at -= 61  # a minute has passed since the last summary
    with caplog.at_level(logging.INFO, logger="exchange_api"):
        asyncio.run(client.get_account())
        asyncio.run(client.get_account())
    summaries = [r for r in caplog.records if r.getMessage() == "SIG request rate"]
    assert len(summaries) == 1 and summaries[0].reads_last_min == 1
