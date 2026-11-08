"""Shared read pause after a 429. Scripted fake session - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

import exchange_api
import main
from exchange_api import ApiError, ReadPausedError
from test_cycle_resilience import SETTINGS
from test_exchange_http import _client


class _Response:
    def __init__(self, status, body, headers):
        self.status, self._body, self.headers, self.reason = status, body, headers, "x"

    async def json(self, content_type=None):
        return self._body


class _Request:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        await asyncio.sleep(0.01)  # a real round trip: concurrent requests are in flight together
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class ScriptedSession:
    """Answers each request with the next scripted (status, retry_after); 200 when empty."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls: list[tuple[float, str]] = []

    def request(self, method, url, params=None, json=None, timeout=None):
        path = url.split("/api/v1")[-1]
        self.calls.append((time.monotonic(), path))
        status, retry_after = self.script.pop(0) if self.script else (200, None)
        if status == 429:
            headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
            body = {"error": {"code": "RATE_LIMITED", "message": "Rate limit exceeded."}}
        else:
            headers, body = {}, {"positions": [], "bids": [], "asks": [], "data": [], "pagination": {}}
        return _Request(_Response(status, body, headers))

    async def close(self):
        pass


def _paused_client(script):
    session = ScriptedSession(script)
    return _client(session), session


def test_429_pauses_books_and_background_reads_and_logs_once(caplog):
    client, session = _paused_client([(429, 0.5)])

    async def scenario():
        with pytest.raises(ApiError):
            await client.get_leaderboard("t")  # 429 -> pause; non-essential: no retry while paused
        with pytest.raises(ReadPausedError):
            await client.get_orderbook("1")
        with pytest.raises(ReadPausedError):
            await client.get_tournament_fills("t")

    with caplog.at_level(logging.INFO, logger="exchange_api"):
        asyncio.run(scenario())
    assert [path for _, path in session.calls] == ["/tournaments/t/leaderboard"]  # nothing else went out
    pauses = [r for r in caplog.records if r.getMessage() == "SIG read pause"]
    assert len(pauses) == 1 and pauses[0].pause_s == 0.5 and "429" in pauses[0].reason


def test_essential_read_waits_out_the_pause_then_succeeds():
    client, session = _paused_client([(429, 0.3)])
    started = time.monotonic()
    positions = asyncio.run(client.get_positions())
    assert positions == []
    (first, _), (second, _) = session.calls
    assert second - first >= 0.29  # retried only after the shared pause, not before
    assert time.monotonic() - started < 1.5


def test_concurrent_429s_share_one_pause(caplog):
    client, session = _paused_client([(429, 0.3), (429, 0.3), (429, 0.3)])

    async def scenario():
        await asyncio.gather(client.get_positions(), client.get_pnl(), client.list_orders())

    with caplog.at_level(logging.INFO, logger="exchange_api"):
        asyncio.run(scenario())
    assert len([r for r in caplog.records if r.getMessage() == "SIG read pause"]) == 1
    retries = sorted(t for t, _ in session.calls)[3:]
    assert len(retries) == 3 and max(retries) - min(retries) < 0.1  # all resumed together at the pause end


def test_pause_is_capped_and_defaults_to_the_cap(monkeypatch):
    monkeypatch.setattr(exchange_api, "SIG_READ_PAUSE_MAX_S", 0.4)
    client, _ = _paused_client([])
    client._start_read_pause(999.0, "/x")
    assert 0.3 < client.read_pause_remaining() <= 0.4 + 1e-6
    client2, _ = _paused_client([])
    client2._start_read_pause(None, "/x")
    assert 0.3 < client2.read_pause_remaining() <= 0.4 + 1e-6


def test_startup_retry_uses_retry_after(caplog):
    bot = main.Bot(SETTINGS, max_cycles=1)
    calls = []

    async def call():
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise ApiError(429, "RATE_LIMITED", "Rate limit exceeded.", retry_after=0.2)
        return "ok"

    with caplog.at_level(logging.WARNING, logger="main"):
        result = asyncio.run(bot._essential_startup_call(call, "/portfolio/positions", time.monotonic() + 30))
    assert result == "ok"
    retry = next(r for r in caplog.records if r.getMessage() == "Startup request failed, retrying")
    assert retry.retry_in_s == 0.2  # the server's Retry-After, not the fixed 5 s first backoff


def test_paused_order_books_skip_their_orders_not_the_cycle(tmp_path):
    from test_sig_reads import BookCountingExchange, _strategy

    class Paused(BookCountingExchange):
        async def get_orderbook(self, exchange_id, depth=20):
            raise ReadPausedError(f"/exchanges/{exchange_id}/orderbook", 30.0)

    report = asyncio.run(_strategy(Paused(), tmp_path).run_cycle())
    assert report.skipped_no_book == 2 and report.planned == 0
