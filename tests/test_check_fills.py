"""scripts/check_fills.py with a fake client - no network, and proof it never writes."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("check_fills", Path(__file__).resolve().parent.parent / "scripts" / "check_fills.py")
check_fills = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_fills)


def _fill(side, price):
    return {"id": 1, "orderId": 7, "exchangeId": "1075", "side": side, "quantity": -10 if side == "no" else 10, "price": price, "filledAt": "2026-10-01T17:00:00Z"}


@pytest.mark.parametrize(
    "fills, exchange_id, expected",
    [
        ([_fill("no", 0.08)], "1075", "Prices look YES-denominated"),
        ([_fill("no", 0.92)], "1075", "Prices look side-denominated"),
        ([_fill("yes", 0.40)], "1075", "Inconclusive: there are no NO-side fills"),
        ([], "1075", "Inconclusive: there are no NO-side fills"),
        ([_fill("no", 0.08), _fill("no", 0.92)], "1075", "Inconclusive: NO fills on both sides"),
        ([_fill("no", 0.08)], "999", "Inconclusive: NO fill prices"),
    ],
)
def test_conclusion(fills, exchange_id, expected):
    assert check_fills.conclusion(fills, exchange_id).startswith(expected)


class _ReadOnlyClient:
    """Fails the test if anything but the fills read is used."""

    def __init__(self, settings):
        self.reads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_tournament_fills(self, slug, cursor=None, limit=200, exchange_id=None):
        self.reads.append((slug, limit, exchange_id))
        return {"data": [_fill("no", 0.08)], "pagination": {"hasMore": False}}

    def __getattr__(self, name):
        raise AssertionError(f"check_fills must not call {name}")


def test_run_reads_once_and_logs_the_conclusion(monkeypatch, caplog):
    clients = []
    monkeypatch.setattr(check_fills, "ExchangeClient", lambda s: clients.append(_ReadOnlyClient(s)) or clients[-1])
    with caplog.at_level(logging.INFO, logger="check_fills"):
        assert asyncio.run(check_fills.run("1075", 5)) == 0
    assert clients[0].reads == [("midterm-elections", 5, "1075")]
    messages = [r.getMessage() for r in caplog.records]
    assert "Fill" in messages and any(m.startswith("Prices look YES-denominated") for m in messages)
