"""research/reader.py: write guard, read budget, 429 handling, raw JSONL - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

import pytest

from pathlib import Path

from config import Settings
from exchange_api import ApiError, ExchangeClient
from research import reader as R


class FakeClock:
    """Monotonic clock + sleep that only advance when the reader sleeps."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeClient:
    """Answers GETs with a fixed body; raises the queued errors first."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = []

    async def _request(self, method, path, params=None, json_body=None):
        self.calls.append((method, path, params, json_body))
        if self.errors:
            raise self.errors.pop(0)
        if path == "/realtime/token":
            return {"token": "secret-jwt", "expiresAt": "2026-10-07T15:00:00Z"}
        return {"path": path, "params": params}


def _reader(tmp_path, client, rpm=10.0, clock=None):
    clock = clock or FakeClock()
    return R.ResearchReader(client, R.RawLog(tmp_path / "raw.jsonl"), rpm, tmp_path, clock=clock, sleep=clock.sleep), clock


def _429(retry_after=5.0):
    return ApiError(429, "RATE_LIMITED", "slow down", retry_after=retry_after)


# ---------------------------------------------------------------------- read budget


def test_reads_per_min_default_env_and_cli():
    assert R.resolve_reads_per_min(None, {}) == 10.0
    assert R.resolve_reads_per_min(None, {"RESEARCH_READS_PER_MIN": "12"}) == 12.0
    assert R.resolve_reads_per_min(5, {"RESEARCH_READS_PER_MIN": "12"}) == 5.0  # CLI wins


@pytest.mark.parametrize("cli, env", [(16, {}), (0, {}), (-1, {}), (None, {"RESEARCH_READS_PER_MIN": "20"}), (None, {"RESEARCH_READS_PER_MIN": "abc"}), (None, {"RESEARCH_READS_PER_MIN": "nan"})])
def test_reads_per_min_hard_max_is_never_clamped(cli, env):
    with pytest.raises(ValueError):
        R.resolve_reads_per_min(cli, env)


def test_reader_refuses_a_rate_above_the_hard_max(tmp_path):
    with pytest.raises(ValueError):
        R.ResearchReader(FakeClient(), R.RawLog(tmp_path / "raw.jsonl"), 15.5, tmp_path)


def test_estimate_runtime_and_yes_gate(caplog):
    estimate = R.ReadEstimate(reads=11, reads_per_min=10, extra_s=30)
    assert estimate.runtime_s == pytest.approx(10 * 6 + 30)
    assert R.ReadEstimate(reads=0, reads_per_min=10).runtime_s == 0
    with caplog.at_level("INFO", logger="research"):
        assert R.confirm_fetch(estimate, yes=False) is False
        assert R.confirm_fetch(estimate, yes=True) is True
    assert any(r.getMessage() == "Planned SIG reads" and r.reads == 11 for r in caplog.records)


def test_research_args():
    parser = argparse.ArgumentParser()
    R.add_research_args(parser)
    args = parser.parse_args([])
    assert args.yes is False and args.reads_per_min is None
    assert parser.parse_args(["--yes", "--reads-per-min", "12"]).reads_per_min == 12.0


# ---------------------------------------------------------------------- spacing and 429


def test_reads_are_spaced_by_the_research_rate(tmp_path):
    reader, clock = _reader(tmp_path, FakeClient(), rpm=10)

    async def run():
        for i in range(4):
            await reader.get(f"/x/{i}")

    asyncio.run(run())
    assert clock.sleeps == pytest.approx([6.0, 6.0, 6.0])  # first read at once, then 60 / 10 s
    assert reader.requests == 4


def test_429_pauses_for_retry_after_then_retries(tmp_path):
    client = FakeClient(errors=[_429(retry_after=42.0)])
    reader, clock = _reader(tmp_path, client)
    body = asyncio.run(reader.get("/leaderboard", {"period": "all"}))
    assert body["path"] == "/leaderboard"
    assert clock.sleeps == pytest.approx([42.0])
    assert reader.rate_limited == 1 and reader.requests == 1
    assert len(client.calls) == 2


def test_429_without_retry_after_waits_the_default(tmp_path):
    reader, clock = _reader(tmp_path, FakeClient(errors=[_429(retry_after=None)]))
    asyncio.run(reader.get("/x"))
    assert clock.sleeps == pytest.approx([R.DEFAULT_RETRY_AFTER_S])


def test_repeated_429_aborts_the_run(tmp_path):
    client = FakeClient(errors=[_429(), _429(), _429(), _429()])
    reader, _ = _reader(tmp_path, client)
    with pytest.raises(R.ResearchAborted):
        asyncio.run(reader.get("/x"))
    assert len(client.calls) == R.MAX_CONSECUTIVE_429


def test_a_success_resets_the_429_streak(tmp_path):
    client = FakeClient(errors=[_429(), _429()])
    reader, _ = _reader(tmp_path, client)

    async def run():
        await reader.get("/a")  # two 429s, then ok
        client.errors = [_429(), _429()]
        await reader.get("/b")  # two more: still under the limit of 3 in a row

    asyncio.run(run())
    assert reader.rate_limited == 4 and reader.requests == 2


def test_other_errors_are_logged_and_raised(tmp_path):
    reader, _ = _reader(tmp_path, FakeClient(errors=[ApiError(404, "NOT_FOUND", "nope")]))
    with pytest.raises(ApiError):
        asyncio.run(reader.get("/missing"))
    rows = R.read_jsonl(tmp_path / "raw.jsonl")
    assert rows[0]["error"] == {"status": 404, "code": "NOT_FOUND"}


# ---------------------------------------------------------------------- raw JSONL


def test_raw_log_is_append_only_and_redacts_the_token(tmp_path):
    reader, _ = _reader(tmp_path, FakeClient())

    async def run():
        await reader.get("/a", {"limit": 5})
        await reader.post_realtime_token()

    asyncio.run(run())
    rows = R.read_jsonl(tmp_path / "raw.jsonl")
    assert [r["path"] for r in rows] == ["/a", "/realtime/token"]
    assert rows[0]["params"] == {"limit": 5} and "ts" in rows[0]
    assert rows[1]["body"]["token"] == "<redacted>" and rows[1]["body"]["expiresAt"]
    # A second reader on the same file appends, never truncates.
    reader2, _ = _reader(tmp_path, FakeClient())
    asyncio.run(reader2.get("/b"))
    assert [r["path"] for r in R.read_jsonl(tmp_path / "raw.jsonl")] == ["/a", "/realtime/token", "/b"]


def test_read_jsonl_skips_a_torn_last_line(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": ', encoding="utf-8")
    assert R.read_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert R.read_jsonl(tmp_path / "absent.jsonl") == []


def test_run_dir_is_script_and_utc_date(tmp_path):
    when = datetime(2026, 10, 7, 23, 30, tzinfo=timezone.utc)
    path = R.run_dir("wall_mid_check", when, root=tmp_path)
    assert path == tmp_path / "wall_mid_check" / "2026-10-07" and path.is_dir()


def test_write_csv_and_json(tmp_path):
    n = R.write_csv(tmp_path / "s.csv", [{"a": 1, "b": 2, "extra": 3}], ["a", "b"])
    assert n == 1 and (tmp_path / "s.csv").read_text(encoding="utf-8").splitlines() == ["a,b", "1,2"]
    R.write_json(tmp_path / "s.json", {"x": [1, 2]})
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8")) == {"x": [1, 2]}


# ---------------------------------------------------------------------- write guard


def _guarded_real_client(monkeypatch):
    """A real ExchangeClient (no session, nothing sent) with the guard installed; the
    original `_request` is replaced by a recorder first, so allowed calls never hit the net."""
    settings = Settings(sig_api_key="test-key", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="DEBUG",
                        log_dir=Path("."), tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    client = ExchangeClient(settings)
    sent = []

    async def recorder(method, path, params=None, json_body=None, **kwargs):
        sent.append((method, path))
        return {}

    monkeypatch.setattr(client, "_request", recorder)
    R.guard_client(client)
    return client, sent


@pytest.mark.parametrize("method", ["place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "acquire_order_token"])
def test_guarded_client_write_methods_raise(monkeypatch, method):
    client, sent = _guarded_real_client(monkeypatch)
    assert client.halted
    with pytest.raises(R.ReadOnlyViolation):
        asyncio.run(getattr(client, method)(None, "key"))
    assert sent == []


@pytest.mark.parametrize("verb, path", [("POST", "/orders"), ("POST", "/orders/multi-leg"), ("DELETE", "/orders/1"), ("POST", "/orders/cancel-all"), ("PATCH", "/x"), ("post", "/orders/batch")])
def test_guarded_client_refuses_any_non_get(monkeypatch, verb, path):
    client, sent = _guarded_real_client(monkeypatch)
    with pytest.raises(R.ReadOnlyViolation):
        asyncio.run(client._request(verb, path, json_body={}))
    assert sent == []


def test_guarded_client_allows_get_and_the_realtime_token(monkeypatch):
    client, sent = _guarded_real_client(monkeypatch)

    async def run():
        await client._request("GET", "/account")
        await client._request("POST", "/realtime/token", json_body={})

    asyncio.run(run())
    assert sent == [("GET", "/account"), ("POST", "/realtime/token")]


def test_research_package_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout"}
    for path in (Path(R.__file__).parent).glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert name.split(".")[0] not in forbidden, f"{path.name} imports {name}"


def test_iter_jsonl_streams_and_file_sha256_hashes_the_concatenation(tmp_path):
    import hashlib

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3', encoding="utf-8")  # a blank line and a torn last line
    b.write_text('{"b": 1}\n', encoding="utf-8")
    it = R.iter_jsonl(a)
    assert next(it) == {"a": 1} and list(it) == [{"a": 2}]
    assert R.file_sha256([a, b], chunk=4) == hashlib.sha256(a.read_bytes() + b.read_bytes()).hexdigest()
