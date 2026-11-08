"""READ-ONLY research harness shared by the research scripts (leaderboard, wall mid, tape, ...).

Three guarantees, in this order:

1. Nothing can trade. `guard_client()` engages the client's kill switch AND makes every
   write raise `ReadOnlyViolation`: the order / cancel methods are replaced on the instance,
   and `_request` refuses any method but GET. The one allowed non-GET is
   POST /realtime/token (a channel token for the realtime probe, not an order).
   Research code never gets the client itself, only a `ResearchReader` (GET + that token).

2. The live bot keeps its read budget. SIG allows 100 reads/min per account and the bot's
   bucket is 80/min, so research reads are spaced at RESEARCH_READS_PER_MIN (default 10,
   hard max 15; env or --reads-per-min). A 429 pauses every research read for its
   Retry-After (60 s if absent); MAX_CONSECUTIVE_429 in a row abort the run
   (`ResearchAborted`). Before any read, the script logs its estimated read count and
   runtime and fetches nothing without --yes.

3. Nothing is fetched twice. Every response (or error) is appended to
   data/edge_scan/<script>/<UTC date>/raw.jsonl (append-only, one JSON object per line);
   summaries go to CSV + JSON in the same folder (`write_csv`, `write_json`).

Typical script skeleton:

    parser = argparse.ArgumentParser(...)
    add_research_args(parser)
    args = parser.parse_args(argv)
    estimate = ReadEstimate(reads=..., reads_per_min=resolve_reads_per_min(args.reads_per_min))
    if not confirm_fetch(estimate, args.yes):
        return 0
    async with research_reader("my_script", estimate.reads_per_min) as reader:
        body = await reader.get("/tournaments/midterm-elections/leaderboard", {"period": "all"})
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exchange_api import ApiError, ExchangeClient
from market_mapper import DATA_DIR

logger = logging.getLogger("research")

OUT_ROOT = DATA_DIR / "edge_scan"
RESEARCH_READS_PER_MIN_DEFAULT = 10.0
RESEARCH_READS_PER_MIN_MAX = 15.0  # the live bot's bucket is 80 of the account's 100 reads/min
MAX_CONSECUTIVE_429 = 3
DEFAULT_RETRY_AFTER_S = 60.0  # SIG sends Retry-After: 60 when the account limit is hit
# The only non-GET a research script may send: a realtime channel token, not an order.
NON_GET_ALLOWLIST = frozenset({("POST", "/realtime/token")})
# Client methods that place or cancel orders; replaced by raisers on a guarded client.
_WRITE_METHODS = ("place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "acquire_order_token")
# Response keys never written to raw.jsonl (the realtime token is a live credential).
_REDACTED_KEYS = frozenset({"token", "accessToken", "access_token", "refreshToken", "refresh_token", "apiKey"})


class ReadOnlyViolation(RuntimeError):
    """A research script tried to send something other than a read."""


class ResearchAborted(RuntimeError):
    """Too many 429s in a row: the run stops so the live bot keeps its budget."""


# ---------------------------------------------------------------------- read budget


def resolve_reads_per_min(cli_value: float | None = None, environ: Mapping[str, str] | None = None) -> float:
    """--reads-per-min if given, else RESEARCH_READS_PER_MIN from the env, else the default.
    Anything outside (0, RESEARCH_READS_PER_MIN_MAX] raises ValueError - no silent clamp."""
    environ = os.environ if environ is None else environ
    if cli_value is not None:
        value, source = float(cli_value), "--reads-per-min"
    elif (environ.get("RESEARCH_READS_PER_MIN") or "").strip():
        source = "RESEARCH_READS_PER_MIN"
        try:
            value = float(environ["RESEARCH_READS_PER_MIN"].strip())
        except ValueError:
            raise ValueError(f"{source} is not a number: {environ['RESEARCH_READS_PER_MIN']!r}") from None
    else:
        return RESEARCH_READS_PER_MIN_DEFAULT
    if not 0 < value <= RESEARCH_READS_PER_MIN_MAX:  # NaN fails this too
        raise ValueError(f"{source} must be in (0, {RESEARCH_READS_PER_MIN_MAX:g}]: the live bot uses 80 of the account's 100 reads/min, got {value}")
    return value


@dataclass(frozen=True)
class ReadEstimate:
    """What a run will cost. `extra_s`: wall time not spent on reads (sampling pauses, a
    realtime subscription)."""

    reads: int
    reads_per_min: float
    extra_s: float = 0.0

    @property
    def runtime_s(self) -> float:
        # The first read goes out at once, each further one waits 60 / reads_per_min.
        return max(0, self.reads - 1) * 60.0 / self.reads_per_min + self.extra_s


def confirm_fetch(estimate: ReadEstimate, yes: bool) -> bool:
    """Log the planned cost; True only with --yes (otherwise nothing may be fetched)."""
    logger.info(
        "Planned SIG reads",
        extra={"reads": estimate.reads, "reads_per_min": estimate.reads_per_min, "est_runtime_min": round(estimate.runtime_s / 60.0, 1)},
    )
    if not yes:
        logger.info("Nothing fetched: re-run with --yes to spend this read budget")
    return yes


def add_research_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yes", action="store_true", help="actually fetch (without it: only print the read estimate)")
    parser.add_argument(
        "--reads-per-min",
        type=float,
        default=None,
        help=f"SIG read spacing (default: env RESEARCH_READS_PER_MIN or {RESEARCH_READS_PER_MIN_DEFAULT:g}; hard max {RESEARCH_READS_PER_MIN_MAX:g})",
    )


# ---------------------------------------------------------------------- write guard


def guard_client(client: ExchangeClient) -> ExchangeClient:
    """Make `client` unable to trade: kill switch on, order / cancel methods raise, and
    `_request` refuses every non-GET outside NON_GET_ALLOWLIST. Patches the instance only."""
    client.halt("research scripts are read-only")

    def _refuse(name: str) -> Callable[..., Awaitable[Any]]:
        async def refused(*args: Any, **kwargs: Any) -> Any:
            raise ReadOnlyViolation(f"{name}() called from a read-only research script")

        return refused

    for name in _WRITE_METHODS:
        setattr(client, name, _refuse(name))

    original = client._request

    async def guarded_request(method: str, path: str, *args: Any, **kwargs: Any) -> Any:
        verb = method.upper()
        if verb != "GET" and (verb, path) not in NON_GET_ALLOWLIST:
            raise ReadOnlyViolation(f"{verb} {path} refused: research scripts are read-only")
        return await original(method, path, *args, **kwargs)

    client._request = guarded_request  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------- raw log + outputs


def run_dir(script: str, now: datetime | None = None, root: Path = OUT_ROOT) -> Path:
    """data/edge_scan/<script>/<UTC date of `now`>/ (created)."""
    now = now or datetime.now(timezone.utc)
    path = root / script / now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if k in _REDACTED_KEYS else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RawLog:
    """Append-only JSONL of every response: {"ts", "method", "path", "params", "body" | "error"}."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records = 0

    def append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": utc_now_iso(), **_redact(dict(record))}, default=str, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.records += 1


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Every record of a JSONL file, one line at a time (never the whole file in memory); a torn
    last line (crash mid-write) is skipped."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable JSONL line", extra={"path": str(path)})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every record of a JSONL file; a torn last line (crash mid-write) is skipped."""
    return list(iter_jsonl(path))


def file_sha256(paths: Iterable[Path], chunk: int = 1 << 20) -> str:
    """One SHA-256 over the bytes of every file, read in chunks."""
    digest = hashlib.sha256()
    for p in paths:
        with p.open("rb") as fh:
            while block := fh.read(chunk):
                digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> int:
    """Write `rows` (extra keys ignored) to a CSV; returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------- reader


class ResearchReader:
    """The only SIG access a research script gets: spaced GETs and the realtime token.

    Reads go out at most `reads_per_min` per minute (first at once). A 429 pauses every
    read for Retry-After (DEFAULT_RETRY_AFTER_S if absent) and retries the same read;
    `max_consecutive_429` of them in a row raise ResearchAborted. Every response or error
    is appended to `raw_log`. Note: the client may retry a timed-out / 5xx GET itself, so
    a flaky network can cost a few reads more than counted here.
    """

    def __init__(
        self,
        client: Any,
        raw_log: RawLog,
        reads_per_min: float,
        out_dir: Path,
        max_consecutive_429: int = MAX_CONSECUTIVE_429,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not 0 < reads_per_min <= RESEARCH_READS_PER_MIN_MAX:
            raise ValueError(f"reads_per_min must be in (0, {RESEARCH_READS_PER_MIN_MAX:g}], got {reads_per_min}")
        self._client = client
        self.raw_log = raw_log
        self.out_dir = out_dir
        self.reads_per_min = reads_per_min
        self._interval = 60.0 / reads_per_min
        self._max_429 = max_consecutive_429
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        self._lock = asyncio.Lock()
        self._consecutive_429 = 0
        self.requests = 0
        self.rate_limited = 0
        self.errors = 0

    async def _throttle(self) -> None:
        # Serialised: one research read at a time, never a burst.
        now = self._clock()
        delay = self._next_slot - now
        if delay > 0:
            await self._sleep(delay)
        self._next_slot = max(self._clock(), self._next_slot) + self._interval

    def _pause_all(self, seconds: float) -> None:
        self._next_slot = max(self._next_slot, self._clock() + seconds)

    async def _send(self, method: str, path: str, params: dict[str, Any] | None, json_body: Any = None) -> Any:
        async with self._lock:
            while True:
                await self._throttle()
                try:
                    if json_body is None:
                        body = await self._client._request(method, path, params=params)
                    else:
                        body = await self._client._request(method, path, params=params, json_body=json_body)
                except ApiError as exc:
                    if exc.status != 429:
                        self.errors += 1
                        self.raw_log.append({"method": method, "path": path, "params": params, "error": {"status": exc.status, "code": exc.code}})
                        raise
                    self.rate_limited += 1
                    self._consecutive_429 += 1
                    self.raw_log.append({"method": method, "path": path, "params": params, "error": {"status": 429, "code": exc.code, "retry_after": exc.retry_after}})
                    if self._consecutive_429 >= self._max_429:
                        logger.error("Aborting research run: repeated 429", extra={"path": path, "consecutive_429": self._consecutive_429})
                        raise ResearchAborted(f"{self._consecutive_429} consecutive 429 responses (last on {method} {path})") from exc
                    wait = exc.retry_after if exc.retry_after is not None else DEFAULT_RETRY_AFTER_S
                    logger.warning("Rate limited, pausing research reads", extra={"path": path, "wait_s": wait, "consecutive_429": self._consecutive_429})
                    self._pause_all(wait)
                    continue
                self._consecutive_429 = 0
                self.requests += 1
                self.raw_log.append({"method": method, "path": path, "params": params, "body": body})
                return body

    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return await self._send("GET", path, dict(params or {}))

    async def post_realtime_token(self, body: Mapping[str, Any] | None = None) -> Any:
        """POST /realtime/token - the one allowed non-GET (a channel token, not an order).
        The token is redacted in raw.jsonl."""
        return await self._send("POST", "/realtime/token", None, dict(body or {}))

    def stats(self) -> dict[str, Any]:
        return {"requests": self.requests, "rate_limited": self.rate_limited, "errors": self.errors, "raw_records": self.raw_log.records}


@asynccontextmanager
async def research_reader(script: str, reads_per_min: float, now: datetime | None = None, root: Path = OUT_ROOT) -> AsyncIterator[ResearchReader]:
    """A guarded, DRY_RUN client wrapped in a ResearchReader writing to
    data/edge_scan/<script>/<UTC date>/raw.jsonl. `reader.settings` holds the settings."""
    from config import load_settings  # late: tests of this module need no .env

    settings = replace(load_settings(), dry_run=True)
    out_dir = run_dir(script, now, root)
    async with ExchangeClient(settings) as client:
        guard_client(client)
        reader = ResearchReader(client, RawLog(out_dir / "raw.jsonl"), reads_per_min, out_dir)
        reader.settings = settings  # type: ignore[attr-defined]
        try:
            yield reader
        finally:
            logger.info("Research reads done", extra={"script": script, "out_dir": str(out_dir), **reader.stats()})
