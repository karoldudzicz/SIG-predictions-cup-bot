"""Bot entry point: startup reconciliation, the cycle loop, heartbeat, graceful shutdown.

Lifecycle:
  1. STARTUP   - load config + market map, open API clients.
  2. RECONCILE - never trust pre-restart memory (Resilience #3):
                 cancel every resting order left over from before the restart, then
                 settle unresolved journal entries. Their effect (if the order went
                 through) is already in /portfolio/positions, which every cycle reads,
                 so they are marked resolved rather than resent. Transient failures of
                 the essential calls (cancel-all, positions) are retried for up to
                 STARTUP_RETRY_BUDGET_S; READY=1 is sent only after this succeeds.
  3. WAIT      - live mode only: sleep until the tournament opens.
  4. LOOP      - strategy.run_cycle() every CYCLE_INTERVAL_S, each bounded by a timeout.
                 A news_watcher headline match about a traded race wakes the loop
                 early, so the next cycle starts immediately.
                 After every successful cycle: heartbeat file + systemd WATCHDOG=1.
                 A hung cycle stops the pings, so systemd's watchdog restarts us
                 (Resilience #2). Too many consecutive failures -> exit non-zero so
                 systemd restarts us with a clean state.
  5. SHUTDOWN  - SIGINT/SIGTERM (or Ctrl+C on Windows): the exchange client is
                 halted immediately (no new orders, even mid-cycle), the current cycle
                 finishes its reads, then all open orders are cancelled, sessions closed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp

from config import (
    CONFIG_WARNINGS,
    DIRECTIONAL_LIMIT_MODE,
    DIRECTIONAL_STRESS_FRAC,
    DIRECTIONAL_STRESS_S,
    LADDER_BOOK_FRAC,
    LADDER_BUCKET_RESERVE,
    LADDER_ENABLED,
    LADDER_MAX_MARKETS,
    LADDER_OFFSETS,
    LADDER_SIZES,
    MARKOUT_ENABLED,
    QUOTER_ALLOW_ONE_SIDED,
    QUOTER_BOOK_FRAC,
    QUOTER_DRY_RUN,
    QUOTER_ENABLED,
    QUOTER_MARKET_ALLOWLIST,
    QUOTER_MAX_MARKETS,
    QUOTER_MIN_SPREAD_TICKS,
    QUOTER_SIZE,
    SIG_READ_BURST,
    SIG_READ_PAUSE_MAX_S,
    SIG_READS_PER_MIN,
    SIG_WRITE_BURST,
    SIG_WRITES_PER_MIN,
    NO_SUCCESS_ALERT_REPEAT_S,
    NO_SUCCESS_ALERT_S,
    SIG_GET_DEADLINE_S,
    SIG_GET_MAX_ATTEMPTS,
    SIG_GET_TIMEOUT_S,
    MIN_CYCLE_GAP_S,
    RELATIONSHIPS_CHECK_ENABLED,
    STARTUP_RETRY_BUDGET_S,
    TAKER_GROSS_FRAC,
    TOTAL_GROSS_EXPOSURE_FRAC,
    Settings,
    load_settings,
)
from exchange_api import ApiError, AuthError, ExchangeClient, RequestTimeoutError, TermsNotAcknowledgedError
from logging_config import setup_logging
from market_mapper import MarketMap
from news_watcher import NewsMatch, NewsWatcher
from oracle_api import KalshiClient, OracleError, OracleTimeoutError, PolymarketClient
from strategy import OrderJournal, Strategy

logger = logging.getLogger("main")

CYCLE_TIMEOUT_S = 120  # a cycle normally takes ~3 s; longer means something is stuck
MAX_CONSECUTIVE_FAILURES = 10
# Pause after the n-th consecutive failed cycle: 5, 10, 20, 40, 60, 60, ... s. Most
# failures are one slow SIG request, so the first retry comes quickly; a success resets it.
FAILURE_BACKOFF_BASE_S = 5
FAILURE_BACKOFF_MAX_S = 60
SHUTDOWN_CANCEL_TIMEOUT_S = 30
PRE_START_LOG_EVERY_S = 3600

EXIT_OK = 0
EXIT_TOO_MANY_FAILURES = 1
EXIT_NEEDS_HUMAN = 2  # auth / terms problems - restarting will not fix them
EXIT_STARTUP_FAILED = 3  # SIG stayed unreachable for the whole STARTUP_RETRY_BUDGET_S

# Startup may now wait minutes for SIG before READY=1. systemd's default start timeout
# (90 s for Type=notify) would kill it first, so reconciliation extends that timeout to
# the retry budget plus this margin (last attempt + its own request deadline).
STARTUP_EXTEND_MARGIN_S = 120
# Max gap between WATCHDOG=1 pings while the loop is alive (unit: WatchdogSec=300).
WATCHDOG_PING_S = 30


class StartupInterrupted(Exception):
    """A stop was requested while startup was waiting on SIG."""


class StartupBudgetExhausted(Exception):
    """An essential startup request kept timing out for the whole retry budget."""


# ---------------------------------------------------------------------- heartbeat / watchdog


def sd_notify(message: str) -> None:
    """Send a status line to systemd (Type=notify). No-op when not run under systemd."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address or not hasattr(socket, "AF_UNIX"):
        return
    if address.startswith("@"):
        address = "\0" + address[1:]  # abstract namespace socket
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode())
    except OSError as exc:
        logger.warning("sd_notify failed", extra={"notify_message": message, "error": repr(exc)})


def write_heartbeat(settings: Settings, cycle: int, report: dict[str, Any]) -> None:
    """Atomic JSON heartbeat for external monitors (and for a human: `cat logs/heartbeat.json`)."""
    path = settings.log_dir / "heartbeat.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cycle": cycle, "dry_run": settings.dry_run, "last_report": report},
            default=str,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def is_transient_api_failure(exc: BaseException) -> bool:
    """True for a pure SIG / oracle outage: request timeouts, retryable ApiError / OracleError
    (429, 5xx, REQUEST_IN_FLIGHT) and connection errors. Auth / terms are never transient."""
    if isinstance(exc, (AuthError, TermsNotAcknowledgedError)):
        return False
    if isinstance(exc, (ApiError, OracleError)):
        return exc.retryable
    return isinstance(exc, aiohttp.ClientError)


class OutageAlarm:
    """Logs ERROR "No successful cycle" once no cycle has succeeded for `after_s`, then
    again every `repeat_s` while it lasts. Logging only; never changes behaviour."""

    def __init__(self, after_s: float, repeat_s: float, now: float) -> None:
        self._after_s = after_s
        self._repeat_s = repeat_s
        self._last_success = now  # counts from loop start until the first success
        self._last_alert: float | None = None

    def record_success(self, now: float) -> None:
        self._last_success = now
        self._last_alert = None

    def maybe_alert(self, now: float, last_error: str, consecutive_failures: int) -> bool:
        quiet = now - self._last_success
        if quiet < self._after_s:
            return False
        if self._last_alert is not None and now - self._last_alert < self._repeat_s:
            return False
        self._last_alert = now
        logger.error(
            "No successful cycle",
            extra={
                "minutes_without_success": round(quiet / 60, 1),
                "last_error": last_error,
                "consecutive_failures": consecutive_failures,
            },
        )
        return True


def failure_backoff(consecutive_failures: int) -> float:
    """Seconds to wait before the next cycle after `consecutive_failures` failed ones (>= 1)."""
    return min(FAILURE_BACKOFF_BASE_S * 2 ** (consecutive_failures - 1), FAILURE_BACKOFF_MAX_S)


def _log_news_task_exit(task: asyncio.Task[None]) -> None:
    # The watcher is optional: if it dies, trading continues on the normal interval.
    if not task.cancelled() and task.exception() is not None:
        logger.error("News watcher crashed; continuing without it", extra={"error": repr(task.exception())})


# ---------------------------------------------------------------------- bot


class Bot:
    def __init__(self, settings: Settings, max_cycles: int | None = None) -> None:
        self._settings = settings
        self._max_cycles = max_cycles
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()  # set by a stop request or a news match
        self._exchange: ExchangeClient | None = None

    def request_stop(self, reason: str) -> None:
        if not self._stop.is_set():
            logger.info("Stop requested", extra={"reason": reason})
            self._stop.set()
            self._wake.set()
        # Kill switch first: the running cycle may still be mid-EXECUTE, and it must
        # not place a single new order from this point on.
        if self._exchange is not None:
            self._exchange.halt(reason)

    def _get_stats(self) -> dict[str, Any]:
        """This cycle's GET counters (get_retries, slowest_get_s) for the failure log lines."""
        return self._exchange.get_stats() if self._exchange is not None else {}

    def _on_news(self, match: NewsMatch) -> None:
        logger.info("News trigger: starting next cycle early", extra={"races": match.race_keys, "headlines": match.feed})
        self._wake.set()

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop, sig.name)
            except (NotImplementedError, RuntimeError):
                # Windows: no add_signal_handler. Ctrl+C then cancels the main task and
                # the `finally` in run() still performs the graceful shutdown.
                pass

    async def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleep, but wake early on a stop request or a news trigger. True if woken early."""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
            woken = True
        except asyncio.TimeoutError:
            woken = False
        self._wake.clear()
        return woken

    async def _idle(self, seconds: float) -> None:
        """`_sleep_or_stop` that keeps the systemd watchdog fed (WATCHDOG=1 every <= WATCHDOG_PING_S).

        The pings come from the main loop itself, never from a background task, so a
        process whose loop is stuck still stops pinging and gets restarted.
        """
        end = time.monotonic() + seconds
        while True:
            sd_notify("WATCHDOG=1")
            remaining = end - time.monotonic()
            if remaining <= 0 or self._stop.is_set():
                return
            if await self._sleep_or_stop(min(remaining, WATCHDOG_PING_S)):
                return  # stop request or news trigger: start the next cycle now

    async def run(self) -> int:
        settings = self._settings
        logger.info(
            "Bot starting",
            extra={
                **settings.summary(),
                # Effective SIG GET behaviour (defaults or env overrides, see config.py).
                "sig_get_timeout_s": SIG_GET_TIMEOUT_S,
                "sig_get_max_attempts": SIG_GET_MAX_ATTEMPTS,
                "sig_get_deadline_s": SIG_GET_DEADLINE_S,
                "sig_writes_per_min": SIG_WRITES_PER_MIN,
                "sig_write_burst": SIG_WRITE_BURST,
                "sig_reads_per_min": SIG_READS_PER_MIN,
                "sig_read_burst": SIG_READ_BURST,
                "relationships_check": RELATIONSHIPS_CHECK_ENABLED,
                "ladder_bucket_reserve": LADDER_BUCKET_RESERVE,
                # Effective ladder settings (defaults or env overrides).
                "ladder_enabled": LADDER_ENABLED,
                "ladder_book_frac": LADDER_BOOK_FRAC,
                "total_gross_exposure_frac": TOTAL_GROSS_EXPOSURE_FRAC,
                # Effective taker sizing limits (defaults or env overrides).
                "taker_gross_frac": TAKER_GROSS_FRAC,
                "directional_limit_mode": DIRECTIONAL_LIMIT_MODE,
                "directional_stress_s": DIRECTIONAL_STRESS_S,
                "directional_stress_frac": DIRECTIONAL_STRESS_FRAC,
                "ladder_max_markets": LADDER_MAX_MARKETS,
                "ladder_offsets": LADDER_OFFSETS,
                "ladder_sizes": LADDER_SIZES,
                "markout_enabled": MARKOUT_ENABLED,
                "quoter_enabled": QUOTER_ENABLED,
                "quoter_max_markets": QUOTER_MAX_MARKETS,
                "quoter_size": QUOTER_SIZE,
                "quoter_book_frac": QUOTER_BOOK_FRAC,
                "quoter_dry_run": QUOTER_DRY_RUN,
                "quoter_min_spread_ticks": QUOTER_MIN_SPREAD_TICKS,
                "quoter_allow_one_sided": QUOTER_ALLOW_ONE_SIDED,
                "quoter_allowlist": sorted(QUOTER_MARKET_ALLOWLIST) if QUOTER_MARKET_ALLOWLIST is not None else None,
            },
        )
        if not settings.dry_run:
            logger.warning("DRY_RUN is OFF - this process will place real orders")

        market_map = MarketMap.load()
        journal = OrderJournal()

        async with ExchangeClient(settings) as exchange, PolymarketClient() as poly, KalshiClient() as kalshi:
            self._exchange = exchange
            if self._stop.is_set():
                exchange.halt("stop requested during startup")
            news_task: asyncio.Task[None] | None = None
            try:
                await self._reconcile(exchange, journal)
                strategy = Strategy(settings, exchange, poly, kalshi, market_map, journal)
                sd_notify("READY=1")
                await self._wait_for_tournament_start(exchange)
                news_task = asyncio.create_task(NewsWatcher(market_map, on_match=self._on_news).run(self._stop))
                news_task.add_done_callback(_log_news_task_exit)
                return await self._loop(strategy)
            except (TermsNotAcknowledgedError, AuthError) as exc:
                logger.critical("Needs human action, stopping", extra={"code": exc.code, "detail": exc.message})
                return EXIT_NEEDS_HUMAN
            except StartupInterrupted:
                logger.info("Stop requested during startup, exiting")
                return EXIT_OK
            except StartupBudgetExhausted:
                return EXIT_STARTUP_FAILED
            finally:
                # Covers every exit path, incl. Windows Ctrl+C (task cancellation, no
                # signal handler) and fatal errors: nothing new may be placed from here.
                exchange.halt("shutdown")
                sd_notify("STOPPING=1")
                if news_task is not None:
                    news_task.cancel()
                await self._shutdown(exchange)

    async def _until_stopped(self, awaitable: Any) -> Any:
        """Await `awaitable`, but abandon it as soon as a stop is requested (StartupInterrupted)."""
        task = asyncio.ensure_future(awaitable)
        stop = asyncio.ensure_future(self._stop.wait())
        try:
            done, _ = await asyncio.wait({task, stop}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop.cancel()
        if task in done:
            return task.result()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)  # let it unwind; its outcome no longer matters
        raise StartupInterrupted

    async def _essential_startup_call(self, call: Any, path: str, deadline: float) -> Any:
        """Run an essential startup request, retrying transient failures until `deadline`.

        Retried: every transient API failure (is_transient_api_failure: timeouts, retryable
        429 / 5xx ApiError, aiohttp connection errors). Backoff 5, 10, 20, 40, 60, 60... s,
        one WARNING per retry. A stop request ends both the in-flight request and the wait
        at once. Auth / terms errors and anything non-transient propagate immediately.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._until_stopped(call())
            except (ApiError, aiohttp.ClientError) as exc:
                if not is_transient_api_failure(exc):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.critical(
                        "Startup retry budget exhausted, exiting",
                        extra={"path": path, "attempts": attempt, "budget_s": STARTUP_RETRY_BUDGET_S, "error": repr(exc)},
                    )
                    raise StartupBudgetExhausted(path) from exc
                delay = min(failure_backoff(attempt), remaining)
                if isinstance(exc, ApiError) and exc.status == 429 and exc.retry_after is not None:
                    delay = min(exc.retry_after, SIG_READ_PAUSE_MAX_S, remaining)  # the server says when
                logger.warning(
                    "Startup request failed, retrying",
                    extra={
                        "path": path,
                        "attempt": attempt,
                        "error": repr(exc),
                        "retry_in_s": round(delay, 1),
                        "budget_left_s": round(remaining, 1),
                    },
                )
                await self._sleep_or_stop(delay)
                if self._stop.is_set():
                    raise StartupInterrupted from exc

    async def _reconcile(self, exchange: ExchangeClient, journal: OrderJournal) -> None:
        deadline = time.monotonic() + STARTUP_RETRY_BUDGET_S
        sd_notify(f"EXTEND_TIMEOUT_USEC={int((STARTUP_RETRY_BUDGET_S + STARTUP_EXTEND_MARGIN_S) * 1_000_000)}")

        # Resting orders from before the restart were sized on old prices - remove them.
        # (No-op in DRY_RUN: we never placed any.) Essential: retried within the budget.
        await self._essential_startup_call(exchange.cancel_all_orders, "/orders/cancel-all", deadline)

        unresolved = journal.unresolved()
        for entry in unresolved:
            journal.mark_resolved(
                entry["key"],
                note=f"startup reconciliation: was {entry['status']}; any fill is reflected in positions, resting remainder cancelled",
            )
        positions = await self._essential_startup_call(exchange.get_positions, "/portfolio/positions", deadline)

        # Non-essential: only feeds the log line below. Auth / terms and non-transient
        # errors still propagate.
        account_value = None
        try:
            pnl = await self._until_stopped(exchange.get_pnl())
            account_value = pnl.get("totalAccountValue")
        except (ApiError, aiohttp.ClientError) as exc:
            if not is_transient_api_failure(exc):
                raise
            logger.warning("Startup account value unavailable, continuing", extra={"path": "/portfolio/pnl", "error": repr(exc)})
        logger.info(
            "Startup reconciliation done",
            extra={
                "unresolved_journal_entries": len(unresolved),
                "positions": len(positions),
                "account_value": account_value,
            },
        )

    async def _wait_for_tournament_start(self, exchange: ExchangeClient) -> None:
        if self._settings.dry_run:
            return  # dry runs are useful before the start too
        tournament = await exchange.get_tournament(self._settings.tournament_slug)
        start = tournament.get("startDate")
        if not start:
            return
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        last_log = 0.0
        while not self._stop.is_set():
            wait = (start_dt - datetime.now(timezone.utc)).total_seconds()
            if wait <= 0:
                return
            if time.monotonic() - last_log >= PRE_START_LOG_EVERY_S:
                logger.info("Waiting for tournament start", extra={"start": start, "hours_left": round(wait / 3600, 2)})
                last_log = time.monotonic()
            sd_notify("WATCHDOG=1")  # waiting is healthy, not hung
            await self._sleep_or_stop(min(wait, 60))

    async def _loop(self, strategy: Strategy) -> int:
        cycle = 0
        failures = 0  # consecutive failed cycles of any kind - drives the backoff
        escalating = 0  # consecutive failures that are NOT a pure API outage - drives the exit
        alarm = OutageAlarm(NO_SUCCESS_ALERT_S, NO_SUCCESS_ALERT_REPEAT_S, now=time.monotonic())
        while not self._stop.is_set():
            cycle += 1
            started = time.monotonic()
            failure: BaseException | None = None
            api_outage = False
            # Our own deadline object, so its expiry can be told apart from a TimeoutError
            # raised inside the cycle (on 3.11+ both are the same TimeoutError class).
            cycle_deadline = asyncio.timeout(CYCLE_TIMEOUT_S)
            try:
                # Not cancelled by a stop request: a cycle that is placing orders is
                # allowed to finish (bounded by the timeout) before we shut down.
                async with cycle_deadline:
                    report = await strategy.run_cycle()
            except (TermsNotAcknowledgedError, AuthError):
                raise
            except (RequestTimeoutError, OracleTimeoutError, TimeoutError) as exc:
                failures += 1
                failure = exc
                if isinstance(exc, TimeoutError) and cycle_deadline.expired():
                    # Our own 120 s deadline: the cycle hung - not classified as an API outage.
                    logger.error(
                        "Cycle timed out",
                        extra={"cycle": cycle, "timeout_s": CYCLE_TIMEOUT_S, "consecutive_failures": failures, **self._get_stats()},
                    )
                else:
                    api_outage = True
                    logger.error(
                        "Cycle failed: API request timed out",
                        extra={
                            "cycle": cycle,
                            "method": getattr(exc, "method", None),
                            "path": getattr(exc, "path", None),
                            "attempts": getattr(exc, "attempts", None),
                            "elapsed_s": getattr(exc, "elapsed_s", None),
                            "cycle_elapsed_s": round(time.monotonic() - started, 2),
                            "consecutive_failures": failures,
                            **self._get_stats(),
                        },
                    )
            except (ApiError, Exception) as exc:  # noqa: BLE001 - the loop must survive anything
                failures += 1
                failure = exc
                api_outage = is_transient_api_failure(exc)
                logger.exception(
                    "Cycle failed",
                    extra={
                        "cycle": cycle,
                        "error": repr(exc),
                        "api_outage": api_outage,
                        "consecutive_failures": failures,
                        **self._get_stats(),
                    },
                )
            else:
                write_heartbeat(self._settings, cycle, report.__dict__)

            if failure is None:
                failures = escalating = 0
                alarm.record_success(time.monotonic())
            else:
                if not api_outage:
                    escalating += 1
                alarm.maybe_alert(time.monotonic(), repr(failure), failures)
            # Fed after EVERY iteration, failed or not: the watchdog is for a hung process
            # (no iteration at all), not for a slow or failing API.
            sd_notify("WATCHDOG=1")

            # A pure API outage (timeouts, retryable 429/5xx, connection errors) never ends
            # the process - it keeps retrying at <= 60 s. Anything else still escalates.
            if escalating >= MAX_CONSECUTIVE_FAILURES:
                logger.critical(
                    "Too many consecutive failures, exiting for a clean restart",
                    extra={"failures": escalating, "last_error": repr(failure)},
                )
                return EXIT_TOO_MANY_FAILURES
            if self._max_cycles is not None and cycle >= self._max_cycles:
                break

            if failures:
                # Never retry before an ongoing SIG read pause (after a 429) is over.
                delay = max(failure_backoff(failures), self._exchange.read_pause_remaining() if self._exchange else 0.0)
            else:
                delay = max(MIN_CYCLE_GAP_S, self._settings.cycle_interval_s - (time.monotonic() - started))
            await self._idle(delay)
        return EXIT_OK

    async def _shutdown(self, exchange: ExchangeClient) -> None:
        logger.info("Shutting down: cancelling open orders")
        try:
            await asyncio.wait_for(exchange.cancel_all_orders(), timeout=SHUTDOWN_CANCEL_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - best effort, but make it loud
            logger.critical(
                "Could not cancel open orders on shutdown - check the website",
                extra={"error": repr(exc)},
            )
        logger.info("Shutdown complete")


async def _amain(max_cycles: int | None) -> int:
    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    for problem in CONFIG_WARNINGS:  # collected at import, before logging existed
        logger.warning("Invalid config override, using default", extra=problem)
    bot = Bot(settings, max_cycles=max_cycles)
    bot.install_signal_handlers()
    return await bot.run()


def main(argv: list[str]) -> int:
    max_cycles = None
    if "--cycles" in argv:
        max_cycles = int(argv[argv.index("--cycles") + 1])
    try:
        return asyncio.run(_amain(max_cycles))
    except KeyboardInterrupt:
        # Windows Ctrl+C: asyncio.run already cancelled the bot, whose `finally`
        # ran the graceful shutdown before we got here.
        logging.getLogger("main").info("Interrupted")
        return EXIT_OK


if __name__ == "__main__":
    #   python main.py              run until stopped (Ctrl+C / SIGTERM)
    #   python main.py --cycles 2   run two cycles, then shut down (sanity check)
    raise SystemExit(main(sys.argv[1:]))
