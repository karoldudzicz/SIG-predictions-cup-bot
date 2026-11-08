"""LIVE WRITE TEST: does a resting order reserve cash? Maximum risk: 1 SUSQie.

The only script in this repo allowed to write to the live API, and only with BOTH
--live and --i-stopped-the-bot. With --live, DRY_RUN from .env is IGNORED (logged as a
WARNING) - that is the point of the test. Stop the bot first: it shares the account.

Steps:
  1. Abort if the account has ANY open order (the bot is probably still running).
  2. Abort if the exchange's book has no asks, or the best ask is <= 0.02 (our 0.005 bid
     must not be able to fill).
  3. Read myBalance (GET /tournaments/midterm-elections).
  4. Place exactly one BUY YES 200 x 0.005 (cost 1.00), with an idempotencyKey.
  5. Read the balance again; confirm in the open-order read that the order rests unfilled.
  6. Cancel that order by id. A `finally` block ALWAYS also sends a cancel-all scoped to
     this exchange as a safety net - after any exception, Ctrl+C or SIGTERM too.
  7. Read the balance a third time; log both deltas.
  8. Conclude: reserved (-1.00 after placing, back after cancelling) / not reserved
     (no change) / inconclusive (anything else, with the numbers).

Every write goes through ExchangeClient (write bucket, retries, idempotencyKey).

    python scripts/reserve_test.py --live --i-stopped-the-bot --exchange-id 1077

Exit codes: 0 conclusion logged, 1 error, 2 refused (flags missing), 3 aborted for safety.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from config import load_settings  # noqa: E402
from exchange_api import ExchangeClient, OrderRequest, new_idempotency_key  # noqa: E402
from logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("reserve_test")

TOURNAMENT_SLUG = "midterm-elections"
TEST_QUANTITY = 200
TEST_PRICE = 0.005  # 200 x 0.005 = 1.00 SUSQie
MIN_SAFE_ASK = 0.02
TOLERANCE = 0.02

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_ABORTED = 3


def conclude(before: float, after_place: float, after_cancel: float, rested_unfilled: bool) -> str:
    placed_delta = after_place - before
    cancel_delta = after_cancel - before
    numbers = f"(after placing {placed_delta:+.4f}, after cancelling {cancel_delta:+.4f})"
    cost = TEST_QUANTITY * TEST_PRICE
    if not rested_unfilled:
        return f"Inconclusive: the test order did not rest unfilled {numbers}."
    if abs(placed_delta + cost) <= TOLERANCE and abs(cancel_delta) <= TOLERANCE:
        return f"Cash IS reserved by resting orders {numbers}."
    if abs(placed_delta) <= TOLERANCE and abs(cancel_delta) <= TOLERANCE:
        return f"Cash is NOT reserved by resting orders {numbers}."
    return f"Inconclusive {numbers}."


async def _balance(client: ExchangeClient) -> tuple[float, str]:
    tournament = await client.get_tournament(TOURNAMENT_SLUG)
    return float(tournament["myBalance"]), tournament["id"]


async def run_test(client: ExchangeClient, exchange_id: str) -> int:
    # 1. The bot must be stopped: no open orders anywhere on the account.
    open_orders = [o async for o in client.iter_orders(status="open")]
    if open_orders:
        logger.error("Aborting: the account has open orders - is the bot still running?", extra={"open_orders": len(open_orders)})
        return EXIT_ABORTED

    # 2. Our 0.005 bid must not be fillable.
    book = await client.get_orderbook(exchange_id, depth=5)
    asks = [float(level["price"]) for level in book.get("asks") or []]
    if not asks:
        logger.error("Aborting: the book has no asks, cannot judge fill risk", extra={"exchange_id": exchange_id})
        return EXIT_ABORTED
    if min(asks) <= MIN_SAFE_ASK:
        logger.error("Aborting: best ask too low, the test order could fill", extra={"exchange_id": exchange_id, "best_ask": min(asks)})
        return EXIT_ABORTED

    # 3. Balance before.
    before, tournament_id = await _balance(client)
    logger.info("Balance before", extra={"my_balance": before})

    order_id: Any = None
    try:
        # 4. Exactly one BUY YES 200 x 0.005.
        order = OrderRequest(exchange_id=exchange_id, side="yes", action="buy", quantity=TEST_QUANTITY, price=TEST_PRICE, tournament_id=tournament_id)
        result = await client.place_order(order, new_idempotency_key("reserve-test"))
        if result is None:
            raise RuntimeError("test order was not sent (DRY_RUN or halted)")
        order_id = result.get("orderId")
        logger.info("Test order placed", extra={"order_id": order_id, "open": result.get("open"), "traded": result.get("quantityTraded")})

        # 5. Balance with the order resting; is it really resting, unfilled?
        after_place, _ = await _balance(client)
        logger.info("Balance with the order resting", extra={"my_balance": after_place})
        resting = [o async for o in client.iter_orders(status="open", exchange_id=exchange_id)]
        mine = next((o for o in resting if str(o.get("id")) == str(order_id)), None)
        rested_unfilled = (
            mine is not None and abs(mine.get("quantity") or 0) == TEST_QUANTITY and not (result.get("quantityTraded") or 0)
        )
        if not rested_unfilled:
            logger.warning("Test order is not resting unfilled", extra={"order_id": order_id, "open_order": mine})

        # 6. Cancel it by id.
        if order_id is not None:
            await client.cancel_order(order_id)
            logger.info("Test order cancelled", extra={"order_id": order_id})
    finally:
        # Safety net - runs after exceptions and after Ctrl+C / SIGTERM (task cancellation).
        try:
            await client.cancel_all_orders(exchange_id=exchange_id, tournament_id=tournament_id)
            logger.info("Safety cancel-all sent", extra={"exchange_id": exchange_id})
        except Exception as exc:  # noqa: BLE001 - must be loud, never silent
            logger.critical("Safety cancel-all FAILED - check the website now", extra={"exchange_id": exchange_id, "order_id": order_id, "error": repr(exc)})

    # 7. Balance after cancelling, and the deltas.
    after_cancel, _ = await _balance(client)
    logger.info(
        "Balance after cancelling",
        extra={"my_balance": after_cancel, "delta_after_place": round(after_place - before, 4), "delta_after_cancel": round(after_cancel - before, 4)},
    )
    # 8. Conclusion.
    logger.info(conclude(before, after_place, after_cancel, rested_unfilled))
    return EXIT_OK


async def _amain(exchange_id: str) -> int:
    settings = load_settings()
    if settings.dry_run:
        logger.warning("DRY_RUN=true from .env is IGNORED because --live was given: this script places a real order")
    settings = replace(settings, dry_run=False)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, task.cancel)  # cancellation -> the finally block still runs
        except (NotImplementedError, RuntimeError):
            pass  # Windows: Ctrl+C already cancels the task via asyncio.run
    async with ExchangeClient(settings) as client:
        return await run_test(client, exchange_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LIVE test: does a resting order reserve cash? Max risk 1 SUSQie.")
    parser.add_argument("--live", action="store_true", help="really place the test order (ignores DRY_RUN)")
    parser.add_argument("--i-stopped-the-bot", action="store_true", help="confirm the bot is stopped")
    parser.add_argument("--exchange-id", required=True)
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    if not (args.live and args.i_stopped_the_bot):
        logger.error("Refusing to run: both --live and --i-stopped-the-bot are required")
        return EXIT_REFUSED
    try:
        return asyncio.run(_amain(str(args.exchange_id)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("Interrupted; the safety cancel-all ran in the finally block")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
