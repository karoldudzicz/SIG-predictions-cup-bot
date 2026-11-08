"""READ-ONLY: are REST fill prices YES-denominated or side-denominated?

Reads GET /tournaments/midterm-elections/portfolio/fills for one exchange and logs every
fill (id, side, quantity, price, time). Then one conclusion line: a NO-side fill whose
price is below 0.5, on a market where NO was bought at ~0.9 (exchange 1075 on day 1),
means prices are YES-denominated; above 0.5 means side-denominated. No NO fills, or NO
fills on both sides of 0.5, is reported as inconclusive instead of guessed.

Only one GET (plus pagination is not needed for small --limit). No writes, the kill
switch and the write bucket are not touched.

    python scripts/check_fills.py [--exchange-id 1075] [--limit 5]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from config import load_settings  # noqa: E402
from exchange_api import ApiError, ExchangeClient  # noqa: E402
from logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("check_fills")

TOURNAMENT_SLUG = "midterm-elections"
KNOWN_NO_BUY = {"1075": 0.92}  # day 1: the bot bought NO on 1075 (R wins Delaware Senate) at ~0.92


def conclusion(fills: list[dict[str, Any]], exchange_id: str) -> str:
    """One sentence; never guesses when the data cannot decide."""
    no_prices = [float(f["price"]) for f in fills if f.get("side") == "no" and f.get("price") is not None]
    if not no_prices:
        return "Inconclusive: there are no NO-side fills with a price in this data, so the denomination cannot be told."
    reference = KNOWN_NO_BUY.get(exchange_id)
    if reference is None:
        return (
            f"Inconclusive: NO fill prices {no_prices}, but the NO buy price on exchange {exchange_id} is not known, "
            "so they cannot be compared against it (use --exchange-id 1075)."
        )
    below = [p for p in no_prices if p < 0.5]
    above = [p for p in no_prices if p >= 0.5]
    if below and not above:
        return f"Prices look YES-denominated: NO fills at {below} vs a NO buy price of ~{reference}."
    if above and not below:
        return f"Prices look side-denominated: NO fills at {above} vs a NO buy price of ~{reference}."
    return f"Inconclusive: NO fills on both sides of 0.5 ({no_prices})."


async def run(exchange_id: str, limit: int) -> int:
    settings = load_settings()
    async with ExchangeClient(settings) as client:
        try:
            page = await client.get_tournament_fills(TOURNAMENT_SLUG, limit=limit, exchange_id=exchange_id)
        except ApiError as exc:
            logger.error("Fills read failed", extra={"status": exc.status, "code": exc.code, "detail": exc.message})
            return 1
    fills = page.get("data") or []
    logger.info("Fills read", extra={"exchange_id": exchange_id, "count": len(fills), "limit": limit})
    for fill in fills:
        logger.info(
            "Fill",
            extra={
                "fill_id": fill.get("id"),
                "order_id": fill.get("orderId"),
                "side": fill.get("side"),
                "quantity": fill.get("quantity"),
                "price": fill.get("price"),
                # The OpenAPI Fill schema names this field filledAt; createdAt is logged too if present.
                "filled_at": fill.get("filledAt"),
                "created_at": fill.get("createdAt"),
            },
        )
    logger.info(conclusion(fills, exchange_id))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only check of how REST fill prices are denominated.")
    parser.add_argument("--exchange-id", default="1075")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    return asyncio.run(run(str(args.exchange_id), args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
