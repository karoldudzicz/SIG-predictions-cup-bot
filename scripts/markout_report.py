"""READ-ONLY: markout summary per fill source (taker / ladder / quoter) from data/markout.jsonl.

For every source and horizon (1, 5, 30 min): fills measured, mean markout in ticks, share
of negative markouts (we were picked off), and how many horizons had no fresh fair (null).
Optionally per exchange too. Reads the journal only - no API access at all.

Markouts of NO fills journalled before the REST price convention was fixed (fill lines
without `price_source`) are recomputed from the corrected price (markout.legacy_price_yes);
`corrected` counts them per row.

    python scripts/markout_report.py [--path data/markout.jsonl] [--by-exchange]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from config import load_settings  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from markout import MARKOUT_JOURNAL_PATH, legacy_price_yes, long_yes_direction, markout_ticks  # noqa: E402

logger = logging.getLogger("markout_report")


def summarize(lines: list[str], by_exchange: bool = False) -> list[dict[str, Any]]:
    """One row per (source[, exchange], horizon): n, mean_ticks, negative_share, no_fair, corrected."""
    groups: dict[tuple, list[float | None]] = defaultdict(list)
    corrected: dict[tuple, int] = defaultdict(int)
    fills: dict[Any, dict[str, Any]] = {}
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") == "fill":
            fills[entry.get("fill_id")] = entry
            continue
        if entry.get("event") != "markout":
            continue
        key = (entry["source"], entry["exchange_id"] if by_exchange else None, entry["horizon_s"])
        ticks = entry.get("markout_ticks")
        fill = fills.get(entry.get("fill_id"))
        if fill is not None and fill.get("price_source") is None and fill.get("side") == "no":
            direction = long_yes_direction(fill["side"], fill.get("action"))
            ticks = markout_ticks(direction, legacy_price_yes(fill), entry.get("fair"))
            corrected[key] += 1
        groups[key].append(ticks)
    rows = []
    for (source, exchange_id, horizon), values in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "", kv[0][2])):
        measured = [v for v in values if v is not None]
        rows.append(
            {
                "source": source,
                **({"exchange_id": exchange_id} if by_exchange else {}),
                "horizon_s": horizon,
                "n": len(measured),
                "mean_ticks": round(sum(measured) / len(measured), 2) if measured else None,
                "negative_share": round(sum(1 for v in measured if v < 0) / len(measured), 3) if measured else None,
                "no_fair": len(values) - len(measured),
                "corrected": corrected[(source, exchange_id, horizon)],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Markout summary per fill source (read-only).")
    parser.add_argument("--path", type=Path, default=MARKOUT_JOURNAL_PATH, help=f"markout journal (default {MARKOUT_JOURNAL_PATH})")
    parser.add_argument("--by-exchange", action="store_true", help="also split by exchange id")
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    if not args.path.exists():
        logger.error("Markout journal not found", extra={"path": str(args.path)})
        return 1
    rows = summarize(args.path.read_text(encoding="utf-8").splitlines(), args.by_exchange)
    if not rows:
        logger.info("No markouts in the journal yet", extra={"path": str(args.path)})
    for row in rows:
        logger.info("Markout", extra=row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
