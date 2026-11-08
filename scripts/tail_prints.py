"""OFFLINE research: "tail prints" - trades far from where the market was trading - per
exchange and UTC day, and whether they survived the opening.

EVERYTHING IS A PROXY: tape prints only - no order book, no queue position, no aggressor -
and the "earnings" are an UPPER BOUND, not P&L. Zero new reads: the trade history saved by
tape_patterns (newest data/edge_scan/tape_patterns/*/raw.jsonl or --raw; up to 40
exchanges, de-duplicated by trade id).

A trade is a TAIL PRINT when its price differs by >= 0.25 from the median price of the
OTHER trades within +-5 minutes (a centred 10-minute window, which may cross midnight); a
trade with fewer than 3 such neighbours is not judged. Per exchange and UTC day: trades,
judged, tail count, tail shares, tail share of the day's trades, tails by price level
(<0.05, 0.05-0.15, 0.15-0.85, 0.85-0.95, >0.95), and the passive-order proxy = sum of
shares * |price - median| over the tails (what a resting order at that price would have
earned IF the print had hit it), plus the same without the first 30 minutes after 16:00Z
on 2026-10-01.

Coverage: tape_patterns pages each history newest-first with a page cap, so a busy
exchange may not reach back to 10-01. An exchange counts for a day only if its saved
history covers that whole day (it reached the start of history, or its oldest saved trade
is before the day began); the daily totals say how many exchanges that is.
-> data/edge_scan/tail_prints/<UTC date>/: tail_prints_by_exchange_day.csv, tail_prints_daily.csv,
   summary.md.

    python scripts/tail_prints.py [--raw PATH]
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, read_jsonl, run_dir, write_csv  # noqa: E402

logger = logging.getLogger("tail_prints")

SCRIPT = "tail_prints"
HALF_WINDOW_S = 300.0  # +-5 min = the surrounding 10 minutes
THRESHOLD = 0.25
MIN_NEIGHBOURS = 3
OPEN = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
OPEN_EXCLUDED = (OPEN, OPEN + timedelta(minutes=30))
OPENING_DAY = "2026-10-01"
LEVELS = tuple(label for _, label in stats.TAIL_LEVELS)
PROXY = "PROXY: tape prints, no order book, no queue position, no aggressor - the passive-order earnings are an upper bound, not P&L"

EXCHANGE_DAY_FIELDS = ("exchange_id", "title", "race", "party", "day", "covered", "trades", "judged", "tails", "tail_shares", "tail_share_of_trades",
                       *(f"tails_{label}" for label in LEVELS), "proxy_earn", "proxy_earn_excl_open30")
DAILY_FIELDS = ("day", "exchanges_covered", "exchanges_partial", "trades", "tails", "tail_share_of_trades", "tail_shares",
                *(f"tails_{label}" for label in LEVELS), "proxy_earn", "proxy_earn_excl_open30", "proxy_earn_per_1000_trades")


def _ts(raw: Any) -> datetime:
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------- inputs


def tapes_from_raw(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], set[str], dict[str, str]]:
    """(exchange -> trades in time order, exchanges whose saved history reached the start,
    exchange -> title). A history is complete when a page without `from` had hasMore false."""
    trades: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    complete: set[str] = set()
    titles: dict[str, str] = {}
    for rec in records:
        if "body" not in rec:
            continue
        path, body = str(rec.get("path") or ""), rec["body"] or {}
        if re.fullmatch(r"/tournaments/[^/]+/markets", path):
            for market in body.get("data") or []:
                for ex in market.get("exchanges") or []:
                    titles[str(ex["id"])] = market.get("title")
            continue
        if not (path.startswith("/exchanges/") and path.endswith("/trades")):
            continue
        eid = path.split("/")[2]
        if not (body.get("pagination") or {}).get("hasMore") and not (rec.get("params") or {}).get("from"):
            complete.add(eid)
        for t in body.get("data") or []:
            if t.get("createdAt") and t.get("price") is not None:
                trades[eid][str(t.get("id"))] = t
    tapes = {eid: sorted(ts.values(), key=lambda t: (str(t["createdAt"]), str(t.get("id")))) for eid, ts in trades.items()}
    return tapes, complete, titles


def covers_day(day: str, oldest: datetime | None, complete: bool) -> bool:
    """A saved history covers a whole UTC day if it reached the start of the history or its
    oldest saved trade is before that day began."""
    if complete:
        return True
    return oldest is not None and oldest < datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------- analysis


def exchange_day_rows(eid: str, trades: Sequence[Mapping[str, Any]], complete: bool, market_map: MarketMap, title: str | None) -> list[dict[str, Any]]:
    times = [_ts(t["createdAt"]) for t in trades]
    prices = [float(t["price"]) for t in trades]
    sizes = [abs(float(t.get("size") or 0)) for t in trades]
    flags = stats.tail_prints([t.timestamp() for t in times], prices, HALF_WINDOW_S, THRESHOLD, MIN_NEIGHBOURS)
    oldest = times[0] if times else None
    m = market_map.get(eid)
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(times):
        by_day[t.strftime("%Y-%m-%d")].append(i)
    rows = []
    for day, idx in sorted(by_day.items()):
        tails = [i for i in idx if flags[i] is not None and flags[i]["tail"]]
        earn = [(i, sizes[i] * abs(flags[i]["deviation"])) for i in tails]
        levels = Counter(stats.tail_level(prices[i]) for i in tails)
        rows.append({"exchange_id": eid, "title": title or (m.title if m else None), "race": m.race.key if m else None, "party": m.party if m else None,
                     "day": day, "covered": covers_day(day, oldest, complete), "trades": len(idx), "judged": sum(1 for i in idx if flags[i] is not None),
                     "tails": len(tails), "tail_shares": math.fsum(sizes[i] for i in tails),
                     "tail_share_of_trades": round(len(tails) / len(idx), 4) if idx else None,
                     **{f"tails_{label}": levels.get(label, 0) for label in LEVELS},
                     "proxy_earn": round(math.fsum(e for _, e in earn), 2),
                     "proxy_earn_excl_open30": round(math.fsum(e for i, e in earn if not OPEN_EXCLUDED[0] <= times[i] < OPEN_EXCLUDED[1]), 2)})
    return rows


def daily_totals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per UTC day, summed over the exchanges whose history covers that whole day."""
    out = []
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    for day, group in sorted(by_day.items()):
        covered = [r for r in group if r["covered"]]
        trades = sum(r["trades"] for r in covered)
        tails = sum(r["tails"] for r in covered)
        earn = math.fsum(r["proxy_earn"] for r in covered)
        out.append({"day": day, "exchanges_covered": len(covered), "exchanges_partial": len(group) - len(covered), "trades": trades, "tails": tails,
                    "tail_share_of_trades": round(tails / trades, 4) if trades else None, "tail_shares": math.fsum(r["tail_shares"] for r in covered),
                    **{f"tails_{label}": sum(r[f"tails_{label}"] for r in covered) for label in LEVELS},
                    "proxy_earn": round(earn, 2), "proxy_earn_excl_open30": round(math.fsum(r["proxy_earn_excl_open30"] for r in covered), 2),
                    "proxy_earn_per_1000_trades": round(1000 * earn / trades, 2) if trades else None})
    return out


def opening_vs_later(daily: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The opening day against each later day (and the later days' median)."""
    opening = next((d for d in daily if d["day"] == OPENING_DAY), None)
    later = [d for d in daily if d["day"] > OPENING_DAY]
    return {"opening": opening, "later_days": len(later),
            "later_median_proxy_earn": stats.median_or_none(d["proxy_earn"] for d in later),
            "later_median_tail_share": stats.median_or_none(d["tail_share_of_trades"] for d in later),
            "later_median_per_1000_trades": stats.median_or_none(d["proxy_earn_per_1000_trades"] for d in later)}


def summary_markdown(daily: Sequence[Mapping[str, Any]], comparison: Mapping[str, Any], exchanges: int, raw: str) -> str:
    o = comparison["opening"] or {}
    lines = ["# Tail prints (PROXY)", "", f"_{PROXY}._", "",
             f"- Source: {raw}; {exchanges} exchanges. Tail = price >= {THRESHOLD} from the median of the other trades within +-{HALF_WINDOW_S / 60:g} min "
             f"(at least {MIN_NEIGHBOURS} of them). Each day counts only the exchanges whose saved history covers the whole day.",
             f"- {OPENING_DAY}: {o.get('tails')} tails in {o.get('trades')} trades ({o.get('tail_share_of_trades')}), proxy {o.get('proxy_earn')} "
             f"({o.get('proxy_earn_excl_open30')} without 16:00-16:30Z), on {o.get('exchanges_covered')} covered exchanges.",
             f"- Later days ({comparison['later_days']}): median proxy {comparison['later_median_proxy_earn']}, median tail share "
             f"{comparison['later_median_tail_share']}, median proxy per 1,000 trades {comparison['later_median_per_1000_trades']}.",
             "", "| day | exchanges covered (partial) | trades | tails | tail share | <0.05 | 0.05-0.15 | 0.15-0.85 | 0.85-0.95 | >0.95 | proxy | proxy excl. 16:00-16:30Z | per 1,000 trades |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
             *[f"| {d['day']} | {d['exchanges_covered']} ({d['exchanges_partial']}) | {d['trades']} | {d['tails']} | {d['tail_share_of_trades']} | "
               + " | ".join(str(d[f'tails_{label}']) for label in LEVELS)
               + f" | {d['proxy_earn']} | {d['proxy_earn_excl_open30']} | {d['proxy_earn_per_1000_trades']} |" for d in daily], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: tail prints per exchange and UTC day (PROXY, 0 reads).")
    parser.add_argument("--raw", type=Path, default=None, help="tape_patterns raw.jsonl (default: the newest)")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    raw = args.raw or (sorted((OUT_ROOT / "tape_patterns").glob("*/raw.jsonl")) or [None])[-1]
    if raw is None or not Path(raw).exists():
        logger.error("No tape_patterns raw.jsonl: run tape_patterns.py --yes first, or pass --raw")
        return 2
    tapes, complete, titles = tapes_from_raw(read_jsonl(Path(raw)))
    if not tapes:
        logger.error("No saved trades in the raw file", extra={"raw": str(raw)})
        return 2
    market_map = MarketMap.load()
    rows = [r for eid, trades in sorted(tapes.items()) for r in exchange_day_rows(eid, trades, eid in complete, market_map, titles.get(eid))]
    daily = daily_totals(rows)
    comparison = opening_vs_later(daily)

    out = run_dir(SCRIPT)
    write_csv(out / "tail_prints_by_exchange_day.csv", rows, EXCHANGE_DAY_FIELDS)
    write_csv(out / "tail_prints_daily.csv", daily, DAILY_FIELDS)
    (out / "summary.md").write_text(summary_markdown(daily, comparison, len(tapes), str(raw)), encoding="utf-8")
    for d in daily:
        logger.info("Tail prints per day (PROXY)", extra=d)
    logger.info("Tail prints done (PROXY, 0 reads)", extra={"out_dir": str(out), "exchanges": len(tapes), "complete_histories": len(complete & set(tapes)),
                                                          "later_days": comparison["later_days"], "label": PROXY})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
