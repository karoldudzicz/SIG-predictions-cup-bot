"""READ-ONLY research: the opening tape (2026-10-01 16:00-18:00Z) in detail, from data
already on disk, plus where our own opening orders filled against our oracle fair.

Inputs (no re-fetch):
  - the newest data/edge_scan/leaderboard_analysis/*/raw.jsonl (or --raw): its
    /exchanges/{id}/trades responses for the opening window (10 markets) and its
    /tournaments/{slug} response (tournament id);
  - data/order_journal.jsonl: our order placements;
  - data/markout.jsonl, if it has fills in the window (it started on 2026-10-03, so
    normally none).
One optional API read: GET /exchanges/prices for the tape markets (one bulk call) for
the current SIG mid. Without --yes it is skipped and the mid columns stay empty.

1. Per market, 10-minute buckets 16:00-18:00Z: trades, shares, VWAP, share of trades
   priced < 0.10 or > 0.90 (YES price) -> buckets.csv.
2. The 20 largest prints by size: time, market, size, YES price, size * YES price and the
   API's own `volume` (size * the executed side's price) -> largest_prints.csv.
3. Per market: VWAP of the first and last non-empty bucket (which bucket is reported) vs
   the current SIG mid -> drift.csv.
4. Our orders journaled in the window: kind, side, limit, fill (the engine's response at
   placement) and the oracle fair. The journal keeps the canonical order WITHOUT
   totalCost / fillPrice, so the entry compared is normally the canonical limit (a taker
   fill is at the limit or better); totalCost / quantityTraded or fillPrice are used only
   if a journal line has them (entry_basis says which). And the oracle fair
   at the time. The code running then (0ec8ce1) logged "edge X" = fair_side - order
   price with no margin, so fair_side = limit + edge exactly when the order was not
   resized to the book; for a "fitted to book" order the sizing price was not stored and
   fair_side lies in [limit + MIN_EDGE, limit + edge]. Fills of a resting remainder after
   placement are not in the journal (they need REST fill reads) -> our_fills.csv.

-> data/edge_scan/opening_analysis/<UTC date>/: buckets.csv, largest_prints.csv,
   drift.csv, our_fills.csv, summary.json, summary.md (+ raw.jsonl for the price read).

    python scripts/opening_analysis.py [--raw PATH] [--yes]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import DATA_DIR, MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min, run_dir, write_csv, write_json  # noqa: E402

logger = logging.getLogger("opening_analysis")

SCRIPT = "opening_analysis"
WINDOW_START = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)
BUCKET_MIN = 10
BUCKETS = 12
EXTREME_LOW, EXTREME_HIGH = 0.10, 0.90
LARGEST_N = 20
OPENING_MIN_EDGE = 0.02  # RiskLimits.min_edge in the code running at the opening (0ec8ce1)
ORDER_JOURNAL_PATH = DATA_DIR / "order_journal.jsonl"
MARKOUT_PATH = DATA_DIR / "markout.jsonl"
_EDGE_RE = re.compile(r"edge (-?\d+(?:\.\d+)?)")

BUCKET_FIELDS = ("exchange_id", "race", "party", "bucket", "bucket_start", "trades", "shares", "vwap", "extreme_share")
PRINT_FIELDS = ("created_at", "exchange_id", "race", "party", "size", "price_yes", "side_space", "size_x_price_yes", "api_volume")
DRIFT_FIELDS = ("exchange_id", "race", "party", "trades", "first_bucket", "first_vwap", "last_bucket", "last_vwap", "last_minus_first", "current_mid", "mid_minus_last", "mid_minus_first")
FILL_FIELDS = ("ts", "exchange_id", "race", "party", "kind", "status", "side", "action", "limit", "quantity", "quantity_traded", "fill_price", "avg_cost_price", "entry_price", "entry_basis",
               "edge_logged", "fitted", "fair_side", "fair_side_low", "fair_side_high", "fair_yes", "fill_vs_fair_side", "markout_fair_yes")


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _label(market_map: MarketMap, eid: str) -> dict[str, Any]:
    m = market_map.get(eid)
    return {"race": m.race.key if m else None, "party": m.party if m else None}


# ---------------------------------------------------------------------- tape from raw.jsonl


def tape_from_raw(records: Sequence[Mapping[str, Any]], start: datetime = WINDOW_START, end: datetime = WINDOW_END) -> dict[str, list[dict[str, Any]]]:
    """exchange_id -> trades inside [start, end), de-duplicated by trade id, time order."""
    tapes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        path = str(rec.get("path") or "")
        if "body" not in rec or not path.startswith("/exchanges/") or not path.endswith("/trades"):
            continue
        eid = path.split("/")[2]
        for trade in (rec["body"] or {}).get("data") or []:
            t = _parse_ts(trade.get("createdAt"))
            if t is not None and start <= t < end:
                tapes[eid][str(trade.get("id"))] = trade
    return {eid: sorted(trades.values(), key=lambda tr: (str(tr.get("createdAt")), str(tr.get("id")))) for eid, trades in tapes.items()}


def tournament_id_from_raw(records: Sequence[Mapping[str, Any]]) -> str | None:
    for rec in records:
        if "body" in rec and re.fullmatch(r"/tournaments/[^/]+", str(rec.get("path") or "")):
            return (rec["body"] or {}).get("id")
    return None


# ---------------------------------------------------------------------- 1-3: buckets, prints, drift


def bucket_rows(eid: str, trades: Sequence[Mapping[str, Any]], market_map: MarketMap) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in trades:
        idx = stats.bucket_index(_parse_ts(trade.get("createdAt")), WINDOW_START, BUCKET_MIN, BUCKETS)
        if idx is not None:
            grouped[idx].append(trade)
    rows = []
    for idx in range(BUCKETS):
        group = grouped.get(idx, [])
        priced = [t for t in group if t.get("price") is not None]
        rows.append({"exchange_id": eid, **_label(market_map, eid), "bucket": idx,
                     "bucket_start": (WINDOW_START + timedelta(minutes=BUCKET_MIN * idx)).strftime("%H:%M"),
                     "trades": len(group), "shares": math.fsum(float(t.get("size") or 0) for t in group),
                     "vwap": stats.vwap((float(t["price"]), float(t.get("size") or 0)) for t in priced),
                     "extreme_share": stats.share_outside((float(t["price"]) for t in priced), EXTREME_LOW, EXTREME_HIGH)})
    return rows


def largest_prints(tapes: Mapping[str, Sequence[Mapping[str, Any]]], market_map: MarketMap, n: int = LARGEST_N) -> list[dict[str, Any]]:
    rows = []
    for eid, trades in tapes.items():
        for t in trades:
            price = None if t.get("price") is None else float(t["price"])
            size = float(t.get("size") or 0)
            rows.append({"created_at": t.get("createdAt"), "exchange_id": eid, **_label(market_map, eid), "size": size, "price_yes": price,
                         "side_space": t.get("side"), "size_x_price_yes": None if price is None else round(size * price, 2), "api_volume": t.get("volume")})
    rows.sort(key=lambda r: (-r["size"], str(r["created_at"])))
    return rows[:n]


def drift_row(eid: str, buckets: Sequence[Mapping[str, Any]], current_mid: float | None, market_map: MarketMap) -> dict[str, Any]:
    """First / last NON-EMPTY bucket VWAP vs the current SIG mid."""
    filled = [b for b in buckets if b["vwap"] is not None]
    first, last = (filled[0], filled[-1]) if filled else (None, None)

    def diff(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else round(a - b, 4)

    fv, lv = (first or {}).get("vwap"), (last or {}).get("vwap")
    return {"exchange_id": eid, **_label(market_map, eid), "trades": sum(b["trades"] for b in buckets),
            "first_bucket": (first or {}).get("bucket_start"), "first_vwap": fv, "last_bucket": (last or {}).get("bucket_start"), "last_vwap": lv,
            "last_minus_first": diff(lv, fv), "current_mid": current_mid, "mid_minus_last": diff(current_mid, lv), "mid_minus_first": diff(current_mid, fv)}


# ---------------------------------------------------------------------- 4: our opening orders


def parse_edge(reason: str | None) -> float | None:
    match = _EDGE_RE.search(reason or "")
    return float(match.group(1)) if match else None


def fair_from_journal(limit: float | None, reason: str | None, min_edge: float = OPENING_MIN_EDGE) -> dict[str, Any]:
    """fair_side from the opening code's "edge X" (= fair_side - sizing price, no margin).
    Not fitted: exact. "fitted to book": the sizing price was not stored -> a range."""
    edge = parse_edge(reason)
    fitted = "fitted to book" in (reason or "")
    if edge is None or limit is None:
        return {"edge_logged": edge, "fitted": fitted, "fair_side": None, "fair_side_low": None, "fair_side_high": None}
    if not fitted:
        exact = round(limit + edge, 4)
        return {"edge_logged": edge, "fitted": False, "fair_side": exact, "fair_side_low": exact, "fair_side_high": exact}
    return {"edge_logged": edge, "fitted": True, "fair_side": None, "fair_side_low": round(limit + min_edge, 4), "fair_side_high": round(limit + edge, 4)}


def journal_orders(entries: Sequence[Mapping[str, Any]], start: datetime = WINDOW_START, end: datetime = WINDOW_END) -> list[dict[str, Any]]:
    """Latest state per idempotency key, for keys first journaled inside [start, end)."""
    first_ts: dict[str, datetime] = {}
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = entry.get("key")
        t = _parse_ts(entry.get("ts"))
        if key is None or t is None:
            continue
        first_ts.setdefault(key, t)
        latest[key] = {**latest.get(key, {}), **entry}
    return [{**latest[k], "_first_ts": first_ts[k]} for k in latest if start <= first_ts[k] < end]


def our_fill_rows(orders: Sequence[Mapping[str, Any]], markout_fills: Mapping[str, float], market_map: MarketMap) -> list[dict[str, Any]]:
    rows = []
    for order in sorted(orders, key=lambda o: o["_first_ts"]):
        payload = order.get("payload") or {}
        result = order.get("result") if isinstance(order.get("result"), dict) else {}
        limit = None if payload.get("price") is None else float(payload["price"])
        side = payload.get("side")
        traded = abs(float(result.get("quantityTraded") or 0))
        total_cost = result.get("totalCost")
        fair = fair_from_journal(limit, order.get("reason")) if order.get("kind") == "increase" else {"edge_logged": None, "fitted": None, "fair_side": None, "fair_side_low": None, "fair_side_high": None}
        fill_price = None if result.get("fillPrice") is None else float(result["fillPrice"])
        avg_cost = round(float(total_cost) / traded, 4) if traded and total_cost is not None else None
        # The journal records the canonical order without totalCost / fillPrice, so normally
        # only the canonical limit is known: a taker fill is at that limit or better. It is in
        # the canonical side's terms; converted when the engine flipped the side.
        canonical_limit = None
        if traded and result.get("price") is not None:
            canonical_limit = float(result["price"]) if result.get("side") in (None, side) else round(1.0 - float(result["price"]), 4)
        # totalCost / shares is per share of the side bought; fillPrice's denomination is unverified.
        entry, basis = next(((v, b) for v, b in ((avg_cost, "avg_cost"), (fill_price, "fill_price"), (canonical_limit, "limit (fill at or better)")) if v is not None),
                            (None, None))
        fair_side = fair["fair_side"]
        eid = str(payload.get("exchangeId"))
        rows.append({
            "ts": order["_first_ts"].isoformat(), "exchange_id": eid, **_label(market_map, eid), "kind": order.get("kind"), "status": order.get("status"),
            "side": side, "action": payload.get("action"), "limit": limit, "quantity": payload.get("quantity"), "quantity_traded": traded,
            "fill_price": fill_price, "avg_cost_price": avg_cost, "entry_price": entry, "entry_basis": basis,
            **fair,
            "fair_yes": None if fair_side is None else round(fair_side if side == "yes" else 1.0 - fair_side, 4),
            "fill_vs_fair_side": None if fair_side is None or entry is None else round(fair_side - entry, 4),
            "markout_fair_yes": markout_fills.get(str(result.get("orderId"))),
        })
    return rows


def markout_fairs(entries: Sequence[Mapping[str, Any]], start: datetime = WINDOW_START, end: datetime = WINDOW_END) -> dict[str, float]:
    """order id -> fair at fill from markout "fill" lines inside the window (normally none:
    the markout journal started on 2026-10-03)."""
    out = {}
    for e in entries:
        if e.get("event") != "fill" or e.get("order_id") is None or e.get("fair_at_fill") is None or e.get("filled_at") is None:
            continue
        t = datetime.fromtimestamp(float(e["filled_at"]), tz=timezone.utc)  # markout stores unix seconds
        if start <= t < end:
            out[str(e["order_id"])] = float(e["fair_at_fill"])
    return out


# ---------------------------------------------------------------------- run


def newest_raw(root: Path | None = None) -> Path | None:
    candidates = sorted((root or OUT_ROOT / "leaderboard_analysis").glob("*/raw.jsonl"))
    return candidates[-1] if candidates else None


def analyse(tapes: Mapping[str, Sequence[Mapping[str, Any]]], mids: Mapping[str, float | None], orders: Sequence[Mapping[str, Any]],
            markout: Mapping[str, float], market_map: MarketMap) -> dict[str, Any]:
    buckets = {eid: bucket_rows(eid, trades, market_map) for eid, trades in sorted(tapes.items())}
    drift = [drift_row(eid, rows, mids.get(eid), market_map) for eid, rows in buckets.items()]
    fills = our_fill_rows(orders, markout, market_map)
    traded = [f for f in fills if f["quantity_traded"]]
    pooled_buckets = []
    for idx in range(BUCKETS):
        group = [rows[idx] for rows in buckets.values()]
        trades = sum(r["trades"] for r in group)
        extreme = math.fsum((r["extreme_share"] or 0) * r["trades"] for r in group)
        pooled_buckets.append({"bucket_start": group[0]["bucket_start"] if group else None, "trades": trades,
                               "extreme_share": round(extreme / trades, 4) if trades else None})
    summary = {
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "markets": len(tapes), "trades": sum(len(t) for t in tapes.values()),
        "pooled_by_bucket": pooled_buckets,
        "drift": {"median_last_minus_first": stats.median_or_none(d["last_minus_first"] for d in drift),
                  "median_abs_mid_minus_last": stats.median_or_none(None if d["mid_minus_last"] is None else abs(d["mid_minus_last"]) for d in drift),
                  "markets_with_mid": sum(1 for d in drift if d["current_mid"] is not None)},
        "our_orders": {"journaled": len(fills), "by_status": dict(sorted(_count(f["status"] for f in fills).items())), "with_fills": len(traded),
                       "fair_exact": sum(1 for f in traded if f["fair_side"] is not None), "fair_range_only": sum(1 for f in traded if f["fair_side"] is None and f["fair_side_low"] is not None),
                       "median_fill_vs_fair_side": stats.median_or_none(f["fill_vs_fair_side"] for f in traded),
                       "note": "fills at placement only (engine response); later fills of a resting remainder are not journaled"},
    }
    return {"buckets": [r for rows in buckets.values() for r in rows], "prints": largest_prints(tapes, market_map), "drift": drift, "fills": fills, "summary": summary}


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def summary_markdown(result: Mapping[str, Any]) -> str:
    s = result["summary"]
    lines = [f"# Opening tape {s['window'][0][:16]} - {s['window'][1][11:16]} UTC", "",
             f"- {s['markets']} markets, {s['trades']} trades.",
             f"- Median VWAP move first -> last non-empty 10-min bucket: {s['drift']['median_last_minus_first']}.",
             f"- Median |current mid - last-bucket VWAP|: {s['drift']['median_abs_mid_minus_last']} ({s['drift']['markets_with_mid']} markets with a mid).",
             "- Share of trades priced < 0.10 or > 0.90, per bucket: " + ", ".join(f"{b['bucket_start']} {b['extreme_share']}" for b in s["pooled_by_bucket"]) + ".",
             f"- Largest print: {result['prints'][0]['size']:g} shares @ {result['prints'][0]['price_yes']} on {result['prints'][0]['exchange_id']}." if result["prints"] else "- No prints.",
             f"- Our orders journaled in the window: {s['our_orders']['journaled']} ({s['our_orders']['by_status']}); with fills: {s['our_orders']['with_fills']}; "
             f"fair exact for {s['our_orders']['fair_exact']}, range only for {s['our_orders']['fair_range_only']}; median fair_side - fill price: {s['our_orders']['median_fill_vs_fair_side']}.",
             f"- {s['our_orders']['note']}.", ""]
    return "\n".join(lines)


async def _read_mids(ids: Sequence[str], tournament_id: str | None, reads_per_min: float) -> dict[str, float | None]:
    async with research_reader(SCRIPT, reads_per_min) as reader:
        body = await reader.get("/exchanges/prices", {"ids": ",".join(ids), "tournamentId": tournament_id})
    mids: dict[str, float | None] = {}
    for row in body.get("data") or []:
        bid, ask = row.get("bestBid"), row.get("bestAsk")
        mids[str(row["exchangeId"])] = None if bid is None or ask is None else round((float(bid) + float(ask)) / 2.0, 4)
    return mids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: opening tape buckets, largest prints, drift and our opening fills.")
    add_research_args(parser)
    parser.add_argument("--raw", type=Path, default=None, help="leaderboard_analysis raw.jsonl (default: the newest)")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    raw = args.raw or newest_raw()
    if raw is None or not raw.exists():
        logger.error("No leaderboard_analysis raw.jsonl found: pass --raw", extra={"raw": str(raw)})
        return 2
    records = read_jsonl(raw)
    tapes = tape_from_raw(records)
    if not tapes:
        logger.error("No opening-window trades in the raw file", extra={"raw": str(raw)})
        return 2
    market_map = MarketMap.load()
    mids: dict[str, float | None] = {}
    if confirm_fetch(ReadEstimate(1, reads_per_min), args.yes):
        mids = asyncio.run(_read_mids(sorted(tapes), tournament_id_from_raw(records), reads_per_min))
    else:
        logger.info("Continuing offline: current mid columns stay empty")
    orders = journal_orders(read_jsonl(ORDER_JOURNAL_PATH))
    result = analyse(tapes, mids, orders, markout_fairs(read_jsonl(MARKOUT_PATH)), market_map)

    out = run_dir(SCRIPT)
    write_csv(out / "buckets.csv", result["buckets"], BUCKET_FIELDS)
    write_csv(out / "largest_prints.csv", result["prints"], PRINT_FIELDS)
    write_csv(out / "drift.csv", result["drift"], DRIFT_FIELDS)
    write_csv(out / "our_fills.csv", result["fills"], FILL_FIELDS)
    write_json(out / "summary.json", {"raw": str(raw), **result["summary"]})
    (out / "summary.md").write_text(summary_markdown(result), encoding="utf-8")
    for row in result["drift"]:
        logger.info("Opening drift", extra=row)
    logger.info("Opening analysis done", extra={"out_dir": str(out), **{k: result["summary"][k] for k in ("markets", "trades", "drift", "our_orders")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
