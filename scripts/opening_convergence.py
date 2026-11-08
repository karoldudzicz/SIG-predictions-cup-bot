"""OFFLINE research: did contracts that opened near 0.50 against a decided oracle converge?

EVERYTHING IS A PROXY: hourly candles (OHLC of the trades in each hour), no order book, and
the sums below are price units, not P&L. Zero new reads - only files already on disk:
  - stage-1 price-history reads of tape_patterns (newest data/edge_scan/tape_patterns/*/raw.jsonl
    or --tape-raw): 1h candles per exchange; titles from its market-list pages. Candle
    `time` is the hour's START (ISO text or epoch s / ms are all accepted); "close at T" =
    the close of the last candle that ENDED at or before T (an hour without trades has no
    candle and keeps the previous close);
  - data/market_map.json: the oracle prices seen when the map was built (check_prices,
    average of Polymarket / Kalshi) - the oracle reference closest to the open;
  - the current SIG mid: the newest /exchanges/prices read saved by dry_run_replay
    (data/edge_scan/dry_run_replay/*/raw.jsonl or --mids-raw);
  - the current oracle mid where position_review saved one ("oracle:mids"), shown only.

Per exchange, from the first candle at or after 16:00Z 2026-10-01: its open, high, low,
close, trades and volume; the close at 20:00Z 2026-10-01 and at 00:00Z 2026-10-03; the
current mid and the oracle fair. "Mis-initialised at open" = the first candle's OPEN in
[0.40, 0.60] AND the oracle fair < 0.15 or > 0.85 (the old definition on its CLOSE is
counted next to it). For each mis-initialised exchange: move from the open to the 20:00Z
close, its component toward the oracle and the share of the opening gap it closed (also by
00:00Z 10-03 and by now); the sum of the moves toward the oracle; the list sorted by it.
Diagnostics: exchanges with a candle starting at 16:00 ... 20:00Z, and with a 20:00Z close,
over ALL exchanges. Controls: the 16:00-17:00Z candle of house:FL-22, house:VA-01,
governor:GA, house:IA-02, senate:IA, senate:RI, senate:DE, senate:OK, senate:FL, house:NH-01.
-> data/edge_scan/opening_convergence/<UTC date>/: opening_convergence.csv, controls.csv, summary.md.

    python scripts/opening_convergence.py [--tape-raw PATH] [--mids-raw PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import MARKET_MAP_PATH, MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, read_jsonl, run_dir, write_csv  # noqa: E402

logger = logging.getLogger("opening_convergence")

SCRIPT = "opening_convergence"
OPEN = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
AT_2000 = datetime(2026, 10, 1, 20, 0, tzinfo=timezone.utc)
AT_OCT3 = datetime(2026, 10, 3, 0, 0, tzinfo=timezone.utc)
COVERAGE_HOURS = tuple(OPEN + timedelta(hours=h) for h in range(5))  # 16:00 ... 20:00Z
CONTROL_RACES = ("house:FL-22", "house:VA-01", "governor:GA", "house:IA-02", "senate:IA",
                 "senate:RI", "senate:DE", "senate:OK", "senate:FL", "house:NH-01")
ORACLE_RECORD_PATH = "oracle:mids"
PROXY = "PROXY: hourly candles (trades in each hour), no order book; sums are price units, not P&L"

FIELDS = ("exchange_id", "title", "race", "party", "mis_initialised", "mis_initialised_close", "first_candle", "first_open", "first_high", "first_low",
          "first_close", "first_trades", "first_volume", "close_2000", "close_2000_candle", "close_oct3", "current_mid", "oracle_fair_map",
          "oracle_now_saved", "gap_at_open", "move_to_2000", "toward_2000", "gap_closed_2000", "gap_closed_oct3", "gap_closed_now")
CONTROL_FIELDS = ("race", "party", "exchange_id", "candle", "open", "high", "low", "close", "trades", "volume", "oracle_fair_map",
                  "open_minus_oracle", "close_minus_oracle", "mis_initialised")


def _num(x: Any) -> float | None:
    return None if x is None else float(x)


def _round(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(x, n)


# ---------------------------------------------------------------------- inputs


def candles_from_raw(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """(exchange -> latest saved 1h candles, exchange -> market title)."""
    candles: dict[str, list[dict[str, Any]]] = {}
    titles: dict[str, str] = {}
    for rec in records:
        if "body" not in rec:
            continue
        path, body = str(rec.get("path") or ""), rec["body"] or {}
        if path.endswith("/price-history"):
            candles[path.split("/")[2]] = list(body.get("candles") or [])  # the latest read wins
        elif re.fullmatch(r"/tournaments/[^/]+/markets", path):
            for market in body.get("data") or []:
                for ex in market.get("exchanges") or []:
                    titles[str(ex["id"])] = market.get("title")
    return candles, titles


def mids_from_raw(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float | None], str | None]:
    """YES mids from saved /exchanges/prices reads (a later read wins) and the newest read's time."""
    mids: dict[str, float | None] = {}
    at = None
    for rec in records:
        if rec.get("path") != "/exchanges/prices" or "body" not in rec:
            continue
        for row in (rec["body"] or {}).get("data") or []:
            bid, ask = row.get("bestBid"), row.get("bestAsk")
            mids[str(row["exchangeId"])] = None if bid is None or ask is None else round((float(bid) + float(ask)) / 2, 4)
        at = rec.get("ts") or at
    return mids, at


def saved_oracle_mids(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float | None], str | None]:
    found: dict[str, float | None] = {}
    at = None
    for rec in records:
        if rec.get("path") == ORACLE_RECORD_PATH and "body" in rec:
            found, at = dict((rec["body"] or {}).get("mids") or {}), rec.get("ts")
    return found, at


def map_oracle_fair(raw_map: Mapping[str, Any]) -> dict[str, float | None]:
    """exchange -> average of the oracle prices recorded when the map was built."""
    out = {}
    for eid, entry in (raw_map.get("entries") or {}).items():
        prices = [float(v) for v in (entry.get("check_prices") or {}).values() if v is not None]
        out[str(eid)] = round(sum(prices) / len(prices), 4) if prices else None
    return out


def _newest(script: str) -> Path | None:
    runs = sorted((OUT_ROOT / script).glob("*/raw.jsonl"))
    return runs[-1] if runs else None


# ---------------------------------------------------------------------- analysis


def exchange_row(eid: str, candles: Sequence[Mapping[str, Any]], title: str | None, market_map: MarketMap, oracle_fair: float | None,
                 oracle_now: float | None, current_mid: float | None) -> dict[str, Any]:
    first = stats.first_candle_from(candles, OPEN)
    first_open = _num(first.get("open")) if first else None
    first_close = _num(first["close"]) if first else None
    c2000, c2000_at = stats.close_as_of(candles, AT_2000)
    c_oct3, _ = stats.close_as_of(candles, AT_OCT3)
    # Moves are measured from the OPENING price (the first candle's open).
    to_2000 = stats.move_toward(first_open, c2000, oracle_fair)
    to_oct3 = stats.move_toward(first_open, c_oct3, oracle_fair)
    to_now = stats.move_toward(first_open, current_mid, oracle_fair)
    m = market_map.get(eid)
    return {"exchange_id": eid, "title": title or (m.title if m else None), "race": m.race.key if m else None, "party": m.party if m else None,
            "mis_initialised": stats.mis_initialised(first_open, oracle_fair), "mis_initialised_close": stats.mis_initialised(first_close, oracle_fair),
            "first_candle": None if not first else stats.candle_start(first).isoformat(), "first_open": first_open,
            "first_high": _num(first.get("high")) if first else None, "first_low": _num(first.get("low")) if first else None, "first_close": first_close,
            "first_trades": first.get("tradeCount") if first else None, "first_volume": first.get("volume") if first else None,
            "close_2000": c2000, "close_2000_candle": None if c2000_at is None else c2000_at.isoformat(), "close_oct3": c_oct3,
            "current_mid": current_mid, "oracle_fair_map": oracle_fair, "oracle_now_saved": oracle_now,
            "gap_at_open": _round(to_2000["gap"]), "move_to_2000": _round(to_2000["move"]), "toward_2000": _round(to_2000["toward"]),
            "gap_closed_2000": _round(to_2000["gap_closed"], 3), "gap_closed_oct3": _round(to_oct3["gap_closed"], 3), "gap_closed_now": _round(to_now["gap_closed"], 3)}


def control_rows(candles: Mapping[str, Sequence[Mapping[str, Any]]], market_map: MarketMap, oracle: Mapping[str, float | None],
                 races: Sequence[str] = CONTROL_RACES) -> list[dict[str, Any]]:
    """The 16:00-17:00Z candle of every exchange of the control races (None-filled if no trade)."""
    by_race = market_map.by_race()
    rows = []
    for race in races:
        members = sorted(by_race.get(race, []), key=lambda x: x.party)
        if not members:
            rows.append({"race": race, "party": None, "exchange_id": None, "candle": "race not in the market map"})
            continue
        for m in members:
            eid = m.sig_exchange_id
            hour = next((c for c in candles.get(eid, []) if stats.candle_start(c) == OPEN), None)
            first = stats.first_candle_from(candles.get(eid, []), OPEN)
            fair = oracle.get(eid)
            o = _num(hour.get("open")) if hour else None
            close = _num(hour.get("close")) if hour else None
            rows.append({"race": race, "party": m.party, "exchange_id": eid, "candle": None if not hour else stats.candle_start(hour).isoformat(),
                         "open": o, "high": _num(hour.get("high")) if hour else None, "low": _num(hour.get("low")) if hour else None, "close": close,
                         "trades": hour.get("tradeCount") if hour else 0, "volume": hour.get("volume") if hour else 0, "oracle_fair_map": fair,
                         "open_minus_oracle": None if o is None or fair is None else round(o - fair, 4),
                         "close_minus_oracle": None if close is None or fair is None else round(close - fair, 4),
                         "mis_initialised": stats.mis_initialised(_num(first.get("open")) if first else None, fair)})
    return rows


def coverage(candles: Mapping[str, Sequence[Mapping[str, Any]]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Why a 20:00Z close may be missing: candles per opening hour, and how far back the
    saved histories reach - counted over ALL exchanges, not only the flagged ones."""
    earliest = [min(stats.candle_start(c) for c in cs) for cs in candles.values() if cs]
    return {"exchanges": len(candles), "with_candles": len(earliest),
            "candles_starting_at": stats.hour_coverage(candles, COVERAGE_HOURS),
            "with_first_candle_from_16": sum(1 for r in rows if r["first_candle"] is not None),
            "with_2000_close_all": sum(1 for r in rows if r["close_2000"] is not None),
            "history_starts_after_2000": sum(1 for t in earliest if t >= AT_2000),
            "earliest_candle": None if not earliest else min(earliest).isoformat(), "latest_first_candle": None if not earliest else max(earliest).isoformat()}


def analyse(candles: Mapping[str, Sequence[Mapping[str, Any]]], titles: Mapping[str, str], market_map: MarketMap, oracle_fair: Mapping[str, float | None],
            oracle_now: Mapping[str, float | None], mids: Mapping[str, float | None]) -> dict[str, Any]:
    rows = [exchange_row(eid, cs, titles.get(eid), market_map, oracle_fair.get(eid), oracle_now.get(eid), mids.get(eid)) for eid, cs in sorted(candles.items())]
    flagged = sorted((r for r in rows if r["mis_initialised"]), key=lambda r: -(r["toward_2000"] if r["toward_2000"] is not None else -math.inf))
    toward = [r["toward_2000"] for r in flagged if r["toward_2000"] is not None]
    summary = {
        "label": PROXY,
        "exchanges_with_history": len(rows),
        "with_oracle_fair": sum(1 for r in rows if r["oracle_fair_map"] is not None),
        "mis_initialised": len(flagged),
        "mis_initialised_close_based": sum(1 for r in rows if r["mis_initialised_close"]),
        "flagged_with_2000_close": len(toward),
        "sum_toward_oracle_2000": round(math.fsum(toward), 4),
        "sum_positive_toward_oracle_2000": round(math.fsum(t for t in toward if t > 0), 4),
        "moved_toward": sum(1 for t in toward if t > 0), "moved_away": sum(1 for t in toward if t < 0), "unchanged": sum(1 for t in toward if t == 0),
        "median_gap_closed_2000": stats.median_or_none(r["gap_closed_2000"] for r in flagged),
        "median_gap_closed_oct3": stats.median_or_none(r["gap_closed_oct3"] for r in flagged),
        "median_gap_closed_now": stats.median_or_none(r["gap_closed_now"] for r in flagged),
        "coverage": coverage(candles, rows),
    }
    return {"rows": rows, "flagged": flagged, "controls": control_rows(candles, market_map, oracle_fair), "summary": summary}


def summary_markdown(result: Mapping[str, Any], sources: Mapping[str, Any]) -> str:
    s = result["summary"]
    cov = s["coverage"]
    lines = ["# Opening convergence (PROXY)", "", f"_{s['label']}._", "",
             f"- Sources: candles {sources['tape_raw']}; current mid {sources['mids_raw']} (read at {sources['mids_at']}); "
             f"oracle fair = market_map.json check_prices (map built {sources['map_built']}){'; current oracle mids saved ' + str(sources['oracle_now_at']) if sources.get('oracle_now_at') else ''}.",
             f"- Coverage over ALL {cov['exchanges']} exchanges: candles starting at " + ", ".join(f"{h[11:16]}Z {n}" for h, n in cov["candles_starting_at"].items())
             + f"; first candle from 16:00Z: {cov['with_first_candle_from_16']}; close known at 20:00Z: {cov['with_2000_close_all']}; "
             f"histories starting at or after 20:00Z: {cov['history_starts_after_2000']} (earliest candle {cov['earliest_candle']}).",
             f"- Mis-initialised at open (first candle OPEN 0.40-0.60, oracle < 0.15 or > 0.85): **{s['mis_initialised']}** "
             f"(old definition on the first CLOSE: {s['mis_initialised_close_based']}).",
             f"- Of those, {s['flagged_with_2000_close']} have a 20:00Z close: {s['moved_toward']} moved toward the oracle, {s['moved_away']} away, "
             f"{s['unchanged']} unchanged; sum of moves toward the oracle {s['sum_toward_oracle_2000']} (positive parts {s['sum_positive_toward_oracle_2000']}) - price units, not P&L.",
             f"- Median share of the opening gap closed: {s['median_gap_closed_2000']} by 20:00Z, {s['median_gap_closed_oct3']} by 10-03 00:00Z, {s['median_gap_closed_now']} by now.",
             "", "| exchange | race | party | open | high / low | first close | trades | 20:00Z | oracle | toward | gap closed 20:00Z / 10-03 / now |",
             "|---|---|---|---|---|---|---|---|---|---|---|",
             *[f"| {r['exchange_id']} | {r['race']} | {r['party']} | {r['first_open']} | {r['first_high']} / {r['first_low']} | {r['first_close']} | {r['first_trades']} | "
               f"{r['close_2000']} | {r['oracle_fair_map']} | {r['toward_2000']} | {r['gap_closed_2000']} / {r['gap_closed_oct3']} / {r['gap_closed_now']} |" for r in result["flagged"]],
             "", "## Controls: 16:00-17:00Z candle", "", "| race | party | open | high | low | close | trades | volume | oracle | open - oracle |",
             "|---|---|---|---|---|---|---|---|---|---|",
             *[f"| {c['race']} | {c['party']} | {c.get('open')} | {c.get('high')} | {c.get('low')} | {c.get('close')} | {c.get('trades')} | {c.get('volume')} | "
               f"{c.get('oracle_fair_map')} | {c.get('open_minus_oracle')} |" for c in result["controls"]], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: did contracts mis-initialised at the open converge to the oracle? (PROXY, 0 reads)")
    parser.add_argument("--tape-raw", type=Path, default=None, help="tape_patterns raw.jsonl with the stage-1 price-history reads (default: newest)")
    parser.add_argument("--mids-raw", type=Path, default=None, help="raw.jsonl with a saved /exchanges/prices read (default: newest dry_run_replay)")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    tape_raw = args.tape_raw or _newest("tape_patterns")
    if tape_raw is None or not tape_raw.exists():
        logger.error("No tape_patterns raw.jsonl with price-history reads: run tape_patterns.py --yes first, or pass --tape-raw")
        return 2
    candles, titles = candles_from_raw(read_jsonl(tape_raw))
    if not candles:
        logger.error("The tape_patterns raw.jsonl has no price-history reads", extra={"raw": str(tape_raw)})
        return 2
    mids_raw = args.mids_raw or _newest("dry_run_replay")
    mids, mids_at = mids_from_raw(read_jsonl(mids_raw)) if mids_raw and mids_raw.exists() else ({}, None)
    review_raw = _newest("position_review")
    oracle_now, oracle_now_at = saved_oracle_mids(read_jsonl(review_raw)) if review_raw else ({}, None)
    raw_map = json.loads(MARKET_MAP_PATH.read_text(encoding="utf-8"))
    result = analyse(candles, titles, MarketMap.load(), map_oracle_fair(raw_map), oracle_now, mids)
    sources = {"tape_raw": str(tape_raw), "mids_raw": str(mids_raw) if mids_raw else None, "mids_at": mids_at, "map_built": raw_map.get("generated_at"),
               "oracle_now_at": oracle_now_at}

    out = run_dir(SCRIPT)
    write_csv(out / "opening_convergence.csv", result["rows"], FIELDS)
    write_csv(out / "controls.csv", result["controls"], CONTROL_FIELDS)
    (out / "summary.md").write_text(summary_markdown(result, sources), encoding="utf-8")
    logger.info("Candle coverage over all exchanges (PROXY)", extra=result["summary"]["coverage"])
    for r in result["flagged"]:
        logger.info("Mis-initialised at open (PROXY)", extra={k: r[k] for k in FIELDS})
    logger.info("Opening convergence done (PROXY, 0 reads)", extra={"out_dir": str(out), **{k: v for k, v in result["summary"].items() if k != "coverage"}, **sources})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
