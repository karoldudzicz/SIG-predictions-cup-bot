"""OFFLINE: what is SIG's tournament valuation price (Position.currentPrice) - last trade, mid
or something else? Makes NO API request: it only reads a raw.jsonl written earlier by a
research script (default: the newest data/edge_scan/leaderboard_analysis/*/raw.jsonl).

Spec (openapi.json): Position.currentPrice = "Global price for unbound keys or tournament
valuation price for organization-bound keys"; /tournaments/{slug}/portfolio/positions
"Prices use tournament valuation_price" (409 instead of a neutral default when it is
missing). marketValue is computed from it. The spec does not say how it is computed
during the tournament; at end-settlement the DMM's proposed close-out price is a 5h VWAP.

Pairing: the last /portfolio/positions response in the file, with the /exchanges/prices
responses read right after it (leaderboard_analysis reads them back to back); a held
market missing there falls back to the latest earlier prices response (its age is
reported). Per position (largest |marketValue| first, --limit of them):
  - denomination: whether currentPrice is YES-normalised or in the held side's terms,
    from marketValue / |quantity| (only NO positions can tell);
  - the YES-normalised mark vs bestBid, bestAsk, mid and latestPrice (last trade) of the
    same moment: which it equals (within 1e-4) and where it sits vs the spread.
A rolling VWAP cannot be tested from these responses (it needs the tape) - if neither
last nor mid explains the marks, that is the next hypothesis.

-> valuation_price.csv and valuation_price.json next to the raw.jsonl, plus the log.

    python scripts/valuation_price_check.py [--raw data/edge_scan/leaderboard_analysis/2026-10-07/raw.jsonl] [--limit 20]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, read_jsonl, write_csv, write_json  # noqa: E402

logger = logging.getLogger("valuation_price_check")

DEFAULT_LIMIT = 20
MATCH_TOL = 1e-4
# A rule "explains" the mark when it matches on at least this share of comparable positions.
EXPLAINS_SHARE = 0.9
MIN_COMPARABLE = 5  # fewer comparable positions -> INSUFFICIENT DATA
SPEC_DEFINITION = {
    "field": "Position.currentPrice (GET /portfolio/positions, /tournaments/{slug}/portfolio/positions); marketValue derives from it",
    "Position.currentPrice": "Global price for unbound keys or tournament valuation price for organization-bound keys. Null only for zero-quantity tournament rows.",
    "TournamentPortfolioPosition.currentPrice": "Tournament valuation price. Null only for zero-quantity rows; nonzero holdings without a valuation return 409.",
    "/tournaments/{slug}/portfolio/positions": "Prices use tournament valuation_price. If nonzero holdings exist without valuation_price, returns 409 instead of valuing them at a neutral default.",
    "end-settlement (DMM, after the tournament)": "proposedPrice = manipulation-resistant 5h VWAP by default; sources vwap / cfmm_pool / seed / backfilled / admin_override / dmm_submission / missing.",
    "computation during the tournament": "not stated in the spec",
}
CSV_FIELDS = ("exchange_id", "quantity", "current_price", "market_value", "denomination", "mark_yes", "bid", "ask", "mid", "last",
              "matches", "vs_spread", "diff_last", "diff_mid", "diff_bid", "price_source", "price_age_s")


def _ts(record: Mapping[str, Any]) -> datetime | None:
    raw = record.get("ts")
    return None if not raw else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def pair_positions_with_prices(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """(positions, exchange_id -> price row with "_source" / "_age_s", meta) from raw records.
    Uses the LAST positions response; prices read right after it first, earlier ones as fallback."""
    ok = [r for r in records if "body" in r]
    idx = max((i for i, r in enumerate(ok) if r.get("path") == "/portfolio/positions"), default=None)
    if idx is None:
        return [], {}, {"positions_at": None}
    positions_rec = ok[idx]
    positions = list((positions_rec["body"] or {}).get("positions") or [])
    t0 = _ts(positions_rec)
    prices: dict[str, dict[str, Any]] = {}

    def take(rec: Mapping[str, Any], source: str) -> None:
        t = _ts(rec)
        age = None if t is None or t0 is None else round((t - t0).total_seconds(), 1)
        for row in (rec["body"] or {}).get("data") or []:
            eid = str(row.get("exchangeId"))
            if eid not in prices:
                prices[eid] = {**row, "_source": source, "_age_s": age}

    j = idx + 1
    while j < len(ok) and ok[j].get("path") == "/exchanges/prices":
        take(ok[j], "snapshot")
        j += 1
    for rec in reversed(ok[:idx]):
        if rec.get("path") == "/exchanges/prices":
            take(rec, "earlier")
    return positions, prices, {"positions_at": positions_rec.get("ts")}


def analyse(positions: Sequence[Mapping[str, Any]], prices: Mapping[str, Mapping[str, Any]], limit: int = DEFAULT_LIMIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    held = [p for p in positions if float(p.get("quantity") or 0) != 0 and not p.get("settled")]
    held.sort(key=lambda p: -abs(float(p.get("marketValue") or 0.0)))
    denominations = Counter(stats.mark_denomination(float(p["quantity"]), p.get("currentPrice"), p.get("marketValue"), MATCH_TOL) for p in held)
    # NO positions decide how currentPrice is denominated for everyone.
    convention = "side" if denominations.get("side") and not denominations.get("yes") else "yes"
    rows = []
    for pos in held[:limit]:
        quantity = float(pos["quantity"])
        current = pos.get("currentPrice")
        quote = prices.get(str(pos.get("exchangeId"))) or {}
        bid, ask, last = (None if quote.get(k) is None else float(quote[k]) for k in ("bestBid", "bestAsk", "latestPrice"))
        row: dict[str, Any] = {"exchange_id": str(pos.get("exchangeId")), "quantity": quantity, "current_price": current, "market_value": pos.get("marketValue"),
                               "denomination": stats.mark_denomination(quantity, current, pos.get("marketValue"), MATCH_TOL),
                               "bid": bid, "ask": ask, "mid": None if bid is None or ask is None else (bid + ask) / 2.0, "last": last,
                               "price_source": quote.get("_source"), "price_age_s": quote.get("_age_s")}
        if current is None:
            rows.append({**row, "mark_yes": None, "matches": "", "vs_spread": "no mark"})
            continue
        mark_yes = float(current) if quantity > 0 or convention == "yes" else 1.0 - float(current)
        cls = stats.classify_mark(mark_yes, bid, ask, last, MATCH_TOL)
        rows.append({**row, "mark_yes": mark_yes, "matches": "+".join(cls["matches"]), "vs_spread": cls["vs_spread"],
                     **{k: cls[k] for k in ("diff_last", "diff_mid", "diff_bid")}})

    def share(name: str) -> dict[str, Any]:
        comparable = [r for r in rows if r.get("mark_yes") is not None and r.get(name) is not None]
        hits = sum(1 for r in comparable if name in r["matches"].split("+"))
        diffs = [abs(r[f"diff_{name}"]) for r in comparable if r.get(f"diff_{name}") is not None]
        return {"comparable": len(comparable), "matches": hits, "share": round(hits / len(comparable), 3) if comparable else None,
                "median_abs_diff": round(statistics.median(diffs), 5) if diffs else None}

    by_rule = {name: share(name) for name in ("last", "mid", "bid")}
    explained = [name for name, s in by_rule.items() if s["comparable"] >= MIN_COMPARABLE and s["share"] >= EXPLAINS_SHARE]
    enough = any(s["comparable"] >= MIN_COMPARABLE for s in by_rule.values())
    summary = {
        "positions_held": len(held),
        "positions_compared": len(rows),
        "denomination_counts": {str(k): v for k, v in denominations.items()},
        "currentPrice_convention": convention,
        "by_rule": by_rule,
        "vs_spread": dict(Counter(r["vs_spread"] for r in rows)),
        "explained_by": explained,
        "verdict": (" / ".join(explained) if explained else "neither last trade nor mid nor bid (test a rolling VWAP next)") if enough else "INSUFFICIENT DATA",
        "note": "Coinciding references (e.g. last == mid) count for both; check the per-position 'matches' column.",
    }
    return rows, summary


def newest_raw(script: str = "leaderboard_analysis", root: Path = OUT_ROOT) -> Path | None:
    candidates = sorted((root / script).glob("*/raw.jsonl"))
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline: SIG valuation price (Position.currentPrice) vs bid / mid / last trade.")
    parser.add_argument("--raw", type=Path, default=None, help="raw.jsonl to read (default: newest leaderboard_analysis run)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"positions compared, largest first (default {DEFAULT_LIMIT})")
    args = parser.parse_args(argv)
    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    raw = args.raw or newest_raw()
    if raw is None or not raw.exists():
        logger.error("No raw.jsonl found: run scripts/leaderboard_analysis.py --yes first, or pass --raw", extra={"raw": str(raw)})
        return 2
    positions, prices, meta = pair_positions_with_prices(read_jsonl(raw))
    rows, summary = analyse(positions, prices, args.limit)
    summary = {"raw": str(raw), **meta, "spec": SPEC_DEFINITION, **summary}
    write_csv(raw.parent / "valuation_price.csv", rows, CSV_FIELDS)
    write_json(raw.parent / "valuation_price.json", {**summary, "rows": rows})
    for row in rows:
        logger.info("Valuation price vs quotes", extra={k: row.get(k) for k in CSV_FIELDS})
    logger.info("Valuation price summary", extra={k: summary[k] for k in ("positions_compared", "currentPrice_convention", "by_rule", "vs_spread", "verdict")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
