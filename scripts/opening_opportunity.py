"""READ-ONLY research: which exchanges offered the most at the opening (2026-10-01
16:00-18:00Z), and did D / R print pairs of one race sum away from 1? ALL NUMBERS ARE
PROXIES: the tape has no aggressor field and no account id, so nothing here says who
traded, or that any quote of ours would have been filled.

Inputs (no re-fetch): every /exchanges/{id}/trades response in the given raw.jsonl files
(default: the newest tape_patterns run and the newest leaderboard_analysis run), merged
and de-duplicated by trade id; exchange titles from tape_patterns' market-list pages.
An exchange counts as COVERED for the window only if some trade page of it ended the
history (hasMore false) and started at or before 16:00Z (no `from`, or from <= 16:00Z);
otherwise its window trades may be incomplete and its sums are lower bounds.
One optional read set: GET /exchanges/prices for the exchanges with window trades
(100 per read) + the tournament read, behind --yes, for the current mid.

1. Opportunity ranking: per exchange, the deviation volume = sum over window trades of
   size * |price - reference| (YES prices), for reference (a) the VWAP of the last
   non-empty 10-minute bucket before 18:00Z ("close VWAP", its bucket reported) and (b) the
   current SIG mid. Also per half of the window's trades (time order). Top 20 by each ->
   log; all -> ranking.csv. With race, party, trades, share priced < 0.10 or > 0.90.
2. D / R pairs: for every race with a D and an R market, each D print is paired with the
   nearest R print within 60 s; sum = D YES + R YES. Away from 1 by >= 0.02 = a print-pair
   mispricing PROXY (prints are trades, not standing quotes; a race with an Independent
   market is flagged - its D + R should sit below 1). Per race (and per half) ->
   pairs_by_race.csv; the 20 pairs furthest from 1 -> extreme_pairs.csv.
-> data/edge_scan/opening_opportunity/<UTC date>/ + summary.json / summary.md.

    python scripts/opening_opportunity.py [--raw PATH ...] [--yes]
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
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min, run_dir, write_csv, write_json  # noqa: E402

logger = logging.getLogger("opening_opportunity")

SCRIPT = "opening_opportunity"
WINDOW_START = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)
BUCKET_MIN, BUCKETS = 10, 12
EXTREME_LOW, EXTREME_HIGH = 0.10, 0.90
TOP_N = 20
PAIR_GAP_S = 60.0
PAIR_THRESHOLD = 0.02  # |D + R - 1| >= this = clearly away from 1 (fixed, 4 ticks)
PRICES_CHUNK = 100
PROXY = "PROXY: the tape has no aggressor and no account id - deviation volume is not P&L, a print pair is not an executable quote"

RANK_FIELDS = ("rank_close", "rank_mid", "exchange_id", "title", "race", "party", "covered", "trades", "shares", "extreme_share",
               "close_vwap", "close_bucket", "dev_vs_close", "dev_vs_close_first_half", "dev_vs_close_second_half", "current_mid", "dev_vs_mid")
PAIR_RACE_FIELDS = ("race", "has_independent", "d_exchange", "r_exchange", "pairs", "median_sum", "min_sum", "max_sum", "share_above", "share_below",
                    "median_sum_first_half", "median_sum_second_half")
PAIR_FIELDS = ("race", "has_independent", "time_d", "time_r", "gap_s", "price_d", "price_r", "size_d", "size_r", "sum", "excess")


def _ts(raw: Any) -> datetime:
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------- raw -> window tapes


def window_tapes(records: Sequence[Mapping[str, Any]], start: datetime = WINDOW_START, end: datetime = WINDOW_END) -> tuple[dict[str, list[dict[str, Any]]], set[str], dict[str, str]]:
    """(exchange -> window trades in time order, exchanges COVERED for the window, exchange -> title)."""
    trades: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    covered: set[str] = set()
    titles: dict[str, str] = {}
    for rec in records:
        path = str(rec.get("path") or "")
        if "body" not in rec:
            continue
        body = rec["body"] or {}
        if re.fullmatch(r"/tournaments/[^/]+/markets", path):
            for market in body.get("data") or []:
                for ex in market.get("exchanges") or []:
                    titles[str(ex["id"])] = market.get("title")
            continue
        if not (path.startswith("/exchanges/") and path.endswith("/trades")):
            continue
        eid = path.split("/")[2]
        page_from = (rec.get("params") or {}).get("from")
        if not (body.get("pagination") or {}).get("hasMore") and (page_from is None or _ts(page_from) <= start):
            covered.add(eid)
        for t in body.get("data") or []:
            if t.get("createdAt") and t.get("price") is not None and start <= _ts(t["createdAt"]) < end:
                trades[eid][str(t.get("id"))] = t
    tapes = {eid: sorted(ts.values(), key=lambda t: (str(t["createdAt"]), str(t.get("id")))) for eid, ts in trades.items()}
    return tapes, covered, titles


def close_vwap(trades: Sequence[Mapping[str, Any]]) -> tuple[float | None, str | None]:
    """VWAP of the last non-empty 10-minute bucket of the window, and that bucket's start."""
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for t in trades:
        idx = stats.bucket_index(_ts(t["createdAt"]), WINDOW_START, BUCKET_MIN, BUCKETS)
        if idx is not None:
            buckets[idx].append((float(t["price"]), float(t.get("size") or 0)))
    for idx in sorted(buckets, reverse=True):
        value = stats.vwap(buckets[idx])
        if value is not None:
            return value, (WINDOW_START + timedelta(minutes=BUCKET_MIN * idx)).strftime("%H:%M")
    return None, None


# ---------------------------------------------------------------------- 1. ranking


def ranking_rows(tapes: Mapping[str, Sequence[Mapping[str, Any]]], covered: set[str], titles: Mapping[str, str], mids: Mapping[str, float | None],
                 market_map: MarketMap) -> list[dict[str, Any]]:
    rows = []
    for eid, trades in tapes.items():
        pairs = [(float(t["price"]), float(t.get("size") or 0)) for t in trades]
        ref, bucket = close_vwap(trades)
        first, second = stats.split_halves(trades, key=lambda t: (str(t["createdAt"]), str(t.get("id"))))
        m = market_map.get(eid)

        def dev(ts: Sequence[Mapping[str, Any]], r: float | None) -> float | None:
            value = stats.deviation_volume(((float(t["price"]), float(t.get("size") or 0)) for t in ts), r)
            return None if value is None else round(value, 2)

        mid = mids.get(eid)
        rows.append({"exchange_id": eid, "title": titles.get(eid) or (m.title if m else None), "race": m.race.key if m else None, "party": m.party if m else None,
                     "covered": eid in covered, "trades": len(trades), "shares": math.fsum(s for _, s in pairs),
                     "extreme_share": stats.share_outside((p for p, _ in pairs), EXTREME_LOW, EXTREME_HIGH),
                     "close_vwap": None if ref is None else round(ref, 4), "close_bucket": bucket,
                     "dev_vs_close": dev(trades, ref), "dev_vs_close_first_half": dev(first, ref), "dev_vs_close_second_half": dev(second, ref),
                     "current_mid": mid, "dev_vs_mid": dev(trades, mid)})
    for key, rank in (("dev_vs_close", "rank_close"), ("dev_vs_mid", "rank_mid")):
        ordered = sorted((r for r in rows if r[key] is not None), key=lambda r: (-r[key], r["exchange_id"]))
        for i, r in enumerate(ordered, start=1):
            r[rank] = i
    return sorted(rows, key=lambda r: (r.get("rank_close") or math.inf, r["exchange_id"]))


# ---------------------------------------------------------------------- 2. D / R pairs


def race_pairs(tapes: Mapping[str, Sequence[Mapping[str, Any]]], market_map: MarketMap) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(per-race summary, every pair) for races with a D and an R market that both printed."""
    summaries, all_pairs = [], []
    for race, group in sorted(market_map.by_race().items()):
        by_party = {m.party: m.sig_exchange_id for m in group}
        d, r = by_party.get("D"), by_party.get("R")
        if not d or not r or not tapes.get(d) or not tapes.get(r):
            continue
        has_i = "I" in by_party

        def items(eid: str) -> list[tuple[float, float, float]]:
            return [(_ts(t["createdAt"]).timestamp(), float(t["price"]), float(t.get("size") or 0)) for t in tapes[eid]]

        pairs = stats.nearest_pairs(items(d), items(r), PAIR_GAP_S)
        rows = [{"race": race, "has_independent": has_i, "time_d": _iso(p["t_a"]), "time_r": _iso(p["t_b"]), "gap_s": round(p["gap_s"], 1),
                 "price_d": p["price_a"], "price_r": p["price_b"], "size_d": p["item_a"][2], "size_r": p["item_b"][2],
                 "sum": round(p["sum"], 4), "excess": round(p["sum"] - 1.0, 4)} for p in pairs]
        all_pairs += rows
        sums = [x["sum"] for x in rows]
        first, second = stats.split_halves(rows, key=lambda x: x["time_d"])
        summaries.append({"race": race, "has_independent": has_i, "d_exchange": d, "r_exchange": r, "pairs": len(rows),
                          "median_sum": stats.median_or_none(sums), "min_sum": min(sums, default=None), "max_sum": max(sums, default=None),
                          "share_above": round(sum(1 for s in sums if s >= 1 + PAIR_THRESHOLD - 1e-9) / len(sums), 4) if sums else None,
                          "share_below": round(sum(1 for s in sums if s <= 1 - PAIR_THRESHOLD + 1e-9) / len(sums), 4) if sums else None,
                          "median_sum_first_half": stats.median_or_none(x["sum"] for x in first),
                          "median_sum_second_half": stats.median_or_none(x["sum"] for x in second)})
    return summaries, all_pairs


# ---------------------------------------------------------------------- run


def default_raws(root: Path = OUT_ROOT) -> list[Path]:
    found = []
    for script in ("tape_patterns", "leaderboard_analysis"):
        runs = sorted((root / script).glob("*/raw.jsonl"))
        if runs:
            found.append(runs[-1])
    return found


def analyse(records: Sequence[Mapping[str, Any]], mids: Mapping[str, float | None], market_map: MarketMap) -> dict[str, Any]:
    tapes, covered, titles = window_tapes(records)
    ranking = ranking_rows(tapes, covered, titles, mids, market_map)
    pair_summary, pairs = race_pairs(tapes, market_map)
    extreme = sorted(pairs, key=lambda p: -abs(p["excess"]))[:TOP_N]
    two_party = [p for p in pairs if not p["has_independent"]]
    summary = {
        "label": PROXY,
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "exchanges_with_window_trades": len(tapes), "covered": sum(1 for r in ranking if r["covered"]),
        "not_covered_note": "an exchange not covered may miss window trades: its sums are lower bounds",
        "top_by_close": [{k: r[k] for k in ("exchange_id", "race", "party", "trades", "dev_vs_close")} for r in ranking if r.get("rank_close") and r["rank_close"] <= TOP_N],
        "top_by_mid": [{k: r[k] for k in ("exchange_id", "race", "party", "trades", "dev_vs_mid")} for r in sorted(ranking, key=lambda r: r.get("rank_mid") or math.inf)
                       if r.get("rank_mid") and r["rank_mid"] <= TOP_N],
        "pairs": {"races": len(pair_summary), "pairs": len(pairs), "threshold": PAIR_THRESHOLD, "gap_s": PAIR_GAP_S,
                  "two_party_pairs": len(two_party),
                  "two_party_share_away_from_1": round(sum(1 for p in two_party if abs(p["excess"]) >= PAIR_THRESHOLD - 1e-9) / len(two_party), 4) if two_party else None,
                  "median_two_party_sum": stats.median_or_none(p["sum"] for p in two_party)},
    }
    return {"ranking": ranking, "pair_summary": pair_summary, "extreme_pairs": extreme, "summary": summary}


def summary_markdown(result: Mapping[str, Any]) -> str:
    s = result["summary"]
    lines = ["# Opening opportunity 2026-10-01 16:00-18:00Z (PROXIES)", "", f"_{s['label']}._", "",
             f"- {s['exchanges_with_window_trades']} exchanges printed in the window; {s['covered']} fully covered by the raw data.",
             "- Top 10 by deviation volume vs the close VWAP: " + ", ".join(f"{r['exchange_id']} ({r['race']} {r['party']}) {r['dev_vs_close']}" for r in s["top_by_close"][:10]) + ".",
             f"- D / R print pairs within {s['pairs']['gap_s']:g} s: {s['pairs']['pairs']} in {s['pairs']['races']} races; two-party races: median sum "
             f"{s['pairs']['median_two_party_sum']}, share >= {s['pairs']['threshold']} away from 1: {s['pairs']['two_party_share_away_from_1']}.", ""]
    return "\n".join(lines)


async def _read_mids(ids: Sequence[str], reads_per_min: float) -> dict[str, float | None]:
    mids: dict[str, float | None] = {}
    async with research_reader(SCRIPT, reads_per_min) as reader:
        tid = (await reader.get(f"/tournaments/{reader.settings.tournament_slug}")).get("id")
        for start in range(0, len(ids), PRICES_CHUNK):
            body = await reader.get("/exchanges/prices", {"ids": ",".join(ids[start : start + PRICES_CHUNK]), "tournamentId": tid})
            for row in body.get("data") or []:
                bid, ask = row.get("bestBid"), row.get("bestAsk")
                mids[str(row["exchangeId"])] = None if bid is None or ask is None else round((float(bid) + float(ask)) / 2.0, 4)
    return mids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: opening opportunity ranking and D / R print-pair sums (proxies).")
    add_research_args(parser)
    parser.add_argument("--raw", type=Path, action="append", default=None, help="raw.jsonl to read (repeatable; default: newest tape_patterns + leaderboard_analysis)")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    raws = args.raw or default_raws()
    if not raws or not all(p.exists() for p in raws):
        logger.error("raw.jsonl not found: run tape_patterns.py / leaderboard_analysis.py first, or pass --raw", extra={"raw": [str(p) for p in raws]})
        return 2
    records = [r for p in raws for r in read_jsonl(p)]
    tapes, _, _ = window_tapes(records)
    if not tapes:
        logger.error("No trades in the opening window in the raw files", extra={"raw": [str(p) for p in raws]})
        return 2
    mids: dict[str, float | None] = {}
    ids = sorted(tapes)
    if confirm_fetch(ReadEstimate(1 + math.ceil(len(ids) / PRICES_CHUNK), reads_per_min), args.yes):
        mids = asyncio.run(_read_mids(ids, reads_per_min))
    else:
        logger.info("Continuing offline: the current-mid ranking stays empty")
    result = analyse(records, mids, MarketMap.load())

    out = run_dir(SCRIPT)
    write_csv(out / "ranking.csv", result["ranking"], RANK_FIELDS)
    write_csv(out / "pairs_by_race.csv", result["pair_summary"], PAIR_RACE_FIELDS)
    write_csv(out / "extreme_pairs.csv", result["extreme_pairs"], PAIR_FIELDS)
    write_json(out / "summary.json", {"raw": [str(p) for p in raws], **result["summary"]})
    (out / "summary.md").write_text(summary_markdown(result), encoding="utf-8")
    for row in result["ranking"][:TOP_N]:
        logger.info("Top by deviation vs close VWAP (PROXY)", extra={k: row.get(k) for k in RANK_FIELDS})
    for row in result["extreme_pairs"]:
        logger.info("D / R print pair far from 1 (PROXY)", extra=row)
    logger.info("Opening opportunity done (PROXIES)", extra={"out_dir": str(out), **result["summary"]["pairs"], "label": PROXY})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
