"""READ-ONLY research A: leaderboard structure, the opening tape, big resting levels, and
which valuation rule the leaderboard uses.

1. Leaderboard - /tournaments/{slug}/leaderboard for periods all, 7d and 1d, every
   participant (100 per page). Per period: zero-sum test (sum of P&L, winners vs losers),
   P&L percentiles and our rank. SIG periods are CALENDAR-aligned (UTC): 7d = since the
   latest Sunday 00:00, 1d = since today 00:00. So the top 20 by all-time P&L are split
   into early = all - 7d (before Sunday), mid = 7d - 1d (Sunday to today) and today = 1d.
   Plus the top 20 by 1d. A participant missing from a period's list has that part None,
   never 0. Summaries carry ranks only: no usernames or profile ids (only "is_us").
2. Opening tape - /exchanges/{id}/trades for 2026-10-01T16:00Z-18:00Z on 10 markets
   (--tape-exchanges, else the 10 mapped D / R markets with a two-sided book whose YES
   mid is closest to 0.50). Size, price and side (YES / NO space) distributions, and an
   aggressor PROXY from the tick rule: the tape has no aggressor field. Reported pooled,
   per market and for the first vs second half of the trades (time order).
3. Big resting levels - order books (depth 50) of the mapped markets with at least one
   quoted side, at most --max-books of them (two-sided books closest to 0.50 first, then
   one-sided); levels of >= 3000 shares after our own resting orders are taken out.
4. Valuation check - the leaderboard reports P&L, not a balance. Implied value =
   initialBalance + our all-time leaderboard P&L, compared with cash (tournament myBalance)
   + our positions valued at the bid, at the mid, and at SIG's own valuation price
   (positions' marketValue), plus /portfolio/pnl totalAccountValue. Read back to back
   (positions, held prices, pnl, tournament, our leaderboard row) so prices barely move.

Reads only, through research.reader (guarded client: any write raises; at most
RESEARCH_READS_PER_MIN reads/min, default 10). Without --yes only the read estimate is
logged. Raw responses -> data/edge_scan/leaderboard_analysis/<UTC date>/raw.jsonl;
summaries -> summary.json, leaderboard_top20.csv, leaderboard_top20_1d.csv,
tape_markets.csv, big_levels.csv, valuation.csv in the same folder.

    python scripts/leaderboard_analysis.py [--yes] [--reads-per-min 10] [--max-books 237]
        [--tape-exchanges 1077,1078] [--participants-estimate 1000]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from exchange_api import book_without_own, yes_terms  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import ReadEstimate, ResearchReader, add_research_args, confirm_fetch, research_reader, resolve_reads_per_min, write_csv, write_json  # noqa: E402

logger = logging.getLogger("leaderboard_analysis")

SCRIPT = "leaderboard_analysis"
PERIODS = ("all", "7d", "1d")
LEADERBOARD_PAGE = 100  # API maximum
TOP_N = 20
TAPE_FROM = "2026-10-01T16:00:00Z"
TAPE_TO = "2026-10-01T18:00:00Z"
TAPE_MARKETS = 10
TAPE_PAGE = 200  # API maximum
TAPE_MAX_PAGES = 5  # per market; 1000 trades in 2 h is more than any market saw
TAPE_PAGES_ESTIMATE = 2  # per market, for the read estimate only
BIG_LEVEL_SHARES = 3000
BOOK_DEPTH = 50
PRICES_CHUNK = 100  # /exchanges/prices maximum
ORDERS_PAGE = 200
DEFAULT_PARTICIPANTS_ESTIMATE = 1000
# Our row is re-read in the valuation snapshot from this many ranks above us.
OWN_ROW_WINDOW = 10

TOP_FIELDS = ("rank", "is_us", "pnl_all", "pnl_7d", "pnl_1d", "early", "mid", "today", "trades_all")
TOP_1D_FIELDS = ("rank_1d", "is_us", "pnl_1d", "pnl_all", "rank_all", "trades_1d")
TAPE_FIELDS = ("exchange_id", "race", "party", "trades", "shares", "vwap", "min_price", "max_price", "median_size", "max_size", "yes_space_share", "proxy_buy_share")
BIG_LEVEL_FIELDS = ("exchange_id", "race", "party", "side", "price", "quantity", "rank", "distance", "best_bid", "best_ask")
VALUATION_FIELDS = ("rule", "value", "residual_vs_leaderboard", "unvalued_positions")


# ---------------------------------------------------------------------- read plan


def estimate_reads(participants: int, map_size: int, max_books: int, tape_markets: int) -> int:
    """Upper-ish estimate: tournament + account, leaderboard pages for three periods, bulk
    prices for the map, open orders, the 5-read valuation snapshot, the tape and the books."""
    leaderboard = len(PERIODS) * max(1, math.ceil(participants / LEADERBOARD_PAGE))
    prices = math.ceil(map_size / PRICES_CHUNK)
    return 2 + leaderboard + prices + 1 + 5 + tape_markets * TAPE_PAGES_ESTIMATE + min(max_books, map_size)


# ---------------------------------------------------------------------- leaderboard


def leaderboard_by_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["profileId"]): dict(r) for r in rows if r.get("profileId") is not None}


def analyse_leaderboard(boards: Mapping[str, Mapping[str, Any]], my_profile_id: str | None, top_n: int = TOP_N) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """(summary, top_n by all-time with the early / mid / today split, top_n by 1d).
    `boards`: period -> {"rows": [LeaderboardEntry], "total", "my_rank"}."""
    by_period = {p: leaderboard_by_profile(boards[p]["rows"]) for p in PERIODS}
    summary: dict[str, Any] = {}
    for period in PERIODS:
        rows = boards[period]["rows"]
        pnls = [float(r.get("pnl") or 0.0) for r in rows]
        summary[period] = {
            "total_reported": boards[period].get("total"),
            "rows_read": len(rows),
            "my_rank": boards[period].get("my_rank"),
            "zero_sum": stats.zero_sum_check(pnls),
            "pnl": stats.describe(pnls),
        }
        if my_profile_id and my_profile_id in by_period[period]:
            mine = float(by_period[period][my_profile_id].get("pnl") or 0.0)
            summary[period]["my_pnl"] = mine
            summary[period]["my_pnl_percentile"] = round(100.0 * sum(1 for v in pnls if v < mine) / len(pnls), 2) if pnls else None

    def pnl(period: str, pid: str) -> float | None:
        row = by_period[period].get(pid)
        return None if row is None else float(row.get("pnl") or 0.0)

    top_all = sorted(boards["all"]["rows"], key=lambda r: (r.get("rank") or math.inf, -(r.get("pnl") or 0.0)))[:top_n]
    top: list[dict[str, Any]] = []
    for row in top_all:
        pid = str(row.get("profileId"))
        parts = stats.pnl_decomposition(pnl("all", pid), pnl("7d", pid), pnl("1d", pid))
        top.append({"rank": row.get("rank"), "is_us": pid == my_profile_id, "pnl_all": pnl("all", pid), "pnl_7d": pnl("7d", pid),
                    "pnl_1d": pnl("1d", pid), **parts, "trades_all": row.get("tradesCount")})

    top_1d_rows = sorted(boards["1d"]["rows"], key=lambda r: (r.get("rank") or math.inf, -(r.get("pnl") or 0.0)))[:top_n]
    top_1d = []
    for row in top_1d_rows:
        pid = str(row.get("profileId"))
        all_row = by_period["all"].get(pid) or {}
        top_1d.append({"rank_1d": row.get("rank"), "is_us": pid == my_profile_id, "pnl_1d": pnl("1d", pid), "pnl_all": pnl("all", pid),
                       "rank_all": all_row.get("rank"), "trades_1d": row.get("tradesCount")})

    # How much of the top 20's all-time P&L came from each calendar part.
    sums = {part: math.fsum(r[part] for r in top if r[part] is not None) for part in ("early", "mid", "today")}
    summary["top_decomposition"] = {
        "top_n": len(top),
        "missing_parts": sum(1 for r in top for part in ("early", "mid", "today") if r[part] is None),
        "sum_by_part": sums,
        "top_by_1d_also_top_all": sum(1 for r in top_1d if r["rank_all"] is not None and r["rank_all"] <= top_n),
    }
    return summary, top, top_1d


async def fetch_leaderboard(reader: ResearchReader, slug: str, period: str) -> dict[str, Any]:
    """Every row of one period (offset pagination until `total` rows or an empty page)."""
    rows: list[dict[str, Any]] = []
    total = my_rank = None
    while True:
        page = await reader.get(f"/tournaments/{slug}/leaderboard", {"period": period, "limit": LEADERBOARD_PAGE, "offset": len(rows)})
        batch = page.get("leaderboard") or []
        total, my_rank = page.get("total"), page.get("myRank")
        rows.extend(batch)
        if not batch or len(batch) < LEADERBOARD_PAGE or (total is not None and len(rows) >= total):
            break
    return {"rows": rows, "total": total, "my_rank": my_rank}


# ---------------------------------------------------------------------- tape


def _mid(row: Mapping[str, Any]) -> float | None:
    bid, ask = row.get("bestBid"), row.get("bestAsk")
    return None if bid is None or ask is None else (float(bid) + float(ask)) / 2.0


def contested_first(market_map: MarketMap, prices: Mapping[str, Mapping[str, Any]], parties: tuple[str, ...] = ("D", "R")) -> list[str]:
    """Mapped exchange ids with a two-sided book, YES mid closest to 0.50 first (ties by id)."""
    ranked = []
    for mapping in market_map:
        mid = _mid(prices.get(mapping.sig_exchange_id) or {})
        if mapping.party in parties and mid is not None:
            ranked.append((abs(mid - 0.5), mapping.sig_exchange_id))
    return [eid for _, eid in sorted(ranked)]


def book_scan_order(market_map: MarketMap, prices: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Every mapped exchange with at least one quoted side (a big level can sit on a
    one-sided book): two-sided books closest to 0.50 first, then one-sided ones by id."""
    two_sided = contested_first(market_map, prices, parties=("D", "R", "I"))
    seen = set(two_sided)
    one_sided = sorted(
        m.sig_exchange_id
        for m in market_map
        if m.sig_exchange_id not in seen
        and ((prices.get(m.sig_exchange_id) or {}).get("bestBid") is not None or (prices.get(m.sig_exchange_id) or {}).get("bestAsk") is not None)
    )
    return two_sided + one_sided


def _trade_time(trade: Mapping[str, Any]) -> str:
    return str(trade.get("createdAt") or "")


def summarise_trades(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Size / price / side mix of a set of trades plus the tick-rule aggressor proxy.
    Trades with no price are counted but left out of price statistics."""
    ordered = sorted(trades, key=lambda t: (_trade_time(t), str(t.get("id"))))
    priced = [t for t in ordered if t.get("price") is not None]
    sizes = [float(t.get("size") or 0) for t in ordered]
    shares = math.fsum(sizes)
    proxy = stats.tick_rule([float(t["price"]) for t in priced])
    called = [c for c in proxy if c is not None]
    vwap_den = math.fsum(float(t.get("size") or 0) for t in priced)
    return {
        "trades": len(ordered),
        "shares": shares,
        "vwap": math.fsum(float(t["price"]) * float(t.get("size") or 0) for t in priced) / vwap_den if vwap_den else None,
        "min_price": min((float(t["price"]) for t in priced), default=None),
        "max_price": max((float(t["price"]) for t in priced), default=None),
        "median_size": stats.describe(sizes)["p50"],
        "max_size": max(sizes, default=None),
        "size": stats.describe(sizes),
        "size_buckets": stats.share_table((stats.size_bucket(s) for s in sizes), [label for _, label in stats.SIZE_BUCKETS]),
        "price_buckets": stats.share_table((stats.price_bucket(float(t["price"])) for t in priced), [label for _, label in stats.PRICE_BUCKETS]),
        "side": stats.share_table((str(t.get("side")) for t in ordered), ["YES", "NO"]),
        "yes_space_share": round(sum(1 for t in ordered if t.get("side") == "YES") / len(ordered), 4) if ordered else None,
        "aggressor_proxy": "tick rule on YES price (the tape has no aggressor field)",
        "proxy_buy_share": round(called.count("buy") / len(called), 4) if called else None,
        "proxy_uncalled": len(proxy) - len(called),
    }


def summarise_tape(tapes: Mapping[str, Sequence[Mapping[str, Any]]], market_map: MarketMap) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(pooled + first / second half summary, one row per market)."""
    rows = []
    for eid, trades in tapes.items():
        mapping = market_map.get(eid)
        rows.append({"exchange_id": eid, "race": mapping.race.key if mapping else None, "party": mapping.party if mapping else None,
                     **{k: v for k, v in summarise_trades(trades).items() if k in TAPE_FIELDS}})
    pooled = [t for trades in tapes.values() for t in trades]
    first, second = stats.split_halves(pooled, key=lambda t: (_trade_time(t), str(t.get("id"))))
    summary = {
        "window": [TAPE_FROM, TAPE_TO],
        "markets": len(tapes),
        "pooled": summarise_trades(pooled),
        "first_half": summarise_trades(first),
        "second_half": summarise_trades(second),
        "halves_split_at": _trade_time(second[0]) if second else None,
    }
    return summary, rows


async def fetch_tape(reader: ResearchReader, exchange_id: str, tournament_id: str | None, start: str, end: str, max_pages: int = TAPE_MAX_PAGES) -> tuple[list[dict[str, Any]], bool]:
    """(trades in [start, end), complete). Stops after `max_pages` pages -> complete False."""
    trades: list[dict[str, Any]] = []
    cursor = None
    for _ in range(max_pages):
        body = await reader.get(f"/exchanges/{exchange_id}/trades", {"from": start, "to": end, "limit": TAPE_PAGE, "cursor": cursor, "tournamentId": tournament_id})
        trades.extend(body.get("data") or [])
        pagination = body.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            return trades, True
    return trades, False


# ---------------------------------------------------------------------- books


def own_orders_by_exchange(open_orders: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[str, float, float]]]:
    """exchange_id -> (book side, YES price, remaining) of our resting orders. The remaining
    quantity is signed by side on SIG (NO negative), so only its size is used."""
    own: dict[str, list[tuple[str, float, float]]] = {}
    for order in open_orders:
        if order.get("open") is False or not order.get("exchangeId"):
            continue
        side, price = yes_terms(order.get("side"), order.get("action"), order.get("priceLimit"))
        remaining = order.get("remainingQuantity")
        size = abs(float(remaining if remaining is not None else order.get("quantity") or 0))
        own.setdefault(str(order["exchangeId"]), []).append((side, price, size))
    return own


def big_levels_rows(exchange_id: str, book: Mapping[str, Any], own: Sequence[tuple[str, float, float]], market_map: MarketMap, min_size: float = BIG_LEVEL_SHARES) -> list[dict[str, Any]]:
    others = book_without_own(dict(book), own)
    bids, asks = others.get("bids") or [], others.get("asks") or []
    mapping = market_map.get(exchange_id)
    context = {"exchange_id": exchange_id, "race": mapping.race.key if mapping else None, "party": mapping.party if mapping else None,
               "best_bid": float(bids[0]["price"]) if bids else None, "best_ask": float(asks[0]["price"]) if asks else None}
    return [{**context, **level} for level in stats.large_levels(others, min_size)]


# ---------------------------------------------------------------------- valuation


def valuation_check(
    initial_balance: float,
    leaderboard_pnl_all: float | None,
    cash: float,
    positions: Sequence[Mapping[str, Any]],
    prices: Mapping[str, Mapping[str, Any]],
    total_account_value: float | None,
) -> dict[str, Any]:
    """Which rule reproduces initialBalance + our leaderboard P&L. Settled positions count
    at their marketValue under every rule."""
    totals: dict[str, float] = {"bid": 0.0, "mid": 0.0, "sig_valuation_price": 0.0}
    unvalued = {"bid": 0, "mid": 0, "sig_valuation_price": 0}
    for pos in positions:
        quantity = float(pos.get("quantity") or 0.0)
        if quantity == 0:
            continue
        market_value = pos.get("marketValue")
        if market_value is None:
            unvalued["sig_valuation_price"] += 1
        else:
            totals["sig_valuation_price"] += float(market_value)
        for rule in ("bid", "mid"):
            if pos.get("settled") and market_value is not None:
                totals[rule] += float(market_value)
                continue
            row = prices.get(str(pos.get("exchangeId"))) or {}
            bid = None if row.get("bestBid") is None else float(row["bestBid"])
            ask = None if row.get("bestAsk") is None else float(row["bestAsk"])
            value = stats.position_value(quantity, bid, ask, rule)
            if value is None:
                unvalued[rule] += 1
            else:
                totals[rule] += value
    candidates: dict[str, float | None] = {f"cash+{rule}": (cash + totals[rule] if unvalued[rule] == 0 else None) for rule in totals}
    candidates["portfolio_pnl_totalAccountValue"] = total_account_value
    result: dict[str, Any] = {"initial_balance": initial_balance, "leaderboard_pnl_all": leaderboard_pnl_all, "cash": cash,
                              "holdings": totals, "unvalued_positions": unvalued, "candidates": candidates}
    if leaderboard_pnl_all is None:
        return {**result, "implied_leaderboard_value": None, "closest_rule": None, "residuals": None}
    implied = initial_balance + leaderboard_pnl_all
    best, residuals = stats.closest_rule(implied, candidates)
    return {**result, "implied_leaderboard_value": implied, "closest_rule": best, "residuals": residuals}


def valuation_rows(check: Mapping[str, Any]) -> list[dict[str, Any]]:
    rules = {"cash+bid": "bid", "cash+mid": "mid", "cash+sig_valuation_price": "sig_valuation_price"}
    return [{"rule": name, "value": value, "residual_vs_leaderboard": (check.get("residuals") or {}).get(name),
             "unvalued_positions": check["unvalued_positions"].get(rules.get(name, ""), 0)} for name, value in check["candidates"].items()]


async def _prices(reader: ResearchReader, ids: Sequence[str], tournament_id: str | None) -> dict[str, dict[str, Any]]:
    prices: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), PRICES_CHUNK):
        body = await reader.get("/exchanges/prices", {"ids": ",".join(ids[start : start + PRICES_CHUNK]), "tournamentId": tournament_id})
        prices.update({str(row["exchangeId"]): row for row in body.get("data") or []})
    return prices


async def _open_orders(reader: ResearchReader) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = await reader.get("/orders", {"status": "open", "limit": ORDERS_PAGE, "cursor": cursor})
        orders.extend(page.get("data") or [])
        pagination = page.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            return orders


async def valuation_snapshot(reader: ResearchReader, slug: str, tournament_id: str | None, my_profile_id: str | None, my_rank: int | None, fallback_pnl: float | None) -> dict[str, Any]:
    """Positions, their prices, pnl, tournament and our leaderboard row, back to back."""
    positions = (await reader.get("/portfolio/positions")).get("positions") or []
    held = sorted({str(p["exchangeId"]) for p in positions if float(p.get("quantity") or 0) != 0 and p.get("exchangeId")})
    prices = await _prices(reader, held, tournament_id) if held else {}
    pnl = await reader.get("/portfolio/pnl", {"period": "all"})
    tournament = await reader.get(f"/tournaments/{slug}")
    lb_pnl, source = fallback_pnl, "full leaderboard read"
    if my_rank is not None and my_profile_id:
        page = await reader.get(f"/tournaments/{slug}/leaderboard", {"period": "all", "limit": OWN_ROW_WINDOW, "offset": max(0, my_rank - 1 - OWN_ROW_WINDOW // 2)})
        mine = leaderboard_by_profile(page.get("leaderboard") or []).get(my_profile_id)
        if mine is not None:
            lb_pnl, source = float(mine.get("pnl") or 0.0), "re-read in the snapshot"
    check = valuation_check(float(tournament.get("initialBalance") or 0.0), lb_pnl, float(tournament.get("myBalance") or 0.0),
                            positions, prices, None if pnl.get("totalAccountValue") is None else float(pnl["totalAccountValue"]))
    holdings_value = pnl.get("totalHoldingsValue")
    cash_from_pnl = None if pnl.get("totalAccountValue") is None or holdings_value is None else float(pnl["totalAccountValue"]) - float(holdings_value)
    return {**check, "leaderboard_pnl_source": source, "cash_from_portfolio_pnl": cash_from_pnl, "positions_held": len(held)}


# ---------------------------------------------------------------------- run


async def run(reader: ResearchReader, slug: str, market_map: MarketMap, max_books: int, tape_exchanges: Sequence[str] | None) -> dict[str, Any]:
    out = reader.out_dir
    tournament = await reader.get(f"/tournaments/{slug}")
    tournament_id = tournament.get("id")
    members = tournament.get("memberCount")
    logger.info("Leaderboard plan", extra={"members": members, "pages_per_period": math.ceil((members or 0) / LEADERBOARD_PAGE)})
    account = await reader.get("/account")
    my_profile_id = None if account.get("id") is None else str(account["id"])

    boards = {period: await fetch_leaderboard(reader, slug, period) for period in PERIODS}
    lb_summary, top, top_1d = analyse_leaderboard(boards, my_profile_id)
    if boards["all"]["my_rank"] is not None and "my_pnl" not in lb_summary["all"]:
        logger.warning("Our account id not found in the leaderboard rows: own P&L / valuation check unavailable", extra={"my_rank": boards["all"]["my_rank"]})
    write_csv(out / "leaderboard_top20.csv", top, TOP_FIELDS)
    write_csv(out / "leaderboard_top20_1d.csv", top_1d, TOP_1D_FIELDS)
    for period in PERIODS:
        logger.info("Leaderboard period", extra={"period": period, **{k: lb_summary[period].get(k) for k in ("rows_read", "my_rank", "my_pnl", "my_pnl_percentile")},
                                                 "sum_pnl": lb_summary[period]["zero_sum"]["sum_pnl"], "median_pnl": lb_summary[period]["pnl"]["p50"]})

    prices = await _prices(reader, [m.sig_exchange_id for m in market_map], tournament_id)
    own = own_orders_by_exchange(await _open_orders(reader))

    my_all = (leaderboard_by_profile(boards["all"]["rows"]).get(my_profile_id or "") or {}).get("pnl")
    valuation = await valuation_snapshot(reader, slug, tournament_id, my_profile_id, boards["all"]["my_rank"], None if my_all is None else float(my_all))
    write_csv(out / "valuation.csv", valuation_rows(valuation), VALUATION_FIELDS)
    logger.info("Valuation check", extra={k: valuation.get(k) for k in ("implied_leaderboard_value", "closest_rule", "residuals", "leaderboard_pnl_source")})

    ranked = contested_first(market_map, prices)
    tape_ids = list(tape_exchanges) if tape_exchanges else ranked[:TAPE_MARKETS]
    tapes, incomplete = {}, []
    for eid in tape_ids:
        tapes[eid], complete = await fetch_tape(reader, eid, tournament_id, TAPE_FROM, TAPE_TO)
        if not complete:
            incomplete.append(eid)
    tape_summary, tape_rows = summarise_tape(tapes, market_map)
    tape_summary["incomplete_markets"] = incomplete
    write_csv(out / "tape_markets.csv", tape_rows, TAPE_FIELDS)
    logger.info("Opening tape", extra={k: tape_summary["pooled"].get(k) for k in ("trades", "shares", "vwap", "median_size", "yes_space_share", "proxy_buy_share")})

    big: list[dict[str, Any]] = []
    books_read = 0
    for eid in book_scan_order(market_map, prices)[:max_books]:
        book = await reader.get(f"/exchanges/{eid}/orderbook", {"depth": BOOK_DEPTH, "tournamentId": tournament_id})
        books_read += 1
        big.extend(big_levels_rows(eid, book, own.get(eid, []), market_map))
    big.sort(key=lambda r: -r["quantity"])
    write_csv(out / "big_levels.csv", big, BIG_LEVEL_FIELDS)
    logger.info("Big resting levels", extra={"books_read": books_read, "levels_ge_3000": len(big), "markets_with_one": len({r["exchange_id"] for r in big})})

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tournament": {"slug": slug, "id": tournament_id, "members": members, "initialBalance": tournament.get("initialBalance")},
        "leaderboard": lb_summary,
        "valuation": valuation,
        "tape": tape_summary,
        "big_levels": {"threshold_shares": BIG_LEVEL_SHARES, "books_read": books_read, "levels": len(big),
                       "by_side": stats.share_table((r["side"] for r in big), ["bid", "ask"]),
                       "by_price_bucket": stats.share_table((stats.price_bucket(r["price"]) for r in big), [label for _, label in stats.PRICE_BUCKETS])},
        # Descriptive script: two questions asked, nothing pre-registered to pass or fail.
        "hypotheses_tested": {"zero_sum": 1, "valuation_rule_candidates": len(valuation["candidates"])},
        "reads": reader.stats(),
    }
    write_json(out / "summary.json", summary)
    return summary


async def _amain(args: argparse.Namespace, reads_per_min: float, market_map: MarketMap) -> int:
    async with research_reader(SCRIPT, reads_per_min) as reader:
        await run(reader, reader.settings.tournament_slug, market_map, args.max_books, args.tape_exchanges)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: leaderboard, opening tape, big resting levels, valuation rule.")
    add_research_args(parser)
    parser.add_argument("--max-books", type=int, default=None, help="order books scanned for big levels (default: every mapped market with a quote)")
    parser.add_argument("--tape-exchanges", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None,
                        help=f"comma list of exchange ids for the opening tape (default: the {TAPE_MARKETS} mapped markets closest to 0.50)")
    parser.add_argument("--participants-estimate", type=int, default=DEFAULT_PARTICIPANTS_ESTIMATE, help="for the read estimate only")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    market_map = MarketMap.load()
    if args.max_books is None:
        args.max_books = len(market_map)
    if args.max_books < 0:
        parser.error("--max-books must be >= 0")
    tape_markets = len(args.tape_exchanges) if args.tape_exchanges else TAPE_MARKETS
    estimate = ReadEstimate(estimate_reads(args.participants_estimate, len(market_map), args.max_books, tape_markets), reads_per_min)
    if not confirm_fetch(estimate, args.yes):
        return 0
    return asyncio.run(_amain(args, reads_per_min, market_map))


if __name__ == "__main__":
    raise SystemExit(main())
