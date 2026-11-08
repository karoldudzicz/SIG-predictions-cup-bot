"""READ-ONLY research: hold-to-settlement candidates - buy YES at the ask or NO at 1 - bid
where the oracle probability of that side is above the price. EVERYTHING IS HYPOTHETICAL:
the oracle is an independent reference, not the truth, and nothing here is an order.

Settlement is taken as 2026-11-04T17:00Z; days to settlement are reported.

1. Prices: GET /exchanges/prices (bulk, 100 ids per read) for every mapped exchange. Oracle
   probability from data/market_map.json through the public Polymarket / Kalshi readers (no
   SIG budget). An oracle is used only when its PARTY matches SIG's (Polymarket label
   suffix "(D)" or a party word, Kalshi ticker suffix "-D"; never candidate names); every
   linked oracle needs a mid; two are averaged and flagged when they differ by > 0.03.
2. Per exchange and side: price (YES ask / 1 - YES bid), oracle p for that side, edge =
   (p - haircut) - price and yield = edge / price for haircuts 0, 0.02, 0.04. A candidate
   has a positive edge at haircut 0; candidates are ranked by yield at haircut 0.02.
3. Depth: the book (depth 50, our own resting orders taken out) of the top 60 candidates
   only; shares available at <= best price + 0.01 and their cost (walking the levels).
4. Capacity under the bot's configured limits (RiskLimits), on STARTING equity, net of our
   CURRENT positions (the newest saved position_review read; read live only if none is
   saved): per contract (max_fraction_per_contract, cost), per race (worst-case loss,
   risk.max_shares_within_race_cap), per cluster (max_fraction_per_cluster, cost;
   risk.build_clusters with the saved relationships) and the overall gross cap
   (max_gross_exposure_fraction). Capacity = min(depth, headroom). Headroom shared by
   candidates of one race / cluster / the gross cap is allocated greedily in rank order
   ("allocated"); "standalone" ignores the other candidates.
5. Totals per haircut: capacity-limited expected P&L at settlement (shares * (p - haircut)
   - cost), capital locked, the loss if the 5 largest allocated candidates all resolve
   against us, and the largest loss if everything allocated in one cluster does. Flags:
   we already hold the contract; we hold an opposite leg in the same race.

Reads through research.reader (any write raises; RESEARCH_READS_PER_MIN, default 10) behind
--yes: tournament, bulk prices, open orders, up to 60 books (+ positions if none saved).
-> data/edge_scan/settlement_carry_scan/<UTC date>/: raw.jsonl, candidates.csv, summary.md.

    python scripts/settlement_carry_scan.py [--yes] [--top-books 60]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from exchange_api import book_without_own, yes_terms  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from market_mapper import MARKET_MAP_PATH, MarketMap  # noqa: E402
from oracle_api import KalshiClient, OracleQuote, PolymarketClient  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, ResearchReader, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min, write_csv  # noqa: E402
from risk import RaceLeg, RiskLimits, build_clusters, max_shares_within_race_cap  # noqa: E402

logger = logging.getLogger("settlement_carry_scan")

SCRIPT = "settlement_carry_scan"
SETTLEMENT = datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc)
HAIRCUTS = (0.0, 0.02, 0.04)
RANK_HAIRCUT = 0.02
TOP_BOOKS = 60
BOOK_DEPTH = 50
DEPTH_SLIPPAGE = 0.01
ORACLE_MAX_GAP = 0.03
WORST_N = 5
PRICES_CHUNK = 100
ORDERS_PAGE = 200
ORACLE_RECORD_PATH = "oracle:quotes"
LABEL = "HYPOTHETICAL: the oracle is an independent reference, not the truth; expected P&L assumes it is right and that we hold to settlement"

FIELDS = ("rank", "exchange_id", "race", "party", "side", "days_to_settlement", "price", "oracle_p_yes", "oracle_p_side", "oracle_gap", "oracle_disagree",
          *(f"edge_{h:g}" for h in HAIRCUTS), *(f"yield_{h:g}" for h in HAIRCUTS), "book_read", "depth_shares", "depth_cost",
          "contract_room_cost", "race_room_shares", "cluster", "cluster_room_cost", "gross_room_cost", "standalone_shares",
          "allocated_shares", "allocated_cost", "avg_fill", *(f"ev_{h:g}" for h in HAIRCUTS), "held", "same_race_opposite", "current_leg")


# ---------------------------------------------------------------------- candidates (pure)


def oracle_links(raw_map: Mapping[str, Any], eid: str) -> dict[str, Any]:
    entry = (raw_map.get("entries") or {}).get(eid) or {}
    pm, ks = entry.get("polymarket") or {}, entry.get("kalshi") or {}
    return {"polymarket_token": pm.get("token_id"), "polymarket_label": pm.get("label"), "kalshi_ticker": ks.get("ticker")}


def evaluate(market_map: MarketMap, raw_map: Mapping[str, Any], prices: Mapping[str, Mapping[str, Any]], oracle_mids: Mapping[str, Mapping[str, float | None]],
             now: datetime) -> tuple[list[dict[str, Any]], Counter]:
    """Candidate sides (edge > 0 at haircut 0) and the count dropped per reason."""
    days = round((SETTLEMENT - now).total_seconds() / 86400, 2)
    out: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for m in market_map:
        eid = m.sig_exchange_id
        links = oracle_links(raw_map, eid)
        linked = {"polymarket": bool(links["polymarket_token"]), "kalshi": bool(links["kalshi_ticker"])}
        if not any(linked.values()):
            dropped["no oracle linked"] += 1
            continue
        parties = stats.oracle_parties(links["polymarket_label"], links["kalshi_ticker"])
        if stats.party_winner_mismatch(m.party, parties, linked):
            dropped["oracle party missing or different"] += 1
            continue
        combined = stats.combine_oracles(oracle_mids.get(eid) or {}, linked, ORACLE_MAX_GAP)
        if combined["p"] is None:
            dropped[f"oracle: {combined['reason']}"] += 1
            continue
        row = prices.get(eid) or {}
        buy = stats.side_buy_prices(None if row.get("bestBid") is None else float(row["bestBid"]), None if row.get("bestAsk") is None else float(row["bestAsk"]))
        found = False
        for side in ("yes", "no"):
            price = buy[side]
            p_side = combined["p"] if side == "yes" else 1.0 - combined["p"]
            edge0, _ = stats.carry_edge(p_side, price, 0.0)
            if edge0 is None or edge0 <= 0:
                continue
            found = True
            cand = {"exchange_id": eid, "race": m.race.key, "party": m.party, "side": side, "days_to_settlement": days, "price": price,
                    "oracle_p_yes": round(combined["p"], 4), "oracle_p_side": round(p_side, 4),
                    "oracle_gap": None if combined["gap"] is None else round(combined["gap"], 4), "oracle_disagree": combined["disagree"]}
            for h in HAIRCUTS:
                edge, yld = stats.carry_edge(p_side, price, h)
                cand[f"edge_{h:g}"], cand[f"yield_{h:g}"] = round(edge, 4), round(yld, 4)
            out.append(cand)
        if not found:
            dropped["no positive edge (or no price)"] += 1
    out.sort(key=lambda c: (-c[f"yield_{RANK_HAIRCUT:g}"], c["exchange_id"], c["side"]))
    for rank, c in enumerate(out, start=1):
        c["rank"] = rank
    return out, dropped


# ---------------------------------------------------------------------- capacity (pure)


@dataclass
class Book:
    """Current exposure, updated as candidates are allocated."""

    equity: float
    limits: RiskLimits
    clusters: Mapping[str, str]  # exchange -> cluster key
    race_legs: dict[str, list[RaceLeg]] = field(default_factory=lambda: defaultdict(list))
    contract_cost: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cluster_cost: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    gross_cost: float = 0.0

    def add(self, eid: str, race: str | None, party: str | None, side: str, shares: float, cost: float) -> None:
        if race and party:
            self.race_legs[race].append(RaceLeg(party, side, shares, cost))
        self.contract_cost[eid] += cost
        self.cluster_cost[self.clusters.get(eid, f"exchange:{eid}")] += cost
        self.gross_cost += cost


def book_from_positions(positions: Sequence[Mapping[str, Any]], market_map: MarketMap, clusters: Mapping[str, str], equity: float, limits: RiskLimits) -> Book:
    book = Book(equity, limits, clusters)
    for p in positions:
        q = float(p.get("quantity") or 0)
        if not q:
            continue
        eid = str(p.get("exchangeId"))
        cost = abs(float(p["costBasis"])) if p.get("costBasis") is not None else abs(q) * float(p.get("avgCost") or 0)
        m = market_map.get(eid)
        book.add(eid, m.race.key if m else None, m.party if m else None, "yes" if q > 0 else "no", abs(q), cost)
    return book


def capacity(cand: Mapping[str, Any], levels: Sequence[tuple[float, float]], book: Book, market_map: MarketMap) -> dict[str, Any]:
    """Shares buyable now under every cap given `book`, and their cost (levels walked to
    best + DEPTH_SLIPPAGE)."""
    eid, side = cand["exchange_id"], cand["side"]
    if not levels:
        return {"depth_shares": 0.0, "depth_cost": 0.0, "shares": 0.0, "cost": 0.0, "contract_room_cost": None, "race_room_shares": None,
                "cluster_room_cost": None, "gross_room_cost": None}
    max_price = levels[0][0] + DEPTH_SLIPPAGE
    depth_shares, depth_cost = stats.walk_levels(levels, max_price)
    lim, equity = book.limits, book.equity
    contract_room = max(0.0, lim.max_fraction_per_contract * equity - book.contract_cost.get(eid, 0.0))
    cluster_key = book.clusters.get(eid, f"exchange:{eid}")
    cluster_room = max(0.0, lim.max_fraction_per_cluster * equity - book.cluster_cost.get(cluster_key, 0.0))
    gross_room = max(0.0, lim.max_gross_exposure_fraction * equity - book.gross_cost)
    budget = min(contract_room, cluster_room, gross_room)
    shares, _ = stats.walk_levels(levels, max_price, max_cost=budget)
    m = market_map.get(eid)
    race_room = None
    if m is not None and shares > 0:
        avg = depth_cost / depth_shares if depth_shares else levels[0][0]
        parties = [x.party for x in market_map.by_race().get(m.race.key, [])]
        race_room = float(max_shares_within_race_cap(parties, book.race_legs.get(m.race.key, []), m.party, side, avg, int(shares),
                                                     lim.max_race_worst_loss_fraction * equity))
        shares = min(shares, race_room)
    shares, cost = stats.walk_levels(levels, max_price, max_shares=shares)
    return {"depth_shares": depth_shares, "depth_cost": depth_cost, "shares": shares, "cost": cost, "contract_room_cost": round(contract_room, 2),
            "race_room_shares": race_room, "cluster_room_cost": round(cluster_room, 2), "gross_room_cost": round(gross_room, 2), "cluster": cluster_key}


def allocate(cands: Sequence[dict[str, Any]], levels: Mapping[tuple[str, str], Sequence[tuple[float, float]]], positions: Sequence[Mapping[str, Any]],
             market_map: MarketMap, clusters: Mapping[str, str], equity: float, limits: RiskLimits) -> None:
    """Fill capacity columns in place: standalone (only current positions) and allocated
    (greedy in rank order: each candidate uses up the shared headroom)."""
    shared = book_from_positions(positions, market_map, clusters, equity, limits)
    held = {str(p.get("exchangeId")): ("yes" if float(p.get("quantity") or 0) > 0 else "no") for p in positions if float(p.get("quantity") or 0)}
    by_race = market_map.by_race()
    for c in cands:
        key = (c["exchange_id"], c["side"])
        lv = levels.get(key)
        c["book_read"] = lv is not None
        m = market_map.get(c["exchange_id"])
        race_parties = [x.party for x in by_race.get(c["race"], [])]
        direction = stats.leg_direction(c["party"], c["side"], race_parties)
        c["held"] = c["exchange_id"] in held
        c["current_leg"] = held.get(c["exchange_id"])
        c["same_race_opposite"] = any(
            stats.leg_direction(x.party, held[x.sig_exchange_id], race_parties) not in (None, direction)
            for x in by_race.get(c["race"], []) if x.sig_exchange_id in held)
        if lv is None:
            continue
        alone = capacity(c, lv, book_from_positions(positions, market_map, clusters, equity, limits), market_map)
        now = capacity(c, lv, shared, market_map)
        c.update({"depth_shares": round(now["depth_shares"], 2), "depth_cost": round(now["depth_cost"], 2), "contract_room_cost": now["contract_room_cost"],
                  "race_room_shares": now["race_room_shares"], "cluster": now.get("cluster"), "cluster_room_cost": now["cluster_room_cost"],
                  "gross_room_cost": now["gross_room_cost"], "standalone_shares": round(alone["shares"], 2),
                  "allocated_shares": round(now["shares"], 2), "allocated_cost": round(now["cost"], 2),
                  "avg_fill": round(now["cost"] / now["shares"], 4) if now["shares"] else None})
        for h in HAIRCUTS:
            c[f"ev_{h:g}"] = round(now["shares"] * (c["oracle_p_side"] - h) - now["cost"], 2)
        if now["shares"] > 0 and m is not None:
            shared.add(c["exchange_id"], m.race.key, m.party, c["side"], now["shares"], now["cost"])


def totals(cands: Sequence[Mapping[str, Any]], worst_n: int = WORST_N) -> dict[str, Any]:
    alloc = [c for c in cands if c.get("allocated_shares")]
    largest = sorted(alloc, key=lambda c: -c["allocated_cost"])[:worst_n]
    by_cluster: dict[str, float] = defaultdict(float)
    for c in alloc:
        by_cluster[c.get("cluster") or c["race"]] += c["allocated_cost"]
    worst_cluster = max(by_cluster.items(), key=lambda kv: kv[1]) if by_cluster else (None, 0.0)
    return {"candidates": len(cands), "books_read": sum(1 for c in cands if c.get("book_read")), "allocated": len(alloc),
            **{f"ev_{h:g}": round(math.fsum(c[f"ev_{h:g}"] for c in alloc), 2) for h in HAIRCUTS},
            "capital_locked": round(math.fsum(c["allocated_cost"] for c in alloc), 2),
            "largest": [f"{c['exchange_id']} {c['side']}" for c in largest], "loss_if_largest_all_lose": round(-math.fsum(c["allocated_cost"] for c in largest), 2),
            "worst_cluster": worst_cluster[0], "loss_if_worst_cluster_all_loses": round(-worst_cluster[1], 2),
            "held_flags": sum(1 for c in cands if c.get("held")), "opposite_leg_flags": sum(1 for c in cands if c.get("same_race_opposite")),
            "oracle_disagree_flags": sum(1 for c in cands if c.get("oracle_disagree"))}


# ---------------------------------------------------------------------- report


def summary_markdown(cands: Sequence[Mapping[str, Any]], tot: Mapping[str, Any], dropped: Mapping[str, int], sources: Mapping[str, Any]) -> str:
    top = [c for c in cands if c.get("book_read")][:30]
    lines = ["# Settlement carry scan (HYPOTHETICAL)", "", f"_{LABEL}._", "",
             f"- Settlement assumed {SETTLEMENT.isoformat()}; prices read {sources['read_at']}; current positions: {sources['positions_from']}; "
             f"caps on starting equity {sources['equity']} (contract {sources['caps']['contract']}, race worst loss {sources['caps']['race']}, "
             f"cluster {sources['caps']['cluster']}, gross {sources['caps']['gross']}).",
             f"- {tot['candidates']} candidate sides with a positive edge at haircut 0; books read for {tot['books_read']}; capacity allocated to {tot['allocated']}.",
             f"- Dropped exchanges: {dict(dropped)}.",
             f"- Capacity-limited expected P&L at settlement: {tot['ev_0']} (haircut 0), {tot['ev_0.02']} (0.02), {tot['ev_0.04']} (0.04) SUSQ; capital locked {tot['capital_locked']}.",
             f"- If the {len(tot['largest'])} largest ({', '.join(tot['largest'])}) all resolve against us: {tot['loss_if_largest_all_lose']} SUSQ; "
             f"if everything in the largest cluster ({tot['worst_cluster']}) does: {tot['loss_if_worst_cluster_all_loses']} SUSQ.",
             f"- Flags: already held {tot['held_flags']}, opposite leg held in the same race {tot['opposite_leg_flags']}, oracles disagree > {ORACLE_MAX_GAP} {tot['oracle_disagree_flags']}.",
             "", "| rank | exchange | race | party | side | price | oracle p | edge 0 / 0.02 / 0.04 | yield 0.02 | depth | allocated | cost | EV 0.02 | flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
             *[f"| {c['rank']} | {c['exchange_id']} | {c['race']} | {c['party']} | {c['side']} | {c['price']} | {c['oracle_p_side']} | "
               f"{c['edge_0']} / {c['edge_0.02']} / {c['edge_0.04']} | {c['yield_0.02']} | {c.get('depth_shares')} | {c.get('allocated_shares')} | {c.get('allocated_cost')} | "
               f"{c.get('ev_0.02')} | {', '.join(f for f, on in (('held', c.get('held')), ('opposite leg', c.get('same_race_opposite')), ('oracles disagree', c.get('oracle_disagree'))) if on)} |"
               for c in top], ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------- reads


def saved_review(root: Path | None = None) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]], str | None]:
    """(positions, relationships, source) from the newest saved position_review raw.jsonl."""
    runs = sorted((root or OUT_ROOT / "position_review").glob("*/raw.jsonl"))
    if not runs:
        return None, [], None
    positions, relationships, at = None, [], None
    for rec in read_jsonl(runs[-1]):
        if "body" not in rec:
            continue
        if rec.get("path") == "/portfolio/positions":
            positions, at = list((rec["body"] or {}).get("positions") or []), rec.get("ts")
        elif rec.get("path") == "/relationships":
            relationships = (relationships if (rec.get("params") or {}).get("cursor") else []) + list((rec["body"] or {}).get("data") or [])
    return positions, relationships, None if positions is None else f"saved {runs[-1]} ({at})"


def own_orders(open_orders: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[str, float, float]]]:
    own: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for o in open_orders:
        if o.get("open") is False or not o.get("exchangeId"):
            continue
        side, price = yes_terms(o.get("side"), o.get("action"), o.get("priceLimit"))
        remaining = o.get("remainingQuantity")
        own[str(o["exchangeId"])].append((side, price, abs(float(remaining if remaining is not None else o.get("quantity") or 0))))
    return own


async def run(reader: ResearchReader, slug: str, market_map: MarketMap, raw_map: Mapping[str, Any], polymarket: Any, kalshi: Any, top_books: int,
              limits: RiskLimits | None = None) -> dict[str, Any]:
    limits = limits or RiskLimits()
    tournament = await reader.get(f"/tournaments/{slug}")
    tid, equity = tournament.get("id"), float(tournament.get("initialBalance") or 0.0)
    ids = sorted(m.sig_exchange_id for m in market_map)
    prices: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), PRICES_CHUNK):
        body = await reader.get("/exchanges/prices", {"ids": ",".join(ids[start : start + PRICES_CHUNK]), "tournamentId": tid})
        prices.update({str(r["exchangeId"]): r for r in body.get("data") or []})
    read_at = datetime.now(timezone.utc)

    links = {m.sig_exchange_id: oracle_links(raw_map, m.sig_exchange_id) for m in market_map}
    pm_quotes, ks_quotes = await asyncio.gather(polymarket.get_quotes(sorted({l["polymarket_token"] for l in links.values() if l["polymarket_token"]})),
                                                kalshi.get_quotes(sorted({l["kalshi_ticker"] for l in links.values() if l["kalshi_ticker"]})))

    def mid(q: OracleQuote | None) -> float | None:
        return None if q is None else q.mid

    oracle_mids = {eid: {"polymarket": mid(pm_quotes.get(l["polymarket_token"] or "")), "kalshi": mid(ks_quotes.get(l["kalshi_ticker"] or ""))} for eid, l in links.items()}
    reader.raw_log.append({"method": "GET", "path": ORACLE_RECORD_PATH, "params": {}, "body": {"mids": oracle_mids}})  # for later offline use
    cands, dropped = evaluate(market_map, raw_map, prices, oracle_mids, read_at)

    orders: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = await reader.get("/orders", {"status": "open", "limit": ORDERS_PAGE, "cursor": cursor})
        orders.extend(page.get("data") or [])
        pagination = page.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            break
    mine = own_orders(orders)
    levels: dict[tuple[str, str], list[tuple[float, float]]] = {}
    books_by_eid: dict[str, dict[str, Any]] = {}
    for c in cands[:top_books]:
        eid = c["exchange_id"]
        if eid not in books_by_eid:
            raw_book = await reader.get(f"/exchanges/{eid}/orderbook", {"depth": BOOK_DEPTH, "tournamentId": tid})
            books_by_eid[eid] = book_without_own(dict(raw_book), mine.get(eid, []))
        levels[(eid, c["side"])] = stats.side_levels(books_by_eid[eid], c["side"])

    positions, relationships, positions_from = saved_review()
    if positions is None:
        positions = (await reader.get("/portfolio/positions")).get("positions") or []
        positions_from = "read now (no saved position_review)"
    clusters = {eid: c.key for c in build_clusters(market_map, relationships) for eid in c.exchange_ids}
    allocate(cands, levels, positions, market_map, clusters, equity, limits)
    tot = totals(cands)
    sources = {"read_at": read_at.isoformat(), "positions_from": positions_from, "equity": equity,
               "caps": {"contract": limits.max_fraction_per_contract, "race": limits.max_race_worst_loss_fraction,
                        "cluster": limits.max_fraction_per_cluster, "gross": limits.max_gross_exposure_fraction}}
    return {"candidates": cands, "totals": tot, "dropped": dict(dropped), "sources": sources}


async def _amain(reads_per_min: float, market_map: MarketMap, raw_map: Mapping[str, Any], top_books: int) -> int:
    async with research_reader(SCRIPT, reads_per_min) as reader, PolymarketClient() as polymarket, KalshiClient() as kalshi:
        result = await run(reader, reader.settings.tournament_slug, market_map, raw_map, polymarket, kalshi, top_books)
        out = reader.out_dir
    write_csv(out / "candidates.csv", result["candidates"], FIELDS)
    (out / "summary.md").write_text(summary_markdown(result["candidates"], result["totals"], result["dropped"], result["sources"]), encoding="utf-8")
    logger.info("Settlement carry scan done (HYPOTHETICAL)", extra={"out_dir": str(out), **result["totals"], "dropped": result["dropped"], "label": LABEL})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: hold-to-settlement candidates vs the oracle, with depth and capacity (HYPOTHETICAL).")
    add_research_args(parser)
    parser.add_argument("--top-books", type=int, default=TOP_BOOKS, help=f"order books read, best candidates first (default {TOP_BOOKS})")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.top_books <= 200:
        parser.error("--top-books must be in [0, 200]")
    market_map = MarketMap.load()
    raw_map = json.loads(MARKET_MAP_PATH.read_text(encoding="utf-8"))
    saved = sorted((OUT_ROOT / "position_review").glob("*/raw.jsonl"))
    reads = 1 + math.ceil(len(market_map) / PRICES_CHUNK) + 1 + args.top_books + (0 if saved else 1)
    if not confirm_fetch(ReadEstimate(reads, reads_per_min), args.yes):
        return 0
    return asyncio.run(_amain(reads_per_min, market_map, raw_map, args.top_books))


if __name__ == "__main__":
    raise SystemExit(main())
