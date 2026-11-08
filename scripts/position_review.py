"""READ-ONLY research: a close look at the worst replay races and at every position we hold.

Everything priced here is MARKED AT ONE CURRENT MID; oracle probabilities are an
independent reference, not the truth.

Scope: every mapped exchange of the races in --races (default: the 8 races with the
largest loss in the dry-run replay - senate:RI, senate:FL, senate:DE, senate:OK, senate:KS,
senate:SC, governor:RI, governor:MA; --from-replay re-reads them from the newest
data/edge_scan/dry_run_replay/*/by_race.csv) plus every exchange with a position now.

Per exchange:
  - SIG title and rules: the API has no free-text rules field; the closest is the market's
    resolution tree (GET /markets/{id}/nodes): every contract node's title, contract
    type, settlement date and contract_details (as stored), printed in full. The
    resolutionType inside contract_details (e.g. "Party Winner") is shown on its own;
  - mapped oracle from data/market_map.json: Polymarket event slug / label / token,
    Kalshi event ticker / label / ticker, the prices seen when the map was built, and the
    oracle mid now (public Polymarket / Kalshi reads, not SIG's budget);
  - SIG bid, ask, mid, top-of-book sizes and spread (book embedded in GET
    /tournaments/{slug}/markets/{id});
  - our position: shares, side, the API's average cost and cost basis, the order-journal
    average entry (limit-weighted, the journal keeps no fill price), notional, and
    unrealized P&L MARKED AT MID and at the bid (what a sale would get);
  - wording flag: ONLY the party (D / R / I) is compared - the party named in the SIG
    contract title (the mapped party if the title names none or several) against the
    party in the oracle labels (Polymarket label suffix "(D)" or a party word, Kalshi
    ticker suffix "-D"). Candidate names are never compared. Flagged only on a real
    mismatch; an oracle label without a party (a bare candidate name) is "unknown";
  - shares of STARTING equity per contract, race (worst case, risk.race_worst_case_loss) and
    cluster (risk.build_clusters), next to the configured caps (applied by the bot to its
    current bankroll).

Expected P&L at settlement, HYPOTHETICAL and oracle-based (expected_pnl.csv; oracle error
is NOT modelled - the oracle probability is taken as if it were right): per held position
expected_pnl = shares * (p_side - avg_cost), p_side = p for YES and 1 - p for NO, p = the
oracle mid now; a second column with p_side lowered by a 0.02 haircut; break_even_p = the
average cost, in the HELD side's terms (break_even_refers_to says which side), also as a
YES probability. Per race all legs are summed - YES-D and NO-R of one race are the same
bet - with the race's direction and settlement P&L per winner scenario
(risk.race_scenario_pnl; worst case = the existing risk.race_worst_case_loss definition).
Grand total, and the stress loss if the 5 races with the LARGEST worst-case loss all
resolve against us.

Swaps, HYPOTHETICAL (swaps.csv; no order is placed or simulated): for each held position,
the hold-to-settlement yield of the money a sale at the CURRENT BID would free -
((p_side - 0.02) - exit) / exit, exit = YES bid for YES, 1 - YES ask for NO, the sale capped at
the top-of-book size - against the candidates of the newest
data/edge_scan/settlement_carry_scan/*/candidates.csv (or --candidates): yield = EV at
haircut 0.02 / allocated cost, capacity = allocated cost. Capital and capacity are matched
greedily, best yield improvement first, each used once (no double-counting); a position
never swaps into its own exchange. Net of spread: exits at the bid, entries at the
candidate's average fill. The 10 best swaps and the total expected gain.

Reads: --yes reads through research.reader (any write raises; RESEARCH_READS_PER_MIN,
default 10): tournament, positions, relationships, 2 per market in scope, plus the public
oracle quotes (saved to raw.jsonl as "oracle:mids"). --offline reads NOTHING from SIG: the
newest saved data/edge_scan/position_review/*/raw.jsonl (or --raw) is replayed; oracle
mids come from its "oracle:mids" record, else from that run's position_review.csv, or
fresh from the public oracles with --fresh-oracle. With neither flag only the read
estimate is logged.
-> data/edge_scan/position_review/<UTC date>/: raw.jsonl, position_review.csv,
   expected_pnl.csv, swaps.csv, summary.md.

    python scripts/position_review.py --yes [--races senate:RI,governor:MA] [--from-replay]
    python scripts/position_review.py --offline [--raw PATH] [--fresh-oracle] [--candidates PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import DATA_DIR, MARKET_MAP_PATH, MarketMap  # noqa: E402
from oracle_api import KalshiClient, OracleQuote, PolymarketClient  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, ResearchReader, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min, run_dir, write_csv  # noqa: E402
from risk import RaceLeg, RiskLimits, build_clusters, race_scenario_pnl, race_worst_case_loss  # noqa: E402

logger = logging.getLogger("position_review")

SCRIPT = "position_review"
DEFAULT_RACES = ("senate:RI", "senate:FL", "senate:DE", "senate:OK", "senate:KS", "senate:SC", "governor:RI", "governor:MA")
REPLAY_TOP = 8
HELD_ESTIMATE = 40  # positions assumed for the read estimate only
RELATIONSHIPS_PAGE = 100
HAIRCUT = 0.02
WORST_RACES = 5
ORACLE_RECORD_PATH = "oracle:mids"  # pseudo path of the saved oracle mids in raw.jsonl
ORDER_JOURNAL_PATH = DATA_DIR / "order_journal.jsonl"
MARK_LABEL = "marked at one current mid; oracle probabilities are an independent reference, not the truth"
EXPECTED_LABEL = "HYPOTHETICAL, oracle-based: expected P&L at settlement takes the oracle probability as right; oracle error is not modelled"
SWAP_LABEL = "HYPOTHETICAL: no order is placed or simulated; yields net of spread (exit at the bid, entry at the candidate's average fill)"
SWAP_HAIRCUT = 0.02
TOP_SWAPS = 10

FIELDS = ("exchange_id", "market_id", "in_scope_because", "race", "party", "sig_title", "resolution_type", "sig_rules", "oracle_polymarket", "oracle_kalshi",
          "oracle_map_prices", "oracle_mid_now", "sig_party", "oracle_party", "party_check", "flag", "flag_reasons", "bid", "ask", "mid", "bid_size", "ask_size", "spread",
          "shares", "side", "direction", "api_avg_cost", "cost_basis", "journal_avg_entry", "journal_filled_shares", "notional_at_mid",
          "unrealized_at_mid", "unrealized_at_bid", "contract_cost_share", "contract_mid_share", "contract_cap",
          "race_cost_share", "race_mid_share", "race_worst_loss_share", "race_cap", "cluster", "cluster_cost_share", "cluster_cap")
EXPECTED_FIELDS = ("level", "exchange_id", "race", "party", "side", "direction", "legs", "shares", "avg_cost", "cost", "break_even_p", "break_even_refers_to",
                   "break_even_yes_p", "oracle_p_yes", "p_side", "expected_pnl", "expected_pnl_haircut", "scenario_pnl", "worst_scenario", "worst_pnl")
SWAP_FIELDS = ("rank", "hold", "hold_race", "hold_yield", "candidate", "candidate_race", "candidate_yield", "amount", "yield_gain", "gain")


# ---------------------------------------------------------------------- scope


def races_from_replay(by_race_csv: Path, top: int = REPLAY_TOP) -> list[str]:
    """The `top` dry-run races with the most negative P&L in a dry_run_replay by_race.csv."""
    with by_race_csv.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("source") == "dry_run" and r.get("pnl") not in (None, "")]
    rows.sort(key=lambda r: float(r["pnl"]))
    return [r["group"] for r in rows[:top] if float(r["pnl"]) < 0]


def scope(market_map: MarketMap, races: Sequence[str], positions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """exchange_id -> {"market_id", "why"} for the races' mapped exchanges and every held one."""
    out: dict[str, dict[str, Any]] = {}
    wanted = set(races)
    for m in market_map:
        if m.race.key in wanted:
            out[m.sig_exchange_id] = {"market_id": m.sig_market_id, "why": "replay race"}
    for p in positions:
        if float(p.get("quantity") or 0) == 0:
            continue
        eid = str(p.get("exchangeId"))
        entry = out.setdefault(eid, {"market_id": str(p.get("marketId") or (market_map.get(eid).sig_market_id if market_map.get(eid) else "")), "why": "held"})
        if entry["why"] == "replay race":
            entry["why"] = "replay race + held"
    return out


# ---------------------------------------------------------------------- rules, oracle, book


def _walk(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out = [node]
    for child in node.get("children") or []:
        out += _walk(child)
    return out


def rules_text(nodes_body: Mapping[str, Any] | None) -> str:
    """Every node of the resolution tree as text: operators, contract titles, types,
    settlement dates and contract_details (compact JSON, as stored)."""
    if not nodes_body:
        return ""
    parts: list[str] = []

    def walk(node: Mapping[str, Any], depth: int) -> None:
        if node.get("node_type", node.get("nodeType")) == "operator" or node.get("operator"):
            parts.append(f"{'  ' * depth}{node.get('operator')}")
        else:
            details = node.get("contract_details", node.get("contractDetails"))
            parts.append(f"{'  ' * depth}[{node.get('contract_type') or node.get('contractType')}] {node.get('title')} "
                         f"(settles {node.get('settlement_date') or node.get('settlementDate')}; settled_with {node.get('settled_with')}) "
                         f"details={json.dumps(details, sort_keys=True, separators=(',', ':')) if details is not None else 'null'}")
        for child in node.get("children") or []:
            walk(child, depth + 1)

    walk(nodes_body.get("root") or {}, 0)
    return " | ".join(p.strip() for p in parts)


def resolution_type(nodes_body: Mapping[str, Any] | None) -> str | None:
    """The first resolutionType found in the contract nodes' contract_details."""
    if not nodes_body:
        return None
    for node in _walk(nodes_body.get("root") or {}):
        details = node.get("contract_details", node.get("contractDetails"))
        if isinstance(details, Mapping):
            value = details.get("resolutionType", details.get("resolution_type"))
            if value:
                return str(value)
    return None


def is_party_winner(rtype: str | None) -> bool:
    return bool(rtype) and re.sub(r"[^a-z]", "", rtype.lower()) == "partywinner"


def oracle_entry(raw_map: Mapping[str, Any], exchange_id: str) -> dict[str, Any]:
    entry = (raw_map.get("entries") or {}).get(exchange_id) or {}
    pm, ks = entry.get("polymarket") or {}, entry.get("kalshi") or {}
    return {"polymarket": None if not pm else f"{pm.get('event_slug')} / {pm.get('label')} / {pm.get('token_id')}",
            "kalshi": None if not ks else f"{ks.get('event_ticker')} / {ks.get('label')} / {ks.get('ticker')}",
            "polymarket_token": pm.get("token_id"), "polymarket_label": pm.get("label"), "kalshi_ticker": ks.get("ticker"),
            "oracle_title_text": " ".join(str(x) for x in (pm.get("event_slug"), pm.get("label"), ks.get("event_ticker"), ks.get("label")) if x),
            "map_prices": entry.get("check_prices")}


def oracle_mid(pm: OracleQuote | None, ks: OracleQuote | None) -> float | None:
    mids = [q.mid for q in (pm, ks) if q is not None and q.mid is not None]
    return round(sum(mids) / len(mids), 4) if mids else None


def party_flag(title: str | None, oracle: Mapping[str, Any], mapped_party: str | None) -> dict[str, Any]:
    """PARTY only: the SIG title's party vs the oracle labels' party; never candidate names."""
    sig_party = stats.sig_title_party(title, mapped_party)
    linked = {"polymarket": oracle.get("polymarket_label") if oracle.get("polymarket") else None,
              "kalshi": oracle.get("kalshi_ticker") if oracle.get("kalshi") else None}
    parties = stats.oracle_parties(linked["polymarket"], linked["kalshi"])
    status, reasons = stats.party_only_check(sig_party, parties) if any(linked.values()) else ("no oracle", [])
    return {"sig_party": sig_party, "oracle_party": ";".join(f"{k}:{v}" for k, v in parties.items() if v) or None, "party_check": status, "reasons": reasons}


def book_of(market_body: Mapping[str, Any], exchange_id: str) -> dict[str, Any]:
    """Top of the embedded YES book of one exchange of a market detail."""
    ex = next((e for e in market_body.get("exchanges") or [] if str(e.get("id")) == exchange_id), {}) or {}
    bids, asks = ex.get("bids") or [], ex.get("asks") or []
    bid = ex.get("bestBid", bids[0]["price"] if bids else None)
    ask = ex.get("bestAsk", asks[0]["price"] if asks else None)
    bid = None if bid is None else float(bid)
    ask = None if ask is None else float(ask)
    return {"bid": bid, "ask": ask, "mid": None if bid is None or ask is None else round((bid + ask) / 2, 4),
            "bid_size": float(bids[0]["quantity"]) if bids else None, "ask_size": float(asks[0]["quantity"]) if asks else None,
            "spread": None if bid is None or ask is None else round(ask - bid, 4)}


# ---------------------------------------------------------------------- position maths


def journal_avg_entry(entries: Sequence[Mapping[str, Any]], exchange_id: str, side: str) -> tuple[float | None, float]:
    """(limit-weighted average entry, shares) over accepted journal orders on `exchange_id`
    whose CANONICAL order bought `side` and traded at placement. A lower bound on the
    quality of the fills: taker fills are at the limit or better."""
    latest: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e.get("key") is not None:
            latest[e["key"]] = {**latest.get(e["key"], {}), **e}
    shares = cost = 0.0
    for o in latest.values():
        result = o.get("result") if isinstance(o.get("result"), dict) else {}
        if o.get("status") != "accepted" or str((o.get("payload") or {}).get("exchangeId")) != exchange_id:
            continue
        if result.get("action") != "buy" or result.get("side") != side or result.get("price") is None:
            continue
        traded = abs(float(result.get("quantityTraded") or 0))
        shares += traded
        cost += traded * float(result["price"])
    return (round(cost / shares, 4) if shares else None), shares


def position_values(quantity: float, cost_basis: float | None, api_avg: float | None, book: Mapping[str, Any]) -> dict[str, Any]:
    """Side, shares, average entry and values in the held side's terms; unrealized P&L
    marked at the mid and at the bid (YES: YES bid; NO: 1 - YES ask)."""
    if not quantity:
        return {"shares": 0.0, "side": None, "avg": None, "notional_at_mid": None, "unrealized_at_mid": None, "unrealized_at_bid": None}
    side = "yes" if quantity > 0 else "no"
    shares = abs(quantity)
    avg = round(abs(cost_basis) / shares, 4) if cost_basis is not None else api_avg
    mid, bid, ask = book.get("mid"), book.get("bid"), book.get("ask")
    mid_side = None if mid is None else (mid if side == "yes" else 1 - mid)
    bid_side = (bid if side == "yes" else (None if ask is None else 1 - ask))
    return {"shares": shares, "side": side, "avg": avg,
            "notional_at_mid": None if mid_side is None else round(shares * mid_side, 2),
            "unrealized_at_mid": None if mid_side is None or avg is None else round(shares * (mid_side - avg), 2),
            "unrealized_at_bid": None if bid_side is None or avg is None else round(shares * (bid_side - avg), 2)}


def race_legs(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[RaceLeg]]:
    legs: dict[str, list[RaceLeg]] = defaultdict(list)
    for r in rows:
        if r.get("shares") and r.get("race"):
            legs[r["race"]].append(RaceLeg(r["party"], r["side"], r["shares"], (r["avg"] or 0.0) * r["shares"]))
    return legs


def race_exposure(rows: Sequence[Mapping[str, Any]], market_map: MarketMap) -> dict[str, dict[str, float]]:
    """race -> cost, value at mid and worst-case loss (risk.race_worst_case_loss)."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"cost": 0.0, "mid_value": 0.0})
    for r in rows:
        if r.get("shares") and r.get("race"):
            totals[r["race"]]["cost"] += (r["avg"] or 0.0) * r["shares"]
            totals[r["race"]]["mid_value"] += r.get("notional_at_mid") or 0.0
    by_race = market_map.by_race()
    for race, legs in race_legs(rows).items():
        totals[race]["worst_loss"] = race_worst_case_loss([m.party for m in by_race.get(race, [])], legs)
    return dict(totals)


def _share(value: float | None, equity: float) -> float | None:
    return None if value is None or not equity else round(value / equity, 4)


# ---------------------------------------------------------------------- settlement carry (HYPOTHETICAL)


def expected_pnl_rows(rows: Sequence[Mapping[str, Any]], haircut: float = HAIRCUT) -> list[dict[str, Any]]:
    """Per held position (level "position"): expected settlement P&L at the oracle probability
    and with p_side lowered by `haircut`; break-even in the held side's terms."""
    out = []
    for r in rows:
        if not r.get("shares"):
            continue
        p_yes = r.get("oracle_mid_now")
        p_side = None if p_yes is None else (p_yes if r["side"] == "yes" else 1 - p_yes)
        expected = stats.carry_pnl(r["shares"], p_side, r["avg"])
        hair = stats.carry_pnl(r["shares"], None if p_side is None else max(0.0, p_side - haircut), r["avg"])
        out.append({"level": "position", "exchange_id": r["exchange_id"], "race": r["race"], "party": r["party"], "side": r["side"], "direction": r.get("direction"),
                    "shares": r["shares"], "avg_cost": r["avg"], "cost": None if r["avg"] is None else round(r["avg"] * r["shares"], 2),
                    "break_even_p": r["avg"], "break_even_refers_to": f"{(r['side'] or '').upper()} price / probability",
                    "break_even_yes_p": None if r["avg"] is None else round(r["avg"] if r["side"] == "yes" else 1 - r["avg"], 4),
                    "oracle_p_yes": p_yes, "p_side": None if p_side is None else round(p_side, 4),
                    "expected_pnl": None if expected is None else round(expected, 2), "expected_pnl_haircut": None if hair is None else round(hair, 2)})
    return out


def race_expected(rows: Sequence[Mapping[str, Any]], positions: Sequence[Mapping[str, Any]], market_map: MarketMap) -> list[dict[str, Any]]:
    """Per race (level "race"), all legs summed - YES-D and NO-R of one race are the same bet:
    direction, cost, expected P&L, settlement P&L per winner scenario, worst scenario.
    Worst first."""
    by_race = market_map.by_race()
    pos_by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for e in positions:
        if e["race"]:
            pos_by[e["race"]].append(e)
    out = []
    for race, legs in race_legs(rows).items():
        held = [r for r in rows if r.get("race") == race and r.get("shares")]
        scenario = race_scenario_pnl([m.party for m in by_race.get(race, [])], legs)
        worst = min(scenario, key=scenario.get)
        es = pos_by.get(race, [])
        expected = None if any(e["expected_pnl"] is None for e in es) else round(math.fsum(e["expected_pnl"] for e in es), 2)
        hair = None if any(e["expected_pnl_haircut"] is None for e in es) else round(math.fsum(e["expected_pnl_haircut"] for e in es), 2)
        out.append({"level": "race", "race": race, "direction": stats.race_direction(r.get("direction") for r in held),
                    "legs": "; ".join(f"{r['party']} {r['side']} {r['shares']:g} @ {r['avg']}" for r in held),
                    "cost": round(math.fsum(l.cost for l in legs), 2), "expected_pnl": expected, "expected_pnl_haircut": hair,
                    "scenario_pnl": json.dumps({k: round(v, 2) for k, v in scenario.items()}), "worst_scenario": worst, "worst_pnl": round(scenario[worst], 2)})
    return sorted(out, key=lambda r: r["worst_pnl"])


def expected_totals(positions: Sequence[Mapping[str, Any]], races: Sequence[Mapping[str, Any]], worst_n: int = WORST_RACES) -> dict[str, Any]:
    """Grand total, and the stress loss if the `worst_n` races with the LARGEST worst-case loss
    all resolve against us (worst case as in risk.race_worst_case_loss)."""
    worst = sorted(races, key=lambda r: r["worst_pnl"])[:worst_n]
    return {"level": "total", "positions": len(positions), "unpriced_positions": sum(1 for e in positions if e["expected_pnl"] is None),
            "expected_pnl": round(math.fsum(e["expected_pnl"] for e in positions if e["expected_pnl"] is not None), 2),
            "expected_pnl_haircut": round(math.fsum(e["expected_pnl_haircut"] for e in positions if e["expected_pnl_haircut"] is not None), 2),
            "cost": round(math.fsum(e["cost"] for e in positions if e["cost"] is not None), 2),
            "stress_races": [r["race"] for r in worst], "stress_loss": round(math.fsum(r["worst_pnl"] for r in worst), 2),
            "worst_single_race": None if not races else {"race": races[0]["race"], "scenario": races[0]["worst_scenario"], "pnl": races[0]["worst_pnl"]}}


# ---------------------------------------------------------------------- swaps (HYPOTHETICAL)


def load_candidates(path: Path | None) -> list[dict[str, Any]]:
    """Allocated candidates of a settlement_carry_scan candidates.csv: id, race, capacity
    (allocated cost) and yield = EV at SWAP_HAIRCUT / allocated cost (net of depth)."""
    if path is None or not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cost, ev = r.get("allocated_cost"), r.get(f"ev_{SWAP_HAIRCUT:g}")
            if cost in (None, "") or ev in (None, "") or float(cost) <= 0:
                continue
            out.append({"id": f"{r['exchange_id']} {r['side']}", "exchange": r["exchange_id"], "race": r.get("race"),
                        "capacity": float(cost), "yield": float(ev) / float(cost)})
    return out


def hold_positions(rows: Sequence[Mapping[str, Any]], haircut: float = SWAP_HAIRCUT) -> list[dict[str, Any]]:
    """Held positions as swap sources: money a sale at the current bid frees (capped at the
    top-of-book size) and the hold-to-settlement yield on it."""
    out = []
    for r in rows:
        if not r.get("shares") or r.get("oracle_mid_now") is None:
            continue
        yes = r["side"] == "yes"
        exit_price = r.get("bid") if yes else (None if r.get("ask") is None else round(1 - r["ask"], 6))
        size = r.get("bid_size") if yes else r.get("ask_size")
        if exit_price is None or exit_price <= 0:
            continue
        shares = r["shares"] if size is None else min(r["shares"], size)
        p_side = r["oracle_mid_now"] if yes else 1 - r["oracle_mid_now"]
        out.append({"id": f"{r['exchange_id']} {r['side']}", "exchange": r["exchange_id"], "race": r["race"], "capital": shares * exit_price,
                    "yield": stats.hold_yield(p_side, exit_price, haircut)})
    return out


def swap_rows(holds: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    """(all swaps best first, total expected gain); greedy, no double-counting."""
    h_by = {h["id"]: h for h in holds}
    c_by = {c["id"]: c for c in candidates}
    swaps = stats.greedy_swaps(holds, candidates)
    rows = [{"hold": sw["hold"], "hold_race": h_by[sw["hold"]]["race"], "hold_yield": round(h_by[sw["hold"]]["yield"], 4),
             "candidate": sw["candidate"], "candidate_race": c_by[sw["candidate"]]["race"], "candidate_yield": round(c_by[sw["candidate"]]["yield"], 4),
             "amount": round(sw["amount"], 2), "yield_gain": round(sw["yield_gain"], 4), "gain": round(sw["gain"], 2)} for sw in swaps]
    rows.sort(key=lambda r: -r["gain"])
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows, round(math.fsum(r["gain"] for r in rows), 2)


# ---------------------------------------------------------------------- inputs (live or saved)


@dataclass
class Inputs:
    equity: float
    positions: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    details: dict[str, dict[str, Any]]
    nodes: dict[str, dict[str, Any]]
    oracle_mids: dict[str, float | None] = field(default_factory=dict)
    oracle_from: str | None = None


def inputs_from_raw(records: Sequence[Mapping[str, Any]]) -> Inputs:
    """Rebuild the inputs from a saved raw.jsonl (latest response of each path wins)."""
    equity, positions, relationships = 0.0, [], []
    details: dict[str, dict[str, Any]] = {}
    nodes: dict[str, dict[str, Any]] = {}
    mids: dict[str, float | None] = {}
    oracle_from = None
    for rec in records:
        if "body" not in rec:
            continue
        path, body = str(rec.get("path") or ""), rec["body"] or {}
        if re.fullmatch(r"/tournaments/[^/]+", path):
            equity = float(body.get("initialBalance") or 0.0)
        elif path == "/portfolio/positions":
            positions = body.get("positions") or []
        elif path == "/relationships":
            relationships = (relationships if (rec.get("params") or {}).get("cursor") else []) + list(body.get("data") or [])
        elif re.fullmatch(r"/tournaments/[^/]+/markets/[^/]+", path):
            details[path.rsplit("/", 1)[1]] = body
        elif re.fullmatch(r"/markets/[^/]+/nodes", path):
            nodes[path.split("/")[2]] = body
        elif path == ORACLE_RECORD_PATH:
            mids, oracle_from = dict(body.get("mids") or {}), f"saved {rec.get('ts')}"
    return Inputs(equity, positions, relationships, details, nodes, mids, oracle_from)


def oracle_mids_from_csv(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["exchange_id"]: (float(r["oracle_mid_now"]) if r.get("oracle_mid_now") not in (None, "") else None) for r in csv.DictReader(fh)}


async def _relationships(reader: ResearchReader, tid: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = await reader.get("/relationships", {"limit": RELATIONSHIPS_PAGE, "cursor": cursor, "tournamentId": tid})
        out.extend(page.get("data") or [])
        pagination = page.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            return out


async def fetch_oracle_mids(in_scope: Mapping[str, Any], raw_map: Mapping[str, Any], polymarket: Any, kalshi: Any) -> dict[str, float | None]:
    oracles = {eid: oracle_entry(raw_map, eid) for eid in in_scope}
    tokens = sorted({o["polymarket_token"] for o in oracles.values() if o["polymarket_token"]})
    tickers = sorted({o["kalshi_ticker"] for o in oracles.values() if o["kalshi_ticker"]})
    pm_quotes, ks_quotes = await asyncio.gather(polymarket.get_quotes(tokens), kalshi.get_quotes(tickers))
    return {eid: oracle_mid(pm_quotes.get(o["polymarket_token"] or ""), ks_quotes.get(o["kalshi_ticker"] or "")) for eid, o in oracles.items()}


async def gather(reader: ResearchReader, slug: str, market_map: MarketMap, raw_map: Mapping[str, Any], races: Sequence[str], polymarket: Any, kalshi: Any) -> Inputs:
    tournament = await reader.get(f"/tournaments/{slug}")
    tid = tournament.get("id")
    positions = (await reader.get("/portfolio/positions")).get("positions") or []
    in_scope = scope(market_map, races, positions)
    market_ids = sorted({v["market_id"] for v in in_scope.values() if v["market_id"]})
    logger.info("Review plan", extra={"exchanges": len(in_scope), "markets": len(market_ids), "reads": 2 * len(market_ids) + 1})
    relationships = await _relationships(reader, tid)
    details, nodes = {}, {}
    for mid in market_ids:
        details[mid] = await reader.get(f"/tournaments/{slug}/markets/{mid}")
        nodes[mid] = await reader.get(f"/markets/{mid}/nodes", {"tournamentId": tid})
    mids = await fetch_oracle_mids(in_scope, raw_map, polymarket, kalshi)
    reader.raw_log.append({"method": "GET", "path": ORACLE_RECORD_PATH, "params": {}, "body": {"mids": mids}})  # for offline re-runs
    return Inputs(float(tournament.get("initialBalance") or 0.0), positions, relationships, details, nodes, mids, "read now")


# ---------------------------------------------------------------------- rows


def build_rows(inp: Inputs, market_map: MarketMap, raw_map: Mapping[str, Any], races: Sequence[str], journal: Sequence[Mapping[str, Any]],
               limits: RiskLimits | None = None) -> list[dict[str, Any]]:
    limits = limits or RiskLimits()
    held = {str(p.get("exchangeId")): p for p in inp.positions if float(p.get("quantity") or 0) != 0}
    clusters = {eid: c for c in build_clusters(market_map, inp.relationships) for eid in c.exchange_ids}
    by_race = market_map.by_race()
    rows = []
    for eid, info in sorted(scope(market_map, races, inp.positions).items()):
        m = market_map.get(eid)
        detail = inp.details.get(info["market_id"]) or {}
        nodes = inp.nodes.get(info["market_id"])
        book = book_of(detail, eid)
        pos = held.get(eid) or {}
        values = position_values(float(pos.get("quantity") or 0), None if pos.get("costBasis") is None else float(pos["costBasis"]),
                                 None if pos.get("avgCost") is None else float(pos["avgCost"]), book)
        j_avg, j_shares = journal_avg_entry(journal, eid, values["side"]) if values["side"] else (None, 0.0)
        oracle = oracle_entry(raw_map, eid)
        title = detail.get("title") or (m.title if m else None)
        rules, rtype = rules_text(nodes), resolution_type(nodes)
        party = party_flag(title, oracle, m.party if m else None)
        direction = stats.leg_direction(m.party if m else None, values["side"], [x.party for x in by_race.get(m.race.key, [])] if m else [])
        rows.append({
            "exchange_id": eid, "market_id": info["market_id"], "in_scope_because": info["why"], "race": m.race.key if m else None, "party": m.party if m else None,
            "sig_title": title, "resolution_type": rtype, "sig_rules": rules, "oracle_polymarket": oracle.get("polymarket"), "oracle_kalshi": oracle.get("kalshi"),
            "oracle_map_prices": json.dumps(oracle.get("map_prices")) if oracle.get("map_prices") else None, "oracle_mid_now": inp.oracle_mids.get(eid),
            "sig_party": party["sig_party"], "oracle_party": party["oracle_party"], "party_check": party["party_check"],
            "flag": bool(party["reasons"]), "flag_reasons": "; ".join(party["reasons"]), **book,
            "shares": values["shares"], "side": values["side"], "direction": direction, "api_avg_cost": pos.get("avgCost"), "cost_basis": pos.get("costBasis"),
            "avg": values["avg"], "journal_avg_entry": j_avg, "journal_filled_shares": j_shares, "notional_at_mid": values["notional_at_mid"],
            "unrealized_at_mid": values["unrealized_at_mid"], "unrealized_at_bid": values["unrealized_at_bid"],
            "contract_cost_share": _share(None if values["avg"] is None else values["avg"] * values["shares"], inp.equity),
            "contract_mid_share": _share(values["notional_at_mid"], inp.equity), "contract_cap": limits.max_fraction_per_contract,
        })
    exposure = race_exposure(rows, market_map)
    cluster_cost: dict[str, float] = defaultdict(float)
    for r in rows:
        c = clusters.get(r["exchange_id"])
        if c is not None and r["shares"]:
            cluster_cost[c.key] += (r["avg"] or 0.0) * r["shares"]
    for r in rows:
        e = exposure.get(r["race"] or "", {})
        c = clusters.get(r["exchange_id"])
        r.update({"race_cost_share": _share(e.get("cost"), inp.equity), "race_mid_share": _share(e.get("mid_value"), inp.equity),
                  "race_worst_loss_share": _share(e.get("worst_loss"), inp.equity), "race_cap": limits.max_race_worst_loss_fraction,
                  "cluster": None if c is None else c.key, "cluster_cost_share": None if c is None else _share(cluster_cost.get(c.key), inp.equity),
                  "cluster_cap": limits.max_fraction_per_cluster})
    return rows


# ---------------------------------------------------------------------- report


def summary_markdown(rows: Sequence[Mapping[str, Any]], races: Sequence[str], race_rows: Sequence[Mapping[str, Any]], totals: Mapping[str, Any],
                     swaps: Sequence[Mapping[str, Any]], swap_total: float, sources: Mapping[str, Any]) -> str:
    held = [r for r in rows if r["shares"]]
    flagged = [r for r in rows if r["flag"]]
    unknown = sum(1 for r in rows if r["party_check"] == "unknown")
    over = [r for r in held if (r["race_worst_loss_share"] or 0) > r["race_cap"] or (r["contract_cost_share"] or 0) > r["contract_cap"]
            or (r["cluster_cost_share"] or 0) > r["cluster_cap"]]
    total_mid = sum(r["unrealized_at_mid"] or 0 for r in held)
    total_bid = sum(r["unrealized_at_bid"] or 0 for r in held)
    worst = totals.get("worst_single_race") or {}
    lines = ["# Position review", "", f"_All values {MARK_LABEL}. SIG data: {sources['sig_from']}. Oracle mids: {sources['oracle_from'] or 'none'}._", "",
             f"- Races reviewed: {', '.join(races)}; exchanges in scope: {len(rows)}; held positions: {len(held)}.",
             f"- Unrealized P&L of held positions: {total_mid:.2f} SUSQ at mid, {total_bid:.2f} at the bid.",
             f"- Party flags (party only, never candidate names): {len(flagged)} mismatches; {unknown} with no party in the oracle label.",
             *[f"  - {r['exchange_id']} {r['race']}: {r['flag_reasons']}" for r in flagged],
             f"- Positions over a configured cap (on starting equity): {len(over)}.",
             *[f"  - {r['exchange_id']} {r['race']} {r['party']} {r['side']}: contract {r['contract_cost_share']} (cap {r['contract_cap']}), "
               f"race worst loss {r['race_worst_loss_share']} (cap {r['race_cap']}), cluster {r['cluster_cost_share']} (cap {r['cluster_cap']})" for r in over],
             "", "## Expected P&L at settlement", "", f"_{EXPECTED_LABEL}._", "",
             f"- Grand total: {totals['expected_pnl']} SUSQ; with a {HAIRCUT} haircut on p_side: {totals['expected_pnl_haircut']} SUSQ; cost {totals['cost']} "
             f"({totals['unpriced_positions']} of {totals['positions']} positions without an oracle mid left out).",
             f"- Stress: if the {len(totals['stress_races'])} races with the largest worst-case loss ({', '.join(totals['stress_races'])}) all resolve against us: "
             f"{totals['stress_loss']} SUSQ.",
             f"- Worst single race: {worst.get('race')} if '{worst.get('scenario')}' wins: {worst.get('pnl')} SUSQ.", "",
             "| race | direction | legs | cost | expected P&L | with haircut | worst case |", "|---|---|---|---|---|---|---|",
             *[f"| {r['race']} | {r['direction']} | {r['legs']} | {r['cost']} | {r['expected_pnl']} | {r['expected_pnl_haircut']} | {r['worst_pnl']} ({r['worst_scenario']}) |"
               for r in race_rows],
             "", "## Swaps", "", f"_{SWAP_LABEL}. Candidates: {sources['candidates_from'] or 'none found'}; yields at haircut {SWAP_HAIRCUT}._", "",
             f"- {len(swaps)} swaps; total expected gain {swap_total} SUSQ.", "",
             "| rank | sell (hold) | hold yield | buy (candidate) | candidate yield | amount | gain |", "|---|---|---|---|---|---|---|",
             *[f"| {w['rank']} | {w['hold']} ({w['hold_race']}) | {w['hold_yield']} | {w['candidate']} ({w['candidate_race']}) | {w['candidate_yield']} | {w['amount']} | {w['gain']} |"
               for w in swaps[:TOP_SWAPS]],
             "", "## Largest unrealized losses (marked at mid)", "",
             *[f"- {r['exchange_id']} {r['race']} {r['party']} {r['side']} {r['shares']:g} sh @ {r['avg']} -> mid {r['mid']}: {r['unrealized_at_mid']} (bid: {r['unrealized_at_bid']})"
               for r in sorted(held, key=lambda r: r["unrealized_at_mid"] if r["unrealized_at_mid"] is not None else 0)[:10]], ""]
    return "\n".join(lines)


def newest_candidates(root: Path | None = None) -> Path | None:
    runs = sorted((root or OUT_ROOT / "settlement_carry_scan").glob("*/candidates.csv"))
    return runs[-1] if runs else None


def write_report(out: Path, inp: Inputs, market_map: MarketMap, raw_map: Mapping[str, Any], races: Sequence[str], journal: Sequence[Mapping[str, Any]], sig_from: str,
                 candidates_path: Path | None = None) -> dict[str, Any]:
    rows = build_rows(inp, market_map, raw_map, races, journal)
    positions = expected_pnl_rows(rows)
    race_rows = race_expected(rows, positions, market_map)
    totals = expected_totals(positions, race_rows)
    swaps, swap_total = swap_rows(hold_positions(rows), load_candidates(candidates_path))
    sources = {"sig_from": sig_from, "oracle_from": inp.oracle_from, "candidates_from": None if candidates_path is None else str(candidates_path)}
    write_csv(out / "position_review.csv", rows, FIELDS)
    total_row = {k: totals[k] for k in ("level", "expected_pnl", "expected_pnl_haircut", "cost")} | {"legs": f"stress {totals['stress_races']}: {totals['stress_loss']}"}
    write_csv(out / "expected_pnl.csv", [*positions, *race_rows, total_row], EXPECTED_FIELDS)
    write_csv(out / "swaps.csv", swaps, SWAP_FIELDS)
    (out / "summary.md").write_text(summary_markdown(rows, races, race_rows, totals, swaps, swap_total, sources), encoding="utf-8")
    for r in rows:
        logger.info("Position review", extra={k: r.get(k) for k in FIELDS if k != "sig_rules"})
    logger.info("Position review done (HYPOTHETICAL, oracle-based)", extra={"out_dir": str(out), "rows": len(rows), "flagged": sum(1 for r in rows if r["flag"]),
                                                                         **totals, "swaps": len(swaps), "swap_gain": swap_total, "label": EXPECTED_LABEL})
    return {"rows": rows, "expected": positions, "races": race_rows, "totals": totals, "swaps": swaps, "swap_total": swap_total}


async def _live(reads_per_min: float, market_map: MarketMap, raw_map: Mapping[str, Any], races: Sequence[str], journal: Sequence[Mapping[str, Any]],
                candidates_path: Path | None) -> int:
    async with research_reader(SCRIPT, reads_per_min) as reader, PolymarketClient() as polymarket, KalshiClient() as kalshi:
        inp = await gather(reader, reader.settings.tournament_slug, market_map, raw_map, races, polymarket, kalshi)
        out = reader.out_dir
    write_report(out, inp, market_map, raw_map, races, journal, "read now", candidates_path)
    return 0


async def _fresh_oracle(inp: Inputs, market_map: MarketMap, raw_map: Mapping[str, Any], races: Sequence[str]) -> dict[str, float | None]:
    async with PolymarketClient() as polymarket, KalshiClient() as kalshi:
        return await fetch_oracle_mids(scope(market_map, races, inp.positions), raw_map, polymarket, kalshi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: worst replay races and held positions - rules, oracle, book, P&L at mid, settlement carry, caps.")
    add_research_args(parser)
    parser.add_argument("--races", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None, help="comma list of race keys")
    parser.add_argument("--from-replay", action="store_true", help=f"take the {REPLAY_TOP} worst dry-run races from the newest dry_run_replay by_race.csv")
    parser.add_argument("--offline", action="store_true", help="replay the saved reads: no SIG request")
    parser.add_argument("--raw", type=Path, default=None, help="--offline: saved raw.jsonl to replay (default: the newest position_review run)")
    parser.add_argument("--candidates", type=Path, default=None, help="settlement_carry_scan candidates.csv for the swaps (default: the newest)")
    parser.add_argument("--fresh-oracle", action="store_true", help="offline: read the public oracle quotes now (0 SIG reads)")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    races = list(args.races or DEFAULT_RACES)
    if args.from_replay:
        csvs = sorted((OUT_ROOT / "dry_run_replay").glob("*/by_race.csv"))
        if not csvs:
            logger.error("No dry_run_replay by_race.csv found")
            return 2
        races = races_from_replay(csvs[-1])
        logger.info("Races from the replay", extra={"file": str(csvs[-1]), "races": races})
    market_map = MarketMap.load()
    raw_map = json.loads(MARKET_MAP_PATH.read_text(encoding="utf-8"))
    unknown = [r for r in races if r not in market_map.by_race()]
    if unknown:
        logger.warning("Races not in the market map (skipped)", extra={"races": unknown})
    journal = read_jsonl(ORDER_JOURNAL_PATH)
    candidates_path = args.candidates or newest_candidates()
    if candidates_path is None:
        logger.info("No settlement_carry_scan candidates.csv found: the swap table stays empty")
    race_markets = {m.sig_market_id for m in market_map if m.race.key in set(races)}
    estimate = ReadEstimate(3 + 2 * (len(race_markets) + HELD_ESTIMATE), reads_per_min)

    if args.yes:
        confirm_fetch(estimate, True)
        return asyncio.run(_live(reads_per_min, market_map, raw_map, races, journal, candidates_path))
    if not args.offline:
        confirm_fetch(estimate, False)
        logger.info("Or replay the saved reads with --offline (0 SIG reads)")
        return 0

    raw = args.raw or (sorted((OUT_ROOT / SCRIPT).glob("*/raw.jsonl")) or [None])[-1]
    if raw is None or not Path(raw).exists():
        logger.info("No saved position_review read to replay: run once with --yes")
        return 0
    raw = Path(raw)
    inp = inputs_from_raw(read_jsonl(raw))
    if args.fresh_oracle:
        inp.oracle_mids, inp.oracle_from = asyncio.run(_fresh_oracle(inp, market_map, raw_map, races)), "public oracles read now"
    elif not inp.oracle_mids:
        inp.oracle_mids = oracle_mids_from_csv(raw.parent / "position_review.csv")
        inp.oracle_from = f"saved {raw.parent / 'position_review.csv'}" if inp.oracle_mids else None
    missing = sorted({v["market_id"] for v in scope(market_map, races, inp.positions).values()} - set(inp.details))
    if missing:
        logger.warning("Markets in scope without a saved detail read (book / rules empty)", extra={"markets": missing})
    logger.info("Offline: replaying saved SIG reads (0 new reads)", extra={"raw": str(raw), "oracle_from": inp.oracle_from})
    write_report(run_dir(SCRIPT), inp, market_map, raw_map, races, journal, f"saved {raw}", candidates_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
