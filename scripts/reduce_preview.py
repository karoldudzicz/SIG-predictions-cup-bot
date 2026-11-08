"""READ-ONLY preview: what selling part of a held position into the SIG book would do, in numbers.
Nothing is sent but GETs; any sell is MANUAL. No recommendation is made - only numbers.

Markets (default): 1076 senate:RI D (YES held), 1077 senate:RI R (NO held), 1065 house:NH-01 R (NO held).
SIG reads: GET /portfolio/positions (shares, cost basis) and ONE order book per contract (depth 50)
- nothing else. Spaced at RESEARCH_READS_PER_MIN (default 10, max 15); the planned reads are logged
and nothing is fetched without --yes. Oracle fair (P(YES)): the newest saved
data/edge_scan/quoter_eligibility/*/oracle_quotes.json with the bot's rules (research.stats.oracle_fair,
quotes aged as at that fetch) - its age is logged and reported; --fair EID=P overrides one market.

Per market and sell fraction (25 / 50 / 75 / 100% of the shares held), selling into the held side's
bids (YES: YES bids; NO: YES asks at 1 - ask), all in the held side's terms:
  - VWAP, levels consumed, depth enough or not; slippage in SUSQ vs the best bid and vs the mid;
  - proceeds vs the cost basis of the shares sold (realized P&L);
  - SUSQ given up vs holding at the oracle-implied value = sold x p_side - proceeds;
  - expected settlement P&L of the whole position before / after (HYPOTHETICAL: oracle p taken as
    truth), and with p shrunk 0.02 toward 0.5;
  - worst case (the race goes the other way) before / after;
  - break-even: the oracle P(YES) at which holding the sold shares equals selling them (= the VWAP in
    YES terms).
Races with two or more of these markets (1076 + 1077: the same RI event) get a combined table:
summed values and the worst settlement scenario over the race's parties and "other".
Per fraction: total spread cost (vs the mid and vs the best bid) over all markets.

NOT modelled: settlement 0 / 1 risk beyond the worst case shown, the oracle-vs-SIG resolution basis
risk, book changes between the read and a manual sell, fees.
-> data/edge_scan/reduce_preview/<UTC date>/: raw.jsonl, by_market.csv, by_race.csv, summary.json, report.md.

    python scripts/reduce_preview.py [--yes] [--reads-per-min N] [--markets 1076,1077,1065] [--fair 1076=0.93]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, add_research_args, confirm_fetch, research_reader, resolve_reads_per_min, write_csv, write_json  # noqa: E402

logger = logging.getLogger("reduce_preview")

SCRIPT = "reduce_preview"
DEFAULT_MARKETS = ("1076", "1077", "1065")
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
BOOK_DEPTH = 50
SHRINK = 0.02
OTHER = "other"
NOTE = ("HYPOTHETICAL numbers only, no recommendation. Any sell is MANUAL. NOT modelled: settlement 0 / 1 risk beyond the worst case shown, "
        "the oracle-vs-SIG resolution basis risk, book changes before a manual sell, fees.")
MARKET_FIELDS = ("exchange_id", "race", "party", "side", "shares_held", "avg_cost", "fraction", "sell_shares", "filled", "depth_enough", "levels_used",
                 "best_bid", "vwap", "last_price", "mid", "slip_vs_best", "slip_vs_mid", "proceeds", "cost_sold", "realized_pnl", "fair_yes", "p_side",
                 "lost_vs_oracle", "remaining_shares", "ev_hold", "ev_after", "ev_hold_shrunk", "ev_after_shrunk", "worst_before", "worst_after",
                 "break_even_yes_p")
RACE_FIELDS = ("race", "fraction", "markets", "filled", "proceeds", "slip_vs_mid", "lost_vs_oracle", "ev_hold", "ev_after", "ev_hold_shrunk",
               "ev_after_shrunk", "worst_before", "worst_after", "worst_winner", "break_even_held_p")


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------------- inputs


def held_position(positions: Sequence[Mapping[str, Any]], exchange_id: str) -> dict[str, Any] | None:
    """(side, shares, average cost in the side's terms) of one open position, or None."""
    for p in positions:
        quantity = float(p.get("quantity") or 0)
        if str(p.get("exchangeId")) != exchange_id or not quantity or p.get("settled"):
            continue
        shares = abs(quantity)
        avg = abs(float(p["costBasis"])) / shares if p.get("costBasis") is not None else float(p.get("avgCost") or 0.0)
        return {"side": "yes" if quantity > 0 else "no", "shares": shares, "avg": avg}
    return None


def newest_oracle(root: Path = OUT_ROOT) -> Path | None:
    runs = sorted((root / "quoter_eligibility").glob("*/oracle_quotes.json"))
    return runs[-1] if runs else None


def saved_fair(oracle: Mapping[str, Any], mapping: Any) -> tuple[float | None, str]:
    """Oracle fair P(YES) of one market from a saved oracle_quotes.json, quotes aged as at that fetch."""
    if mapping is None:
        return None, "not mapped"
    at = datetime.fromisoformat(str(oracle["fetched_at"]).replace("Z", "+00:00")).timestamp()

    def quote(raw: Mapping[str, Any] | None) -> Any:
        return None if raw is None else SimpleNamespace(mid=raw.get("mid"), spread=raw.get("spread"), age_seconds=at - float(raw.get("fetched_at") or 0.0))

    pm = (oracle.get("polymarket") or {}).get(mapping.polymarket_token) if mapping.polymarket_token else None
    ks = (oracle.get("kalshi") or {}).get(mapping.kalshi_ticker) if mapping.kalshi_ticker else None
    if pm is None and ks is None:
        return None, "not in the saved oracle pass"
    return stats.oracle_fair((quote(pm), quote(ks)), float(oracle.get("max_age_s") or 10.0))


def parse_fair_overrides(values: Sequence[str]) -> dict[str, float]:
    out = {}
    for v in values:
        eid, _, p = v.partition("=")
        prob = float(p)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"--fair {v}: probability must be in [0, 1]")
        out[eid.strip()] = prob
    return out


# ---------------------------------------------------------------------- per market


def market_rows(eid: str, race: str | None, party: str | None, parties: Sequence[str], held: Mapping[str, Any], book: Mapping[str, Any],
                fair_yes: float | None) -> list[dict[str, Any]]:
    """One row per sell fraction: book sweep, realized P&L, oracle value given up, EV and worst case."""
    side, shares, avg = held["side"], held["shares"], held["avg"]
    levels = stats.sell_levels(book, side)
    bids, asks = book.get("bids") or [], book.get("asks") or []
    yes_mid = (max(float(l["price"]) for l in bids) + min(float(l["price"]) for l in asks)) / 2 if bids and asks else None
    mid = None if yes_mid is None else (yes_mid if side == "yes" else 1.0 - yes_mid)
    p_side = None if fair_yes is None else (fair_yes if side == "yes" else 1.0 - fair_yes)
    p_shrunk = None if fair_yes is None else (lambda f: f if side == "yes" else 1.0 - f)(stats.shrink_toward_half(fair_yes, SHRINK))
    winners = [*parties, OTHER]
    worst_before = min(stats.race_scenario_pnl([(party, side, shares, shares * avg)], winners).values())
    rows = []
    for fraction in FRACTIONS:
        sell = shares if fraction >= 1.0 else math.floor(fraction * shares)
        s = stats.sweep_sell(levels, sell)
        filled, proceeds = s["filled"], s["proceeds"]
        remaining = shares - filled

        def ev(p: float | None) -> float | None:
            return None if p is None else proceeds + remaining * p - shares * avg

        worst_after = min(stats.race_scenario_pnl([(party, side, remaining, shares * avg - proceeds)], winners).values())
        rows.append({
            "exchange_id": eid, "race": race, "party": party, "side": side, "shares_held": shares, "avg_cost": _r(avg, 4), "fraction": fraction,
            "sell_shares": sell, "filled": filled, "depth_enough": s["enough"], "levels_used": s["levels"], "best_bid": s["best"], "vwap": _r(s["vwap"], 4),
            "last_price": s["last"], "mid": _r(mid, 4),
            "slip_vs_best": None if s["best"] is None else _r(filled * s["best"] - proceeds),
            "slip_vs_mid": None if mid is None else _r(filled * mid - proceeds),
            "proceeds": _r(proceeds), "cost_sold": _r(filled * avg), "realized_pnl": _r(proceeds - filled * avg),
            "fair_yes": _r(fair_yes, 4), "p_side": _r(p_side, 4), "lost_vs_oracle": None if p_side is None else _r(filled * p_side - proceeds),
            "remaining_shares": remaining,
            "ev_hold": None if p_side is None else _r(shares * (p_side - avg)), "ev_after": _r(ev(p_side)),
            "ev_hold_shrunk": None if p_shrunk is None else _r(shares * (p_shrunk - avg)), "ev_after_shrunk": _r(ev(p_shrunk)),
            "worst_before": _r(worst_before), "worst_after": _r(worst_after),
            "break_even_yes_p": None if s["vwap"] is None else _r(s["vwap"] if side == "yes" else 1.0 - s["vwap"], 4),
        })
    return rows


def race_rows(race: str, parties: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The markets of one race together, per fraction: sums, and the worst settlement scenario."""
    winners = [*parties, OTHER]
    out = []
    for fraction in FRACTIONS:
        part = [r for r in rows if r["fraction"] == fraction]
        before = stats.race_scenario_pnl([(r["party"], r["side"], r["shares_held"], r["shares_held"] * r["avg_cost"]) for r in part], winners)
        after = stats.race_scenario_pnl([(r["party"], r["side"], r["remaining_shares"], r["shares_held"] * r["avg_cost"] - r["proceeds"]) for r in part], winners)
        worst = min(after, key=lambda w: (after[w], w))

        def total(key: str) -> float | None:
            values = [r[key] for r in part]
            return None if any(v is None for v in values) else _r(math.fsum(values))

        filled = math.fsum(r["filled"] for r in part)
        out.append({"race": race, "fraction": fraction, "markets": ",".join(r["exchange_id"] for r in part), "filled": filled, "proceeds": total("proceeds"),
                    "slip_vs_mid": total("slip_vs_mid"), "lost_vs_oracle": total("lost_vs_oracle"), "ev_hold": total("ev_hold"), "ev_after": total("ev_after"),
                    "ev_hold_shrunk": total("ev_hold_shrunk"), "ev_after_shrunk": total("ev_after_shrunk"), "worst_before": _r(min(before.values())),
                    "worst_after": _r(after[worst]), "worst_winner": worst,
                    "break_even_held_p": _r(math.fsum(r["proceeds"] for r in part) / filled, 4) if filled else None})
    return out


def spread_totals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"fraction": f, "slip_vs_mid": _r(math.fsum(r["slip_vs_mid"] or 0.0 for r in rows if r["fraction"] == f)),
             "slip_vs_best": _r(math.fsum(r["slip_vs_best"] or 0.0 for r in rows if r["fraction"] == f)),
             "depth_short": [r["exchange_id"] for r in rows if r["fraction"] == f and not r["depth_enough"]]} for f in FRACTIONS]


def markdown(rows: Sequence[Mapping[str, Any]], races: Sequence[Mapping[str, Any]], totals: Sequence[Mapping[str, Any]], oracle_note: str) -> str:
    def table(items: Sequence[Mapping[str, Any]], cols: Sequence[str]) -> list[str]:
        return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in items]

    cols = ("fraction", "filled", "depth_enough", "levels_used", "vwap", "slip_vs_best", "slip_vs_mid", "proceeds", "realized_pnl", "lost_vs_oracle",
            "ev_hold", "ev_after", "ev_after_shrunk", "worst_before", "worst_after", "break_even_yes_p")
    lines = ["# Reduce preview (HYPOTHETICAL)", "", f"_{NOTE}_", "", f"Oracle: {oracle_note}", ""]
    for eid in dict.fromkeys(r["exchange_id"] for r in rows):
        part = [r for r in rows if r["exchange_id"] == eid]
        h = part[0]
        lines += [f"## {eid} {h['race']} {h['party']} {h['side'].upper()} {h['shares_held']:g} sh @ {h['avg_cost']} (fair YES {h['fair_yes']}, mid {h['mid']})", "",
                  *table(part, cols), ""]
    for race in dict.fromkeys(r["race"] for r in races):
        lines += [f"## {race} combined", "", *table([r for r in races if r["race"] == race], RACE_FIELDS[1:]), ""]
    lines += ["## Spread cost, all markets", "", *table(totals, ("fraction", "slip_vs_mid", "slip_vs_best", "depth_short")), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------- run


async def fetch(reader: Any, markets: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions = (await reader.get("/portfolio/positions")).get("positions") or []
    books = {eid: await reader.get(f"/exchanges/{eid}/orderbook", {"depth": BOOK_DEPTH}) for eid in markets}
    return positions, books


def load_market_map() -> MarketMap:
    return MarketMap.load()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: sell-fraction preview of held positions against the SIG book (HYPOTHETICAL, no orders).")
    add_research_args(parser)
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help=f"comma-separated exchange ids (default {','.join(DEFAULT_MARKETS)})")
    parser.add_argument("--oracle", type=Path, default=None, help="a saved oracle_quotes.json (default: the newest quoter_eligibility one)")
    parser.add_argument("--fair", action="append", default=[], help="EID=P: use P(YES) = P for this market instead of the saved oracle")
    args = parser.parse_args(argv)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
        overrides = parse_fair_overrides(args.fair)
    except ValueError as exc:
        parser.error(str(exc))

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    if not confirm_fetch(ReadEstimate(1 + len(markets), reads_per_min), args.yes):
        return 0
    market_map = load_market_map()
    oracle_path = args.oracle or newest_oracle()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path is not None and oracle_path.exists() else None
    if oracle is None:
        oracle_note = "no saved oracle pass: oracle-based columns empty unless --fair is given"
    else:
        fetched = datetime.fromisoformat(str(oracle["fetched_at"]).replace("Z", "+00:00"))
        oracle_note = f"{oracle_path} fetched {oracle['fetched_at']}, {(datetime.now(timezone.utc) - fetched).total_seconds() / 3600:.1f} h old at this run"
    logger.info("Oracle source", extra={"note": oracle_note, "overrides": overrides})

    async def go() -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
        async with research_reader(SCRIPT, reads_per_min) as reader:
            positions, books = await fetch(reader, markets)
            return positions, books, reader.out_dir

    positions, books, out = asyncio.run(go())

    rows: list[dict[str, Any]] = []
    for eid in markets:
        held = held_position(positions, eid)
        if held is None:
            logger.warning("No open position on this market: skipped", extra={"exchange_id": eid})
            continue
        m = market_map.get(eid)
        race, party = (m.race.key, m.party) if m is not None else (None, None)
        parties = sorted({x.party for x in market_map.by_race().get(race, [])}) if race else [party or "?"]
        if eid in overrides:
            fair, reason = overrides[eid], "manual --fair"
        elif oracle is not None:
            fair, reason = saved_fair(oracle, m)
        else:
            fair, reason = None, "no saved oracle pass"
        if fair is None:
            logger.warning("No oracle fair: oracle-based columns empty", extra={"exchange_id": eid, "reason": reason})
        rows += market_rows(eid, race, party or "?", parties, held, books[eid], fair)
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["race"]:
            by_race[r["race"]].append(r)
    races = [x for race, part in sorted(by_race.items()) if len({r["exchange_id"] for r in part}) >= 2
             for x in race_rows(race, sorted({m.party for m in market_map.by_race().get(race, [])}), part)]
    totals = spread_totals(rows)

    write_csv(out / "by_market.csv", rows, MARKET_FIELDS)
    write_csv(out / "by_race.csv", races, RACE_FIELDS)
    write_json(out / "summary.json", {"note": NOTE, "oracle": oracle_note, "fair_overrides": overrides, "fractions": FRACTIONS, "book_depth": BOOK_DEPTH,
                                      "shrink_toward_half": SHRINK, "spread_cost": totals, "markets": rows, "races": races})
    (out / "report.md").write_text(markdown(rows, races, totals, oracle_note), encoding="utf-8")

    for r in rows:
        logger.info("Reduce preview", extra={k: r[k] for k in ("exchange_id", "side", "fraction", "filled", "depth_enough", "levels_used", "vwap", "slip_vs_best",
                                                                 "slip_vs_mid", "proceeds", "realized_pnl", "lost_vs_oracle", "ev_hold", "ev_after",
                                                                 "ev_after_shrunk", "worst_before", "worst_after", "break_even_yes_p")})
    for r in races:
        logger.info("Reduce preview, race combined", extra=r)
    for t in totals:
        logger.info("Spread cost, all markets", extra=t)
    logger.info(NOTE)
    logger.info("Reduce preview done", extra={"out_dir": str(out), "markets": len({r["exchange_id"] for r in rows})})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
