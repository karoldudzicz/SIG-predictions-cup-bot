"""READ-ONLY feasibility count: which maker_markout_quotes markets could the quoter quote today, and
how did the 300 s maker markout vs the quote mid look on just those markets?

NOT a hypothesis test: no pre-registered rule, nothing frozen. HYPOTHETICAL / PROXY throughout:
fills are tape prints at the touch, queue position is not modelled, nor settlement risk or fees.

Markets: every market with priced trades in the saved tape AND quote snapshots (the inputs and
defaults of maker_markout_quotes). Per market:
  - mapped in market_map.json, as a D or R party-market (the quoter quotes nothing else);
  - oracle fair and uncertainty u from ONE fresh pass of the bot's oracle readers (Polymarket and
    Kalshi public endpoints) with the bot's rules: research.stats.oracle_fair / oracle_uncertainty,
    copies of strategy.compute_fair_price / ladder.uncertainty_band;
  - fair within [FAIR_LOW, FAIR_HIGH] and u <= MAX_U (the quoter's tail and oracle guards);
  - carry: a position held per the newest saved position_review /portfolio/positions (no read);
  - ladder market: open ladder inventory in data/ladder_journal.jsonl. Resting ladder orders
    cannot be seen without a read; LADDER_ENABLED is logged.
ELIGIBLE = mapped D / R, not a ladder market, a fresh fair, u <= MAX_U, fair in range: the quoter's
market guards, checked in its order. Carry does not block the quoter today, so the markouts are
cut for all eligible markets (scope "eligible") and for those without carry (scope "no_carry").

Markouts, eligible markets only: maker_markout_quotes.match_market (Lee-Ready against the
prevailing quote, tick-rule fallback; mid at t + 300 s at most 45 s old). Mean in ticks with the
cluster bootstrap (cluster = market-day, 1000 resamples, seed 7, 95%) by quoted spread at the trade
(all / 1 / 2 / >=3) x time half (all / first / second, split over the called prints of that scope
and price group) x price (all / cheap = <0.15 or >0.85). Eligibility is as of NOW (one oracle
pass) and is applied to the whole saved history.

Feasibility: eligible markets and matched prints per UTC day. INSUFFICIENT for a pilot with fewer
than MIN_MARKETS eligible markets or fewer than MIN_PRINTS_SPREAD2 matched prints at spread >= 2.

Reads: 0 SIG reads. One oracle pass (public endpoints) for the mapped D / R markets, only with
--yes. Memory: streamed like maker_markout_quotes; --max-rss-mb (default 350) stops cleanly, exit 3.
-> data/edge_scan/quoter_eligibility/<UTC date>/: eligibility.csv, oracle_quotes.json, by_cut.csv,
   by_day.csv, summary.json.

    python scripts/quoter_eligibility.py --yes [--tape PATH ...] [--snapshots PATH ...] [--positions PATH] [--max-rss-mb 350]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import DATA_DIR, MarketMap  # noqa: E402
from oracle_api import KalshiClient, PolymarketClient  # noqa: E402
from research import stats  # noqa: E402
from research.memory import MemoryLimitExceeded, RssGuard  # noqa: E402
from research.reader import OUT_ROOT, iter_jsonl, run_dir, write_csv, write_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("maker_markout_quotes", ROOT / "scripts" / "maker_markout_quotes.py")
mq = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("maker_markout_quotes", mq)
_spec.loader.exec_module(mq)  # reuse the streaming stores and the per-market matching
maker = mq.maker

logger = logging.getLogger("quoter_eligibility")

SCRIPT = "quoter_eligibility"
TICK = 0.005
HORIZON = 300
FAIR_LOW, FAIR_HIGH = 0.05, 0.95  # the quoter's tail guard (QUOTER_TAIL_LOW / HIGH defaults)
MAX_U = 0.04  # the quoter's oracle guard (QUOTER_MAX_U)
MIN_MARKETS = 8
MIN_PRINTS_SPREAD2 = 300
DEFAULT_MAX_RSS_MB = 350.0
EXIT_MEMORY = 3
LADDER_JOURNAL = DATA_DIR / "ladder_journal.jsonl"
SCOPES = ("eligible", "no_carry")
PRICES = ("all", "cheap")
SPREADS = ("all", "1", "2", ">=3")
HALVES = ("all", "first_half", "second_half")
_EPS = 1e-9
LABEL = ("HYPOTHETICAL / PROXY maker markout vs the quote mid on quoter-eligible markets: fills = tape prints at the touch; "
         "NOT modelled: queue position, settlement risk (0 / 1), fees. Eligibility as of now, applied to the whole history. Feasibility count, not a test.")
TABLE_FIELDS = ("exchange_id", "title", "race", "party", "mapped", "polymarket", "kalshi", "fair", "u", "fair_reason", "fair_in_range", "u_ok",
                "carry", "ladder", "eligible", "reason", "matched_prints", "matched_spread_ge2", "last_bid", "last_ask", "last_spread_ticks", "last_quote_age_min")
CUT_FIELDS = ("scope", "price", "spread", "half", "horizon_s", "prints", "qualifying", "drop_rate", "markets", "mean_ticks", "ci_low_ticks", "ci_high_ticks", "label")
DAY_FIELDS = ("scope", "day", "markets", "matched_prints", "matched_spread_ge2", "called_prints", "qualifying_300")


# ---------------------------------------------------------------------- pure helpers


def block_reason(party: str | None, in_ladder: bool, fair: float | None, fair_reason: str, u: float | None) -> str | None:
    """Why the quoter would not quote this market (None = eligible). `party` None = not in the map.
    Same order as the quoter: D / R, ladder market, fair, u, tails."""
    if party is None:
        return "not mapped"
    if party not in ("D", "R"):
        return "not D / R"
    if in_ladder:
        return "ladder market"
    if fair is None:
        return f"no oracle fair ({fair_reason})"
    if u is not None and u > MAX_U + _EPS:
        return f"u > {MAX_U}"
    if not FAIR_LOW - _EPS <= fair <= FAIR_HIGH + _EPS:
        return f"fair outside [{FAIR_LOW}, {FAIR_HIGH}]"
    return None


def held_from_positions(positions: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, int]]:
    """/portfolio/positions -> exchange id -> (side, shares); quantity > 0 = YES, < 0 = NO; settled skipped."""
    held: dict[str, tuple[str, int]] = {}
    for p in positions:
        quantity = p.get("quantity") or 0
        if p.get("settled") or not quantity or not p.get("exchangeId"):
            continue
        held[str(p["exchangeId"])] = ("yes" if quantity > 0 else "no", int(abs(quantity)))
    return held


def ladder_markets(lines: Iterable[Mapping[str, Any]]) -> set[str]:
    """Exchanges with open ladder inventory, replayed like LadderJournal.replay (fill +, exit_fill -,
    round_closed clears)."""
    shares: dict[str, int] = {}
    for e in lines:
        eid, event = str(e.get("exchange_id")), e.get("event")
        if event == "fill":
            shares[eid] = shares.get(eid, 0) + int(e.get("quantity") or 0)
        elif event == "exit_fill" and eid in shares:
            shares[eid] -= int(e.get("quantity") or 0)
        elif event == "round_closed":
            shares.pop(eid, None)
    return {eid for eid, n in shares.items() if n > 0}


def feasibility(eligible_markets: int, prints_spread_ge2: int) -> str:
    if eligible_markets < MIN_MARKETS or prints_spread_ge2 < MIN_PRINTS_SPREAD2:
        return "INSUFFICIENT for a pilot"
    return "enough sample for a pilot (feasibility count only, not a test)"


def last_quote(q: Any, now: float) -> dict[str, Any]:
    """The newest snapshot of one market: bid, ask, spread in ticks, age in minutes."""
    if not len(q.t):
        return {"last_bid": None, "last_ask": None, "last_spread_ticks": None, "last_quote_age_min": None}
    i = max(range(len(q.t)), key=lambda k: q.t[k])
    bid, ask = mq._opt(q.bid[i]), mq._opt(q.ask[i])
    return {"last_bid": bid, "last_ask": ask, "last_spread_ticks": None if bid is None or ask is None else round((ask - bid) / TICK),
            "last_quote_age_min": round((now - q.t[i]) / 60, 1)}


# ---------------------------------------------------------------------- inputs (no SIG reads)


def saved_carry(path: Path | None) -> tuple[dict[str, tuple[str, int]] | None, str | None]:
    """Positions held per the newest /portfolio/positions record of a saved position_review raw.jsonl."""
    if path is None or not path.exists():
        return None, None
    positions, at = None, None
    for rec in iter_jsonl(path):
        if rec.get("path") == "/portfolio/positions" and "body" in rec:
            positions, at = list((rec["body"] or {}).get("positions") or []), rec.get("ts")
    return (None, None) if positions is None else (held_from_positions(positions), f"{path} ({at})")


def newest_review(root: Path = OUT_ROOT) -> Path | None:
    runs = sorted((root / "position_review").glob("*/raw.jsonl"))
    return runs[-1] if runs else None


async def _no_quotes() -> dict[str, Any]:
    return {}


async def fetch_oracles(tokens: Sequence[str], tickers: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One pass of the bot's oracle readers (public Polymarket / Kalshi endpoints)."""
    async with PolymarketClient() as polymarket, KalshiClient() as kalshi:
        pm, ks = await asyncio.gather(polymarket.get_quotes(list(tokens)) if tokens else _no_quotes(),
                                      kalshi.get_quotes(list(tickers)) if tickers else _no_quotes())
    return pm, ks


def load_market_map() -> MarketMap:
    return MarketMap.load()


# ---------------------------------------------------------------------- eligibility table


def eligibility_rows(markets: Sequence[str], market_map: Any, ladder: set[str], carry: Mapping[str, tuple[str, int]] | None,
                     pm: Mapping[str, Any], ks: Mapping[str, Any], max_age_s: float, quotes: Any, now: float) -> list[dict[str, Any]]:
    rows = []
    for eid in markets:
        m = market_map.get(eid)
        pm_q = pm.get(m.polymarket_token) if m is not None and m.polymarket_token else None
        ks_q = ks.get(m.kalshi_ticker) if m is not None and m.kalshi_ticker else None
        fair, fair_reason = stats.oracle_fair((pm_q, ks_q), max_age_s) if m is not None else (None, "not mapped")
        u = stats.oracle_uncertainty((pm_q, ks_q), max_age_s, TICK) if m is not None else None
        held = None if carry is None else carry.get(eid)
        reason = block_reason(None if m is None else m.party, eid in ladder, fair, fair_reason, u)
        rows.append({
            "exchange_id": eid, "title": None if m is None else m.title, "race": None if m is None else m.race.key, "party": None if m is None else m.party,
            "mapped": m is not None, "polymarket": pm_q is not None, "kalshi": ks_q is not None,
            "fair": None if fair is None else round(fair, 4), "u": None if u is None else round(u, 4), "fair_reason": fair_reason,
            "fair_in_range": None if fair is None else FAIR_LOW - _EPS <= fair <= FAIR_HIGH + _EPS, "u_ok": None if u is None else u <= MAX_U + _EPS,
            "carry": "unknown" if carry is None else (f"{held[0]} {held[1]}" if held else ""), "ladder": eid in ladder,
            "eligible": reason is None, "reason": reason or "", "matched_prints": None, "matched_spread_ge2": None,
            **last_quote(quotes.markets[eid], now),
        })
    return rows


def table_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in rows if r["eligible"]]
    return {"markets": len(rows), "mapped": sum(r["mapped"] for r in rows), "d_or_r": sum(r["party"] in ("D", "R") for r in rows),
            "ladder": sum(r["ladder"] for r in rows), "with_fair": sum(r["fair"] is not None for r in rows),
            "fair_in_range": sum(bool(r["fair_in_range"]) for r in rows), "u_ok": sum(bool(r["u_ok"]) for r in rows),
            "eligible": len(eligible), "eligible_with_carry": sum(r["carry"] not in ("", "unknown") for r in eligible),
            "eligible_no_carry": None if any(r["carry"] == "unknown" for r in rows) else sum(r["carry"] == "" for r in eligible),
            "blocked": dict(Counter(r["reason"] for r in rows if r["reason"]))}


# ---------------------------------------------------------------------- markouts on eligible markets


class EligibleMarkouts:
    """pass1(scopes, rows) for every eligible market, finish_pass1(), then pass2(market, scopes, rows)
    in the SAME order. Only aggregates outlive a market."""

    def __init__(self) -> None:
        self.agg = stats.CutAggregator(TICK)
        self._times: dict[tuple[str, str], array] = defaultdict(lambda: array("d"))
        self.halves: dict[tuple[str, str], stats.TimeHalves] = {}
        self.days: dict[tuple[str, str], dict[str, Any]] = {}
        self.per_market: dict[str, dict[str, int]] = {}
        self.totals: dict[str, Counter] = defaultdict(Counter)
        self.markets: dict[str, set[str]] = defaultdict(set)

    def pass1(self, scopes: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
        for r in rows:
            if r["call"] is not None:
                for scope in scopes:
                    for price in PRICES if r["cheap"] else ("all",):
                        self._times[(scope, price)].append(r["ts"])

    def finish_pass1(self) -> None:
        self.halves = {key: stats.TimeHalves(times) for key, times in self._times.items()}
        self._times = defaultdict(lambda: array("d"))

    def pass2(self, market: str, scopes: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
        items: dict[tuple[str, str, str, str], list[tuple[str, float | None]]] = defaultdict(list)
        ge2 = sum(r["spread"] in ("2", ">=3") for r in rows)
        self.per_market[market] = {"matched_prints": len(rows), "matched_spread_ge2": ge2}
        for scope in scopes:
            self.markets[scope].add(market)
            self.totals[scope].update({"matched": len(rows), "spread_ge2": ge2})
        for r in rows:
            for scope in scopes:
                d = self.days.setdefault((scope, r["day"]), {"markets": set(), "matched": 0, "spread_ge2": 0, "called": 0, "qualifying": 0})
                d["markets"].add(market)
                d["matched"] += 1
                d["spread_ge2"] += r["spread"] in ("2", ">=3")
                if r["call"] is None:
                    continue
                d["called"] += 1
                d["qualifying"] += r[f"m{HORIZON}"] is not None
                for price in PRICES if r["cheap"] else ("all",):
                    half = self.halves[(scope, price)].label(r["ts"])  # once per row and group, in row order
                    for spread in ("all", r["spread"]) if r["spread"] is not None else ("all",):
                        for h in ("all", half):
                            items[(scope, price, spread, h)].append((r["day"], r[f"m{HORIZON}"]))
        for key, its in items.items():
            self.agg.add_market(key, market, its)

    def cut_rows(self, scopes: Sequence[str]) -> list[dict[str, Any]]:
        return [{"scope": scope, "price": price, "spread": spread, "half": half, "horizon_s": HORIZON,
                 **self.agg.result((scope, price, spread, half)), "label": LABEL}
                for scope in scopes for price in PRICES for spread in SPREADS for half in HALVES]

    def day_rows(self, scopes: Sequence[str]) -> list[dict[str, Any]]:
        return [{"scope": scope, "day": day, "markets": len(d["markets"]), "matched_prints": d["matched"], "matched_spread_ge2": d["spread_ge2"],
                 "called_prints": d["called"], "qualifying_300": d["qualifying"]}
                for (scope, day), d in sorted(self.days.items(), key=lambda kv: (SCOPES.index(kv[0][0]), kv[0][1])) if scope in scopes]


def run_markouts(tapes: Any, quotes: Any, eligible: Sequence[str], no_carry: set[str] | None, guard: RssGuard | None = None) -> EligibleMarkouts:
    """Two passes market by market (sorted ids), matching each market's prints twice instead of keeping them."""
    result = EligibleMarkouts()
    markets = sorted(eid for eid in eligible if eid in tapes.markets and eid in quotes.markets)

    def scopes(eid: str) -> tuple[str, ...]:
        return ("eligible", "no_carry") if no_carry is not None and eid in no_carry else ("eligible",)

    for eid in markets:
        result.pass1(scopes(eid), mq.match_market(eid, tapes.markets[eid], quotes.markets[eid]))
        if guard is not None:
            guard.check(f"pass 1 {eid}")
    result.finish_pass1()
    if guard is not None:
        guard.stage("pass 1 (eligible markets)")
    for eid in markets:
        result.pass2(eid, scopes(eid), mq.match_market(eid, tapes.markets[eid], quotes.markets[eid]))
        if guard is not None:
            guard.check(f"pass 2 {eid}")
    if guard is not None:
        guard.stage("pass 2 (aggregates)")
    return result


def _quote_dict(q: Any) -> dict[str, Any] | None:
    return None if q is None else {**asdict(q), "mid": q.mid, "spread": q.spread}


# ---------------------------------------------------------------------- run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only: quoter eligibility of the maker_markout_quotes markets, and their markouts (HYPOTHETICAL, 0 SIG reads).")
    parser.add_argument("--yes", action="store_true", help="run the one oracle pass (public Polymarket / Kalshi reads); without it nothing is fetched")
    parser.add_argument("--tape", type=Path, nargs="*", default=None, help="tape raw.jsonl files (default: as maker_markout_quotes)")
    parser.add_argument("--snapshots", type=Path, nargs="*", default=None, help="snapshots.jsonl files (default: every quote_snapshots day)")
    parser.add_argument("--positions", type=Path, default=None, help="position_review raw.jsonl for carry (default: the newest)")
    parser.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB, help=f"stop cleanly above this peak RSS (default {DEFAULT_MAX_RSS_MB:g})")
    args = parser.parse_args(argv)

    from config import LADDER_ENABLED, load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    tape_paths = args.tape or maker.default_raws()
    snap_paths = args.snapshots or sorted((OUT_ROOT / "quote_snapshots").glob("*/snapshots.jsonl"))
    if not tape_paths or not snap_paths:
        logger.error("Need both a saved tape and quote snapshots", extra={"tape": [str(p) for p in tape_paths], "snapshots": [str(p) for p in snap_paths]})
        return 2
    guard = RssGuard(args.max_rss_mb)
    try:
        tapes = maker.stream_tapes(tape_paths, guard)
        quotes = mq.stream_snapshots(snap_paths, keep=set(tapes.markets), guard=guard)
        guard.stage("inputs loaded")
        markets = sorted(set(tapes.markets) & set(quotes.markets))
        market_map = load_market_map()
        ladder = ladder_markets(iter_jsonl(LADDER_JOURNAL))
        carry, carry_source = saved_carry(args.positions or newest_review())
        candidates = [market_map.get(eid) for eid in markets]
        candidates = [m for m in candidates if m is not None and m.party in ("D", "R") and m.sig_exchange_id not in ladder]
        tokens = sorted({m.polymarket_token for m in candidates if m.polymarket_token})
        tickers = sorted({m.kalshi_ticker for m in candidates if m.kalshi_ticker})
        logger.info("Inputs", extra={"markets_in_both": len(markets), "d_r_candidates": len(candidates), "polymarket_tokens": len(tokens),
                                     "kalshi_tickers": len(tickers), "sig_reads": 0, "ladder_enabled": LADDER_ENABLED, "ladder_markets": sorted(ladder),
                                     "carry_source": carry_source or "none saved: carry unknown"})
        if not args.yes:
            logger.info("Nothing fetched: rerun with --yes for the one oracle pass (public Polymarket / Kalshi reads, 0 SIG reads)")
            return 0
        pm, ks = asyncio.run(fetch_oracles(tokens, tickers))
        now = datetime.now(timezone.utc).timestamp()
        out = run_dir(SCRIPT)
        write_json(out / "oracle_quotes.json", {"fetched_at": datetime.now(timezone.utc).isoformat(), "max_age_s": settings.oracle_staleness_threshold_s,
                                                "polymarket": {k: _quote_dict(v) for k, v in pm.items()}, "kalshi": {k: _quote_dict(v) for k, v in ks.items()}})
        rows = eligibility_rows(markets, market_map, ladder, carry, pm, ks, settings.oracle_staleness_threshold_s, quotes, now)
        eligible = [r["exchange_id"] for r in rows if r["eligible"]]
        no_carry = None if carry is None else {r["exchange_id"] for r in rows if r["eligible"] and r["carry"] == ""}
        scopes = SCOPES if no_carry is not None else ("eligible",)
        result = run_markouts(tapes, quotes, eligible, no_carry, guard)
        for r in rows:
            r.update(result.per_market.get(r["exchange_id"], {}))
        counts = table_counts(rows)
        cuts, days = result.cut_rows(scopes), result.day_rows(scopes)
        n_markets = {"eligible": len(eligible), "no_carry": None if no_carry is None else len(no_carry)}
        verdicts = {scope: {"eligible_markets": n_markets[scope], "markets_with_matched_prints": len(result.markets[scope]),
                            "matched_prints": result.totals[scope]["matched"], "matched_spread_ge2": result.totals[scope]["spread_ge2"],
                            "result": feasibility(n_markets[scope], result.totals[scope]["spread_ge2"])} for scope in scopes}
        write_csv(out / "eligibility.csv", rows, TABLE_FIELDS)
        write_csv(out / "by_cut.csv", cuts, CUT_FIELDS)
        write_csv(out / "by_day.csv", days, DAY_FIELDS)
        write_json(out / "summary.json", {"label": LABEL, "counts": counts, "feasibility": verdicts, "carry_source": carry_source,
                                          "ladder_enabled": LADDER_ENABLED, "ladder_markets": sorted(ladder), "cuts_reported": len(cuts),
                                          "thresholds": {"fair": [FAIR_LOW, FAIR_HIGH], "max_u": MAX_U, "min_markets": MIN_MARKETS, "min_prints_spread_ge2": MIN_PRINTS_SPREAD2},
                                          "not_modelled": ["queue position", "settlement risk (0 / 1)", "fees"]})
        guard.stage("done")
    except MemoryLimitExceeded as exc:
        logger.error("Memory limit reached: stopped cleanly before the OS would kill the process - raise --max-rss-mb only if the machine has room",
                     extra={"stage": exc.stage, "peak_rss_mb": round(exc.peak_mb, 1), "limit_mb": exc.limit_mb})
        return EXIT_MEMORY
    for r in rows:
        logger.info("Market", extra={k: r[k] for k in ("exchange_id", "race", "party", "fair", "u", "carry", "ladder", "eligible", "reason",
                                                       "matched_prints", "matched_spread_ge2", "last_spread_ticks")})
    logger.info("Eligibility counts", extra=counts)
    for d in days:
        logger.info("Per UTC day", extra=d)
    for c in cuts:
        if c["price"] == "all" and c["prints"]:
            logger.info("Markout 300 s vs quote mid (HYPOTHETICAL)", extra={k: c[k] for k in ("scope", "spread", "half", "qualifying", "markets", "mean_ticks", "ci_low_ticks", "ci_high_ticks")})
    for scope, v in verdicts.items():
        logger.info("Feasibility", extra={"scope": scope, **v})
    logger.info(LABEL)
    logger.info("Quoter eligibility done", extra={"out_dir": str(out), "cuts_reported": len(cuts), "peak_rss_mb": None if guard.peak_mb is None else round(guard.peak_mb, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
