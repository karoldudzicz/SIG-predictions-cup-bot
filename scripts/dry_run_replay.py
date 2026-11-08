"""READ-ONLY research: HYPOTHETICAL P&L of the taker's dry-run intents vs its live orders.

Every taker order journaled in data/order_journal.jsonl since 2026-10-01 16:00Z (latest
state per idempotency key, dated by its first entry) falls into:
  - dry_run: never sent. The bot repeats an unfilled intent every cycle, so repeats of
    the same (exchange, side, action, limit) are collapsed: a repeat within 10 min of the
    cluster's FIRST entry is dropped (anchored, the counting rule). Sensitivity: "chained"
    clusters, where a repeat within 10 min of the PREVIOUS one extends the cluster - an
    intent repeated for hours is then one fill, which is what a filled order would have
    done (the target is met, the bot stops). Each cluster is ASSUMED to fill its first
    entry's quantity at its limit. HYPOTHETICAL.
  - live: accepted by the engine. The journal keeps the CANONICAL order (action, side,
    price, quantityTraded) but no fill price, so a live fill is also valued at its limit
    (canonical price, the traded side's terms): a taker fill is at the limit or better, so
    this is a lower bound for buys. Only quantity traded AT PLACEMENT is known; a resting
    remainder that filled later is not in the journal (counted, not valued).
P&L = shares x (current SIG mid - limit) for a buy (reversed for a sell), in the traded
side's terms (NO mid = 1 - YES mid). Marked at ONE mid read now, so it includes every
price move since the order - dry-run (early) and live (later) orders are marked over
different holding periods. Per UTC day of the order, per race, and first vs second half (time order).

Calibration (all HYPOTHETICAL, marked at one mid): the edge the bot CLAIMED is parsed from
each order's journal reason ("edge 0.069") and binned (<0.02, 0.02-0.04, 0.04-0.07,
>=0.07). Per bin: fills, shares, races, share-weighted mean markout per share vs the mid,
share of winners, and a t-statistic computed on PER-RACE averages (fills inside one race
share its outcome, so races - not fills - are the observations). Dry-run and live apart,
for all fills and for the first vs second half of each sample (time order). The same table
by entry price bucket (<0.15, 0.15-0.85, >0.85, traded side's terms), by leg (YES / NO)
and by mapped oracle source (market_map.json: polymarket / kalshi / both / none - the
journal does not say which quote priced an order). Everything is computed twice: all
contracts, and EXCLUDING --exclude (default senate:RI, senate:FL, senate:DE, senate:OK),
side by side, so concentration is visible. If realized markout does not rise with the
claimed edge, the summary says so. Per-race table sorted by P&L with fills, share-weighted
entry price, and the share of the loss carried by the 5 worst races.

Mids: --yes reads them now (GET /tournaments/{slug} + GET /exchanges/prices, 100 ids per
read, through research.reader). Without --yes NOTHING is read: the newest
/exchanges/prices responses already saved in data/edge_scan/dry_run_replay/*/raw.jsonl
(or --raw) are reused, and their time is reported as the mark time; with no saved read
the counts are written and the P&L columns stay empty.
-> data/edge_scan/dry_run_replay/<UTC date>/: dry_run_clusters.csv, live_orders.csv,
   by_day.csv, by_race.csv, calibration.csv, race_table.csv, summary.json, summary.md.

    python scripts/dry_run_replay.py [--yes | --raw PATH] [--since 2026-10-01T16:00:00Z] [--window-min 10]
        [--exclude senate:RI,senate:FL,senate:DE,senate:OK]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from logging_config import setup_logging  # noqa: E402
from market_mapper import DATA_DIR, MarketMap  # noqa: E402
from research import stats  # noqa: E402
from research.reader import OUT_ROOT, ReadEstimate, add_research_args, confirm_fetch, read_jsonl, research_reader, resolve_reads_per_min, run_dir, write_csv, write_json  # noqa: E402

logger = logging.getLogger("dry_run_replay")

SCRIPT = "dry_run_replay"
DEFAULT_SINCE = "2026-10-01T16:00:00Z"
DEFAULT_WINDOW_MIN = 10.0
EXPECTED_LIVE_ACCEPTED = 202  # the count Karol expects; a mismatch is logged, not fatal
ORDER_JOURNAL_PATH = DATA_DIR / "order_journal.jsonl"
PRICES_CHUNK = 100
TICK = 0.005
LABEL = "HYPOTHETICAL: dry-run intents assumed filled at their limit; live orders valued at their limit (journal has no fill price)"
MARK_LABEL = "HYPOTHETICAL, marked at one current mid"
DEFAULT_EXCLUDE = ("senate:RI", "senate:FL", "senate:DE", "senate:OK")
WORST_RACES = 5
DIMENSIONS = (
    ("claimed_edge", lambda f: stats.edge_bin(f["edge"]), tuple(label for _, label in stats.EDGE_BINS)),
    ("entry_price", lambda f: stats.price_bucket(f["entry"]), tuple(label for _, label in stats.PRICE_BUCKETS)),
    ("leg", lambda f: f["leg"], ("yes", "no")),
    ("oracle_source", lambda f: f["oracle_source"], ("polymarket", "kalshi", "both", "none")),
)

CLUSTER_FIELDS = ("ts", "day", "exchange_id", "race", "party", "oracle_source", "kind", "edge_claimed", "side", "action", "limit", "quantity", "repeats",
                  "repeats_chained_cluster", "mid_yes", "mark_side", "pnl", "pnl_ticks_per_share")
LIVE_FIELDS = ("ts", "day", "exchange_id", "race", "party", "oracle_source", "kind", "edge_claimed", "sent_side", "sent_action", "side", "action", "limit",
               "quantity_traded", "remaining", "mid_yes", "mark_side", "pnl", "pnl_ticks_per_share")
CALIBRATION_FIELDS = ("variant", "sample", "half", "dimension", "group", "fills", "shares", "races", "mean_markout", "mean_markout_ticks", "winners_share",
                      "t_race_clustered")
RACE_TABLE_FIELDS = ("sample", "rank", "race", "fills", "shares", "mean_entry", "pnl", "pnl_ticks_per_share", "share_of_net_loss")
GROUP_FIELDS = ("source", "group", "fills", "shares", "cost", "pnl", "unvalued", "pnl_ticks_per_share")


def _ts(raw: Any) -> datetime:
    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def latest_by_key(entries: Sequence[Mapping[str, Any]], since: datetime) -> list[dict[str, Any]]:
    """Latest merged state per idempotency key whose FIRST entry is at or after `since`."""
    first: dict[str, datetime] = {}
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = entry.get("key")
        if key is None or not entry.get("ts"):
            continue
        first.setdefault(key, _ts(entry["ts"]))
        latest[key] = {**latest.get(key, {}), **entry}
    return [{**latest[k], "_first": first[k]} for k in latest if first[k] >= since]


def oracle_source(market_map: MarketMap, eid: str) -> str:
    """Which oracle(s) the market map links to this exchange (not which priced an order)."""
    m = market_map.get(eid)
    pm, ks = bool(m and m.polymarket_token), bool(m and m.kalshi_ticker)
    return "both" if pm and ks else "polymarket" if pm else "kalshi" if ks else "none"


def _label(market_map: MarketMap, eid: str) -> dict[str, Any]:
    m = market_map.get(eid)
    return {"race": m.race.key if m else "unmapped", "party": m.party if m else None, "oracle_source": oracle_source(market_map, eid)}


def _mark(mid_yes: float | None, side: str | None) -> float | None:
    if mid_yes is None:
        return None
    return mid_yes if side == "yes" else 1.0 - mid_yes


def _ticks(pnl: float | None, shares: float) -> float | None:
    return None if pnl is None or not shares else round(pnl / shares / TICK, 2)


def dry_run_clusters(orders: Sequence[Mapping[str, Any]], mids: Mapping[str, float | None], market_map: MarketMap, window_s: float) -> list[dict[str, Any]]:
    dry = [o for o in orders if o.get("status") == "dry_run" and (o.get("payload") or {}).get("price") is not None]

    def key(o: Mapping[str, Any]) -> tuple[Any, ...]:
        p = o["payload"]
        return (str(p.get("exchangeId")), p.get("side"), p.get("action"), round(float(p["price"]), 4))

    def when(o: Mapping[str, Any]) -> float:
        return o["_first"].timestamp()

    # Per key, the chained clusters as [start, last] spans: each anchored cluster is labelled
    # with the size of the chained cluster it falls in.
    spans: dict[tuple[Any, ...], list[tuple[float, float, int]]] = defaultdict(list)
    for c in stats.dedupe_repeats(dry, key, when, window_s, chained=True):
        spans[key(c["row"])].append((c["start"], c["last"], c["repeats"]))

    def chained_size(o: Mapping[str, Any]) -> int | None:
        t = when(o)
        return next((n for start, last, n in spans[key(o)] if start <= t <= last), None)

    rows = []
    for cluster in stats.dedupe_repeats(dry, key, when, window_s):
        o = cluster["row"]
        p = o["payload"]
        eid, side, action = str(p.get("exchangeId")), p.get("side"), p.get("action")
        limit, quantity = float(p["price"]), abs(float(p.get("quantity") or 0))
        mid = mids.get(eid)
        mark = _mark(mid, side)
        pnl = stats.pnl_vs_mark(action, quantity, limit, mark)
        rows.append({"ts": o["_first"].isoformat(), "day": o["_first"].strftime("%Y-%m-%d"), "exchange_id": eid, **_label(market_map, eid), "kind": o.get("kind"),
                     "edge_claimed": stats.parse_claimed_edge(o.get("reason")),
                     "side": side, "action": action, "limit": limit, "quantity": quantity, "repeats": cluster["repeats"],
                     "repeats_chained_cluster": chained_size(o), "mid_yes": mid, "mark_side": None if mark is None else round(mark, 4),
                     "pnl": None if pnl is None else round(pnl, 2), "pnl_ticks_per_share": _ticks(pnl, quantity)})
    return rows


def chained_count(orders: Sequence[Mapping[str, Any]], window_s: float) -> int:
    dry = [o for o in orders if o.get("status") == "dry_run" and (o.get("payload") or {}).get("price") is not None]
    return len(stats.dedupe_repeats(dry, lambda o: (str(o["payload"].get("exchangeId")), o["payload"].get("side"), o["payload"].get("action"),
                                                    round(float(o["payload"]["price"]), 4)), lambda o: o["_first"].timestamp(), window_s, chained=True))


def live_rows(orders: Sequence[Mapping[str, Any]], mids: Mapping[str, float | None], market_map: MarketMap) -> list[dict[str, Any]]:
    """Accepted orders, valued from the CANONICAL response (never the payload)."""
    rows = []
    for o in sorted((o for o in orders if o.get("status") == "accepted"), key=lambda o: o["_first"]):
        result = o.get("result") if isinstance(o.get("result"), dict) else {}
        payload = o.get("payload") or {}
        eid = str(payload.get("exchangeId") or result.get("exchangeId"))
        side, action = result.get("side") or payload.get("side"), result.get("action") or payload.get("action")
        limit = result.get("price") if result.get("price") is not None else payload.get("price")
        traded = abs(float(result.get("quantityTraded") or 0))
        mid = mids.get(eid)
        mark = _mark(mid, side)
        pnl = None if limit is None or not traded else stats.pnl_vs_mark(action, traded, float(limit), mark)
        rows.append({"ts": o["_first"].isoformat(), "day": o["_first"].strftime("%Y-%m-%d"), "exchange_id": eid, **_label(market_map, eid), "kind": o.get("kind"),
                     "edge_claimed": stats.parse_claimed_edge(o.get("reason")), "sent_side": payload.get("side"), "sent_action": payload.get("action"), "side": side, "action": action,
                     "limit": None if limit is None else float(limit), "quantity_traded": traded,
                     "remaining": None if result.get("remainingQuantity") is None else abs(float(result["remainingQuantity"])),
                     "mid_yes": mid, "mark_side": None if mark is None else round(mark, 4),
                     "pnl": None if pnl is None else round(pnl, 2), "pnl_ticks_per_share": _ticks(pnl, traded)})
    return rows


def group_rows(source: str, rows: Sequence[Mapping[str, Any]], by: str, shares_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r[shares_key]:
            groups[str(r[by])].append(r)
    out = []
    for group, members in sorted(groups.items()):
        valued = [m for m in members if m["pnl"] is not None]
        shares = math.fsum(m[shares_key] for m in valued)
        pnl = math.fsum(m["pnl"] for m in valued)
        out.append({"source": source, "group": group, "fills": len(members), "shares": shares,
                    "cost": round(math.fsum(m[shares_key] * m["limit"] for m in valued), 2), "pnl": round(pnl, 2) if valued else None,
                    "unvalued": len(members) - len(valued), "pnl_ticks_per_share": _ticks(pnl, shares) if valued else None})
    return out


def _totals(rows: Sequence[Mapping[str, Any]], shares_key: str) -> dict[str, Any]:
    filled = [r for r in rows if r[shares_key]]
    valued = [r for r in filled if r["pnl"] is not None]
    shares = math.fsum(r[shares_key] for r in valued)
    pnl = math.fsum(r["pnl"] for r in valued)
    return {"fills": len(filled), "valued": len(valued), "shares": shares, "pnl": round(pnl, 2) if valued else None,
            "pnl_ticks_per_share": _ticks(pnl, shares) if valued else None,
            "winners_share": round(sum(1 for r in valued if r["pnl"] > 0) / len(valued), 3) if valued else None}


def as_fills(rows: Sequence[Mapping[str, Any]], shares_key: str) -> list[dict[str, Any]]:
    """Valued rows in the shape research.stats.markout_table expects (cluster = race)."""
    return [{"ts": r["ts"], "race": r["race"], "cluster": r["race"], "shares": r[shares_key], "pnl": r["pnl"], "edge": r.get("edge_claimed"),
             "entry": r["limit"], "leg": r["side"], "oracle_source": r.get("oracle_source")}
            for r in rows if r[shares_key] and r["pnl"] is not None and r["limit"] is not None]


def calibration(samples: Mapping[str, Sequence[Mapping[str, Any]]], exclude: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(calibration.csv rows, verdicts): every dimension x sample x half, for all contracts
    and without `exclude`. Verdict per variant / sample / half: does mean markout rise
    with the claimed edge bin?"""
    rows: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {}
    excluded = set(exclude)
    for variant, keep in (("all", lambda f: True), ("excluding", lambda f: f["race"] not in excluded)):
        for sample, fills in samples.items():
            kept = [f for f in fills if keep(f)]
            first, second = stats.split_halves(kept, key=lambda f: f["ts"])
            for half, part in (("all", kept), ("first_half", first), ("second_half", second)):
                for dimension, group, order in DIMENSIONS:
                    for r in stats.markout_table(part, group, order):
                        rows.append({"variant": variant, "sample": sample, "half": half, "dimension": dimension, "group": r["group"], "fills": r["fills"],
                                     "shares": r["shares"], "races": r["clusters"],
                                     "mean_markout": None if r["mean_markout"] is None else round(r["mean_markout"], 5),
                                     "mean_markout_ticks": None if r["mean_markout"] is None else round(r["mean_markout"] / TICK, 2),
                                     "winners_share": r["winners_share"], "t_race_clustered": None if r["t_clustered"] is None else round(r["t_clustered"], 2)})
                means = [r["mean_markout"] for r in rows if (r["variant"], r["sample"], r["half"], r["dimension"]) == (variant, sample, half, "claimed_edge")]
                verdicts.setdefault(variant, {}).setdefault(sample, {})[half] = stats.rises(means)
    return rows, verdicts


def race_table(samples: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per sample: races sorted by P&L (worst first) with fills, shares, share-weighted entry,
    P&L and each race's share of the net loss; plus the 5-worst-races concentration."""
    rows: list[dict[str, Any]] = []
    concentration = {}
    for sample, fills in samples.items():
        by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for f in fills:
            by_race[f["race"]].append(f)
        pnl = {race: math.fsum(f["pnl"] for f in fs) for race, fs in by_race.items()}
        conc = stats.loss_concentration(pnl, WORST_RACES)
        concentration[sample] = conc
        for rank, (race, value) in enumerate(sorted(pnl.items(), key=lambda kv: kv[1]), start=1):
            fs = by_race[race]
            shares = math.fsum(f["shares"] for f in fs)
            rows.append({"sample": sample, "rank": rank, "race": race, "fills": len(fs), "shares": shares,
                         "mean_entry": round(math.fsum(f["entry"] * f["shares"] for f in fs) / shares, 4) if shares else None,
                         "pnl": round(value, 2), "pnl_ticks_per_share": _ticks(value, shares),
                         "share_of_net_loss": round(value / conc["net_pnl"], 4) if conc["net_pnl"] < 0 else None})
    return rows, concentration


def mids_from_raw(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float | None], str | None]:
    """YES mids from the /exchanges/prices responses of a raw.jsonl (a later read of an
    exchange wins) and the time of the newest one used."""
    mids: dict[str, float | None] = {}
    mark_ts = None
    for rec in records:
        if rec.get("path") != "/exchanges/prices" or "body" not in rec:
            continue
        for row in (rec["body"] or {}).get("data") or []:
            bid, ask = row.get("bestBid"), row.get("bestAsk")
            mids[str(row["exchangeId"])] = None if bid is None or ask is None else round((float(bid) + float(ask)) / 2.0, 4)
        mark_ts = rec.get("ts") or mark_ts
    return mids, mark_ts


def newest_raw(root: Path | None = None) -> Path | None:
    runs = sorted((root or OUT_ROOT / SCRIPT).glob("*/raw.jsonl"))
    return runs[-1] if runs else None


def summarise(orders: Sequence[Mapping[str, Any]], clusters: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]], window_s: float) -> dict[str, Any]:
    statuses: dict[str, int] = defaultdict(int)
    for o in orders:
        statuses[str(o.get("status"))] += 1
    halves = {}
    for name, rows, shares_key in (("dry_run", clusters, "quantity"), ("live", live, "quantity_traded")):
        first, second = stats.split_halves([r for r in rows if r[shares_key]], key=lambda r: r["ts"])
        halves[name] = {"first_half": _totals(first, shares_key), "second_half": _totals(second, shares_key)}
    return {
        "label": LABEL,
        "orders_by_final_status": dict(sorted(statuses.items())),
        "dry_run": {"journal_entries": statuses.get("dry_run", 0), "clusters_anchored": len(clusters), "clusters_chained": chained_count(orders, window_s),
                    "window_min": window_s / 60, **_totals(clusters, "quantity")},
        "live": {"accepted": len(live), "expected_accepted": EXPECTED_LIVE_ACCEPTED, "filled_at_placement": sum(1 for r in live if r["quantity_traded"]),
                 "with_resting_remainder": sum(1 for r in live if r["remaining"]), **_totals(live, "quantity_traded")},
        "halves": halves,
        "caveats": ["marked at one current mid: includes every move since the order; dry-run (early) and live (later) orders have different holding periods",
                    "t-statistics are computed on per-race averages: fills inside one race are not independent",
                    "oracle_source is the oracle(s) the market map links to an exchange, not the quote that priced a given order",
                    "live fills valued at their limit: taker fills are at the limit or better, so buys are a lower bound",
                    "fills of a live order's resting remainder after placement are not in the journal",
                    "chained clusters (repeats within 10 min of the previous one) count an intent repeated for hours once - a sensitivity check on the anchored count"],
    }


def summary_markdown(summary: Mapping[str, Any]) -> str:
    d, l = summary["dry_run"], summary["live"]
    return "\n".join([
        "# Dry-run replay (HYPOTHETICAL)", "", f"_{summary['label']}._", "",
        f"- Dry-run: {d['journal_entries']} journal entries -> {d['clusters_anchored']} intents after the {d['window_min']:g}-min dedupe "
        f"({d['clusters_chained']} if chained). Hypothetical P&L at the current mid: {d['pnl']} SUSQ over {d['shares']:g} shares ({d['pnl_ticks_per_share']} ticks/share).",
        f"- Live: {l['accepted']} accepted (expected {l['expected_accepted']}), {l['filled_at_placement']} filled at placement. "
        f"P&L at the same mid, valued at the limit: {l['pnl']} SUSQ over {l['shares']:g} shares ({l['pnl_ticks_per_share']} ticks/share).",
        *[f"- Caveat: {c}." for c in summary["caveats"]], "",
        *calibration_markdown(summary.get("calibration") or {}),
    ])


def calibration_markdown(cal: Mapping[str, Any]) -> list[str]:
    """Claimed-edge bins, all contracts vs excluding the concentrated races, side by side."""
    if not cal:
        return []
    rows = cal.get("rows") or []
    lines = [f"## Calibration by claimed edge ({MARK_LABEL}; t on per-race averages)", "",
             f"Excluded in the right-hand columns: {', '.join(cal.get('excluded') or [])}.", ""]
    for sample in ("dry_run", "live"):
        lines += [f"### {sample}", "", "| claimed edge | fills (all) | ticks/share (all) | t (all) | fills (excl.) | ticks/share (excl.) | t (excl.) |",
                  "|---|---|---|---|---|---|---|"]
        for _, label in stats.EDGE_BINS:
            cell = {}
            for variant in ("all", "excluding"):
                r = next((x for x in rows if (x["variant"], x["sample"], x["half"], x["dimension"], x["group"]) == (variant, sample, "all", "claimed_edge", label)), None)
                cell[variant] = (r["fills"], r["mean_markout_ticks"], r["t_race_clustered"]) if r else (0, None, None)
            lines.append(f"| {label} | {cell['all'][0]} | {cell['all'][1]} | {cell['all'][2]} | {cell['excluding'][0]} | {cell['excluding'][1]} | {cell['excluding'][2]} |")
        lines.append("")
        for variant in ("all", "excluding"):
            verdict = ((cal.get("verdicts") or {}).get(variant) or {}).get(sample) or {}
            for half in ("all", "first_half", "second_half"):
                v = verdict.get(half)
                if v is False:
                    lines.append(f"- {sample}, {variant}, {half}: realized markout does NOT rise with the claimed edge.")
                elif v is None:
                    lines.append(f"- {sample}, {variant}, {half}: too few edge bins with fills to judge.")
                else:
                    lines.append(f"- {sample}, {variant}, {half}: realized markout rises with the claimed edge.")
        lines.append("")
    for sample, conc in (cal.get("concentration") or {}).items():
        if not conc["worst"]:
            continue
        lines.append(f"- {sample}: the {WORST_RACES} worst races ({', '.join(conc['worst'])}) carry {conc['worst_pnl']:.2f} SUSQ = "
                     f"{conc['share_of_net_loss']} of the net loss, {conc['share_of_gross_losses']} of the gross losses.")
    lines.append("")
    return lines


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
    parser = argparse.ArgumentParser(description="Read-only: hypothetical P&L of dry-run intents vs live orders, marked at the current mid.")
    add_research_args(parser)
    parser.add_argument("--since", default=DEFAULT_SINCE, help=f"first journal entry at or after this (default {DEFAULT_SINCE})")
    parser.add_argument("--window-min", type=float, default=DEFAULT_WINDOW_MIN, help=f"dedupe window in minutes (default {DEFAULT_WINDOW_MIN:g})")
    parser.add_argument("--raw", type=Path, default=None, help="raw.jsonl whose saved price reads give the mids (default: the newest dry_run_replay run)")
    parser.add_argument("--exclude", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=list(DEFAULT_EXCLUDE),
                        help=f"races left out in the second variant (default {','.join(DEFAULT_EXCLUDE)})")
    args = parser.parse_args(argv)

    from config import load_settings  # late: --help works without a .env

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    try:
        reads_per_min = resolve_reads_per_min(args.reads_per_min)
    except ValueError as exc:
        parser.error(str(exc))
    if args.window_min <= 0:
        parser.error("--window-min must be > 0")
    orders = latest_by_key(read_jsonl(ORDER_JOURNAL_PATH), _ts(args.since))
    if not orders:
        logger.error("No journaled orders since the start", extra={"journal": str(ORDER_JOURNAL_PATH), "since": args.since})
        return 2
    ids = sorted({str((o.get("payload") or {}).get("exchangeId")) for o in orders if o.get("status") in ("dry_run", "accepted")})
    mids: dict[str, float | None] = {}
    mark_from, mark_ts = None, None
    if args.yes and confirm_fetch(ReadEstimate(1 + math.ceil(len(ids) / PRICES_CHUNK), reads_per_min), True):
        mids, mark_from, mark_ts = asyncio.run(_read_mids(ids, reads_per_min)), "read now", datetime.now(timezone.utc).isoformat()
    else:
        raw = args.raw or newest_raw()
        if raw is not None and raw.exists():
            mids, mark_ts = mids_from_raw(read_jsonl(raw))
            mark_from = str(raw)
            missing = [i for i in ids if i not in mids]
            logger.info("Offline: reusing saved price reads (0 new reads)", extra={"raw": mark_from, "mark_ts": mark_ts, "exchanges_without_mid": len(missing)})
        else:
            logger.info("Offline and no saved price read: counts only, P&L columns stay empty")
    market_map = MarketMap.load()
    window_s = args.window_min * 60
    clusters = dry_run_clusters(orders, mids, market_map, window_s)
    live = live_rows(orders, mids, market_map)
    summary = summarise(orders, clusters, live, window_s)
    samples = {"dry_run": as_fills(clusters, "quantity"), "live": as_fills(live, "quantity_traded")}
    cal_rows, verdicts = calibration(samples, args.exclude)
    race_rows, concentration = race_table(samples)
    summary["mark"] = {"label": MARK_LABEL, "from": mark_from, "at": mark_ts}
    summary["calibration"] = {"excluded": list(args.exclude), "verdicts": verdicts, "concentration": concentration, "rows": cal_rows}
    if summary["live"]["accepted"] != EXPECTED_LIVE_ACCEPTED:
        logger.warning("Live accepted count differs from the expected one", extra={"accepted": summary["live"]["accepted"], "expected": EXPECTED_LIVE_ACCEPTED})

    out = run_dir(SCRIPT)
    write_csv(out / "dry_run_clusters.csv", clusters, CLUSTER_FIELDS)
    write_csv(out / "live_orders.csv", live, LIVE_FIELDS)
    write_csv(out / "by_day.csv", group_rows("dry_run", clusters, "day", "quantity") + group_rows("live", live, "day", "quantity_traded"), GROUP_FIELDS)
    write_csv(out / "by_race.csv", group_rows("dry_run", clusters, "race", "quantity") + group_rows("live", live, "race", "quantity_traded"), GROUP_FIELDS)
    write_csv(out / "calibration.csv", cal_rows, CALIBRATION_FIELDS)
    write_csv(out / "race_table.csv", race_rows, RACE_TABLE_FIELDS)
    write_json(out / "summary.json", {**summary, "calibration": {k: v for k, v in summary["calibration"].items() if k != "rows"}})
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    logger.info("Dry-run replay (HYPOTHETICAL)", extra={"out_dir": str(out), "dry_run": summary["dry_run"], "live": summary["live"], "mark": summary["mark"]})
    logger.info("Calibration verdicts: does markout rise with claimed edge (HYPOTHETICAL)", extra={"verdicts": verdicts, "concentration": concentration})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
