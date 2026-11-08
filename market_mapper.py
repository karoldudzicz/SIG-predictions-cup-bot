"""Map SIG exchanges to their Polymarket / Kalshi equivalents.

Two phases:
1. BUILD (offline, `python market_mapper.py build`): parse every open SIG market
   into a structured race (office, state, district) + party, discover the matching
   Polymarket token and Kalshi ticker, cross-check that both venues roughly agree
   on price, and write everything to data/market_map.json for human review.
2. LOAD (runtime, `MarketMap.load()`): read the reviewed JSON, apply hand-written
   fixes from data/market_map_overrides.json, and answer "which oracle refs belong
   to this SIG exchange?".

Matching is deterministic (title regexes + known slug/ticker conventions), not
fuzzy text similarity: a wrong mapping is worse than a missing one, so anything
uncertain is left unmatched or flagged in `notes` for review.

Overrides file format (only the keys you set are replaced; null removes a match):
    {"1077": {"polymarket_token": "5516...", "kalshi_ticker": null}}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from config import PROJECT_ROOT
from exchange_api import ExchangeClient
from oracle_api import KalshiClient, OracleError, OracleQuote, PolymarketClient

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
MARKET_MAP_PATH = DATA_DIR / "market_map.json"
OVERRIDES_PATH = DATA_DIR / "market_map_overrides.json"

DISCOVERY_CONCURRENCY = 5
# If both venues are matched but their mids differ by more than this, the mapping
# is probably wrong (or one venue is illiquid) - flag it for review.
VENUE_DISAGREEMENT_THRESHOLD = 0.10
# A Kalshi event without a "2026" label is still the 2026 race if it settles before
# this (Nov 2026 vote, January 2027 inauguration). ISO strings compare correctly.
KALSHI_2026_CYCLE_END = "2027-07-01"

Office = Literal["senate", "governor", "house", "senate_control", "house_control"]
Party = Literal["D", "R", "I"]
Note = tuple["Party | None", str]  # (party the note is about, or None for the whole race; text)

US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
_STATE_NAMES = {code: name for name, code in US_STATES.items()}

_PARTY_WORDS: dict[str, Party] = {"Republican": "R", "Democratic": "D", "Independent": "I"}
_SIG_TITLE_RE = re.compile(r"^Will the (Republican|Democratic|Independent) Party win the (.+)\?$")
_HOUSE_RACE_RE = re.compile(r"^([A-Z]{2})-(\d{2}) House race$")
_STATE_RACE_RE = re.compile(r"^(.+) (Senate|Governor)$")


# ---------------------------------------------------------------------- data model


@dataclass(frozen=True)
class Race:
    office: Office
    state: str | None = None  # two-letter code
    district: str | None = None  # two digits, House only

    @property
    def key(self) -> str:
        """Stable id shared by every party-market of the same race, e.g. 'house:NH-01'."""
        if self.office == "house":
            return f"house:{self.state}-{self.district}"
        if self.state:
            return f"{self.office}:{self.state}"
        return self.office


@dataclass(frozen=True)
class MarketMapping:
    sig_exchange_id: str
    sig_market_id: str
    title: str
    race: Race
    party: Party
    polymarket_token: str | None
    kalshi_ticker: str | None
    notes: tuple[str, ...] = field(default=())


def parse_sig_title(title: str) -> tuple[Race, Party] | None:
    """'Will the Republican Party win the NH-01 House race?' -> (Race(house, NH, 01), 'R')."""
    match = _SIG_TITLE_RE.match(title.strip())
    if not match:
        return None
    party = _PARTY_WORDS[match.group(1)]
    place = match.group(2)

    if place == "U.S. Senate":
        return Race("senate_control"), party
    if place == "U.S. House":
        return Race("house_control"), party
    if house := _HOUSE_RACE_RE.match(place):
        return Race("house", house.group(1), house.group(2)), party
    if state_race := _STATE_RACE_RE.match(place):
        code = US_STATES.get(state_race.group(1))
        if code is None:
            return None
        return Race(state_race.group(2).lower(), code), party  # type: ignore[arg-type]
    return None


# ---------------------------------------------------------------------- runtime map


class MarketMap:
    """Read-only lookup of SIG exchange id -> oracle refs."""

    def __init__(self, mappings: dict[str, MarketMapping]) -> None:
        self._mappings = mappings

    @classmethod
    def load(cls, path: Path = MARKET_MAP_PATH, overrides_path: Path = OVERRIDES_PATH) -> MarketMap:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python market_mapper.py build` first")
        raw = json.loads(path.read_text(encoding="utf-8"))
        overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}

        mappings: dict[str, MarketMapping] = {}
        for exchange_id, entry in raw["entries"].items():
            polymarket_token = (entry.get("polymarket") or {}).get("token_id")
            kalshi_ticker = (entry.get("kalshi") or {}).get("ticker")
            notes = list(entry.get("notes") or [])
            if exchange_id in overrides:
                override = overrides[exchange_id]
                if "polymarket_token" in override:
                    polymarket_token = override["polymarket_token"]
                if "kalshi_ticker" in override:
                    kalshi_ticker = override["kalshi_ticker"]
                notes.append("manual override applied")
            mappings[exchange_id] = MarketMapping(
                sig_exchange_id=exchange_id,
                sig_market_id=entry["sig_market_id"],
                title=entry["title"],
                race=Race(**entry["race"]),
                party=entry["party"],
                polymarket_token=polymarket_token,
                kalshi_ticker=kalshi_ticker,
                notes=tuple(notes),
            )

        unknown = set(overrides) - set(mappings)
        if unknown:
            logger.warning("Overrides reference unknown SIG exchanges", extra={"exchange_ids": sorted(unknown)})
        logger.info(
            "Market map loaded",
            extra={"path": str(path), "entries": len(mappings), "overrides": len(overrides), "built_at": raw.get("generated_at")},
        )
        return cls(mappings)

    def get(self, sig_exchange_id: str) -> MarketMapping | None:
        return self._mappings.get(str(sig_exchange_id))

    def __iter__(self) -> Iterator[MarketMapping]:
        return iter(self._mappings.values())

    def __len__(self) -> int:
        return len(self._mappings)

    def by_race(self) -> dict[str, list[MarketMapping]]:
        """Group party-markets of the same race (e.g. D and R for 'senate:RI')."""
        groups: dict[str, list[MarketMapping]] = defaultdict(list)
        for mapping in self._mappings.values():
            groups[mapping.race.key].append(mapping)
        return dict(groups)

    def polymarket_tokens(self) -> list[str]:
        return sorted({m.polymarket_token for m in self._mappings.values() if m.polymarket_token})

    def kalshi_tickers(self) -> list[str]:
        return sorted({m.kalshi_ticker for m in self._mappings.values() if m.kalshi_ticker})


# ---------------------------------------------------------------------- discovery: Polymarket


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _polymarket_slug_candidates(race: Race) -> list[str]:
    if race.office == "senate_control":
        return ["which-party-will-win-the-senate-in-2026"]
    if race.office == "house_control":
        return ["which-party-will-win-the-house-in-2026"]
    if race.office == "house":
        return [f"{race.state.lower()}-{race.district}-house-election-winner"]
    state_slug = _STATE_NAMES[race.state].lower().replace(" ", "-")
    if race.office == "senate":
        return [f"{state_slug}-senate-election-winner"]
    return [
        f"{state_slug}-governor-winner-2026",
        f"{state_slug}-governor-election-winner",
        f"{state_slug}-governor-election-2026",
    ]


def _polymarket_expected_title(race: Race) -> str | None:
    if race.office == "house":
        return f"{race.state}-{race.district} House Election Winner"
    if race.office in ("senate", "governor"):
        return f"{_STATE_NAMES[race.state]} {race.office.capitalize()} Election Winner"
    return None


def _polymarket_party(market: dict[str, Any]) -> Party | None:
    label = (market.get("groupItemTitle") or "").strip()
    suffix = re.search(r"\(([DRI])\)$", label)
    if suffix:
        return suffix.group(1)  # type: ignore[return-value]
    text = _norm(f"{label} {market.get('question')}")
    hits = {party for word, party in (("democrat", "D"), ("republican", "R"), ("independent", "I")) if word in text}
    return hits.pop() if len(hits) == 1 else None


async def _find_polymarket_event(poly: PolymarketClient, race: Race) -> dict[str, Any] | None:
    for slug in _polymarket_slug_candidates(race):
        event = await poly.get_event(slug)
        if event and not event.get("closed"):
            return event
    expected = _polymarket_expected_title(race)
    if expected:
        for event in await poly.search_events(expected, limit=5):
            if _norm(event.get("title")) == _norm(expected) and not event.get("closed"):
                return await poly.get_event(event["slug"])  # search results may carry partial markets
    return None


async def _discover_polymarket(poly: PolymarketClient, race: Race) -> tuple[dict[Party, dict[str, Any]], list[Note]]:
    """Return party -> {event_slug, label, token_id}, plus review notes."""
    event = await _find_polymarket_event(poly, race)
    if event is None:
        return {}, [(None, "polymarket: no open event found")]

    by_party: dict[Party, list[dict[str, Any]]] = defaultdict(list)
    for market in event.get("markets") or []:
        if market.get("closed") or not market.get("active", True):
            continue
        party = _polymarket_party(market)
        token = PolymarketClient.outcome_tokens(market).get("Yes")
        if party and token:
            by_party[party].append({"event_slug": event["slug"], "label": market.get("groupItemTitle"), "token_id": token})

    matches: dict[Party, dict[str, Any]] = {}
    notes: list[Note] = []
    for party, candidates in by_party.items():
        if len(candidates) == 1:
            matches[party] = candidates[0]
        else:
            notes.append((party, f"polymarket: {len(candidates)} candidates for party {party} in {event['slug']}, left unmatched"))
    return matches, notes


# ---------------------------------------------------------------------- discovery: Kalshi


def _kalshi_series_candidates(race: Race) -> list[str]:
    if race.office == "senate_control":
        return ["CONTROLS"]
    if race.office == "house_control":
        return ["CONTROLH"]
    if race.office == "house":
        district = int(race.district)
        return [f"HOUSE{race.state}{district}", f"KXHOUSE{race.state}{district}"]
    if race.office == "senate":
        # SENATE{ST}S = special election (e.g. Florida 2026); SENATE{ST} may be the 2028 seat.
        return [f"SENATE{race.state}", f"SENATE{race.state}S", f"SENATEPARTY{race.state}", f"SENATEPARTY-{race.state}"]
    return [f"GOVPARTY{race.state}", f"KXGOVPARTY{race.state}"]


def _resolves_by_2026_cycle_end(event: dict[str, Any]) -> bool:
    """True if every market is expected to settle before mid-2027 (i.e. after a Nov 2026 vote + inauguration)."""
    expirations = [m.get("expected_expiration_time") for m in event.get("markets") or []]
    return bool(expirations) and all(exp and exp < KALSHI_2026_CYCLE_END for exp in expirations)


async def _find_kalshi_event(
    kalshi: KalshiClient, race: Race, known_series: set[str]
) -> tuple[dict[str, Any] | None, list[Note]]:
    # Event ticker years are unreliable (GOVPARTYNH-28 is the 2026 race, SENATEFL-28 is
    # the 2028 one), so accept an event only if it is labelled 2026 or settles by mid-2027.
    for series in _kalshi_series_candidates(race):
        if series not in known_series:
            continue
        try:
            events = (await kalshi.list_events(series_ticker=series, status="open")).get("events") or []
        except OracleError as exc:
            if exc.status == 404:
                continue
            raise
        for event in events:
            if "2026" in f"{event.get('sub_title')} {event.get('title')}":
                return event, []
        for event in events:
            if _resolves_by_2026_cycle_end(event):
                return event, [(None, f"kalshi: {event['event_ticker']} ({event.get('sub_title')}) accepted by settlement date, not by '2026' label")]
    return None, [(None, "kalshi: no open event found")]


async def _discover_kalshi(
    kalshi: KalshiClient, race: Race, known_series: set[str]
) -> tuple[dict[Party, dict[str, Any]], list[Note]]:
    """Return party -> {event_ticker, label, ticker}, plus review notes."""
    event, notes = await _find_kalshi_event(kalshi, race, known_series)
    if event is None:
        return {}, notes

    matches: dict[Party, dict[str, Any]] = {}
    others: list[dict[str, Any]] = []
    for market in event.get("markets") or []:
        suffix = market["ticker"][len(event["event_ticker"]) + 1 :]
        entry = {"event_ticker": event["event_ticker"], "label": market.get("yes_sub_title"), "ticker": market["ticker"]}
        if suffix in ("D", "R"):
            matches[suffix] = entry  # type: ignore[index]
        else:
            others.append(entry)

    # Independents are listed under a candidate code (e.g. -DOSB), not a party letter.
    if len(others) == 1:
        matches["I"] = others[0]
        notes.append(("I", f"kalshi: independent matched by elimination -> {others[0]['ticker']} ({others[0]['label']})"))
    elif others:
        notes.append(("I", f"kalshi: {len(others)} non-party candidates in {event['event_ticker']}, independent left unmatched"))
    return matches, notes


# ---------------------------------------------------------------------- build


def _reference_price(quote: OracleQuote | None) -> float | None:
    if quote is None:
        return None
    return quote.mid if quote.mid is not None else quote.last


async def build_market_map(
    exchange: ExchangeClient, poly: PolymarketClient, kalshi: KalshiClient
) -> dict[str, Any]:
    """Discover mappings for every open SIG market. Returns the JSON document."""
    parsed: dict[str, tuple[dict[str, Any], Race, Party]] = {}
    unparsed: list[str] = []
    async for market in exchange.iter_markets(status="open"):
        result = parse_sig_title(market["title"])
        if result is None or len(market.get("exchanges") or []) != 1:
            unparsed.append(f"{market['id']}: {market['title']}")
            continue
        race, party = result
        parsed[market["exchanges"][0]["id"]] = (market, race, party)

    races = sorted({race for _, race, _ in parsed.values()}, key=lambda r: r.key)
    logger.info("SIG markets parsed", extra={"markets": len(parsed), "races": len(races), "unparsed": len(unparsed)})

    # One call lists every Kalshi election series, so we only query tickers that exist.
    known_series = {series["ticker"] for series in await kalshi.list_series(category="Elections")}
    semaphore = asyncio.Semaphore(DISCOVERY_CONCURRENCY)

    async def discover(race: Race):
        async with semaphore:
            return race, await _discover_polymarket(poly, race), await _discover_kalshi(kalshi, race, known_series)

    discovered = {race: (pm, ks) for race, pm, ks in await asyncio.gather(*(discover(r) for r in races))}

    entries: dict[str, dict[str, Any]] = {}
    for exchange_id, (market, race, party) in parsed.items():
        (pm_matches, pm_notes), (ks_matches, ks_notes) = discovered[race]
        # Keep race-wide notes (party None) and notes about this entry's own party only.
        notes = [text for note_party, text in (*pm_notes, *ks_notes) if note_party in (None, party)]
        if party not in pm_matches and not any(t.startswith("polymarket: no open") for _, t in pm_notes):
            notes.append(f"polymarket: event found but no market for party {party}")
        if party not in ks_matches and not any(t.startswith("kalshi: no open") for _, t in ks_notes):
            notes.append(f"kalshi: event found but no market for party {party}")
        entries[exchange_id] = {
            "sig_market_id": market["id"],
            "title": market["title"],
            "race": asdict(race),
            "party": party,
            "polymarket": pm_matches.get(party),
            "kalshi": ks_matches.get(party),
            "notes": notes,
        }

    await _cross_check_prices(entries, poly, kalshi)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unparsed_sig_markets": unparsed,
        "entries": dict(sorted(entries.items(), key=lambda item: int(item[0]))),
    }


async def _cross_check_prices(entries: dict[str, dict[str, Any]], poly: PolymarketClient, kalshi: KalshiClient) -> None:
    """Attach reference prices and flag entries where the two venues disagree."""
    tokens = [e["polymarket"]["token_id"] for e in entries.values() if e["polymarket"]]
    tickers = [e["kalshi"]["ticker"] for e in entries.values() if e["kalshi"]]
    pm_quotes, ks_quotes = await asyncio.gather(poly.get_quotes(tokens), kalshi.get_quotes(tickers))

    for entry in entries.values():
        pm_price = _reference_price(pm_quotes.get(entry["polymarket"]["token_id"])) if entry["polymarket"] else None
        ks_price = _reference_price(ks_quotes.get(entry["kalshi"]["ticker"])) if entry["kalshi"] else None
        entry["check_prices"] = {"polymarket": pm_price, "kalshi": ks_price}
        if pm_price is not None and ks_price is not None and abs(pm_price - ks_price) > VENUE_DISAGREEMENT_THRESHOLD:
            entry["notes"].append(f"venues disagree: polymarket {pm_price:.3f} vs kalshi {ks_price:.3f} - verify mapping")


def write_market_map(document: dict[str, Any], path: Path = MARKET_MAP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written map


# ---------------------------------------------------------------------- CLI


async def _cli_build(log: logging.Logger) -> int:
    from config import load_settings

    settings = load_settings()
    async with ExchangeClient(settings) as exchange, PolymarketClient() as poly, KalshiClient() as kalshi:
        document = await build_market_map(exchange, poly, kalshi)
    write_market_map(document)

    entries = document["entries"].values()
    log.info(
        "Market map written",
        extra={
            "path": str(MARKET_MAP_PATH),
            "sig_exchanges": len(document["entries"]),
            "polymarket_matched": sum(1 for e in entries if e["polymarket"]),
            "kalshi_matched": sum(1 for e in entries if e["kalshi"]),
            "both_matched": sum(1 for e in entries if e["polymarket"] and e["kalshi"]),
            "none_matched": sum(1 for e in entries if not e["polymarket"] and not e["kalshi"]),
            "flagged": sum(1 for e in entries if e["notes"]),
        },
    )
    for title in document["unparsed_sig_markets"]:
        log.warning("Unparsed SIG market", extra={"market": title})
    for exchange_id, entry in document["entries"].items():
        if entry["notes"]:
            log.info(entry["title"], extra={"exchange_id": exchange_id, "notes": entry["notes"]})
    return 0


async def _cli_show(log: logging.Logger, needle: str) -> int:
    market_map = MarketMap.load()
    selected = [m for m in market_map if needle.lower() in m.title.lower()]
    async with PolymarketClient() as poly, KalshiClient() as kalshi:
        pm_quotes, ks_quotes = await asyncio.gather(
            poly.get_quotes([m.polymarket_token for m in selected if m.polymarket_token]),
            kalshi.get_quotes([m.kalshi_ticker for m in selected if m.kalshi_ticker]),
        )
    for mapping in selected:
        pm = _reference_price(pm_quotes.get(mapping.polymarket_token)) if mapping.polymarket_token else None
        ks = _reference_price(ks_quotes.get(mapping.kalshi_ticker)) if mapping.kalshi_ticker else None
        log.info(
            mapping.title,
            extra={
                "exchange_id": mapping.sig_exchange_id,
                "race": mapping.race.key,
                "polymarket": round(pm, 3) if pm is not None else None,
                "kalshi": round(ks, 3) if ks is not None else None,
            },
        )
    log.info("Shown", extra={"count": len(selected), "filter": needle})
    return 0


async def _main(argv: list[str]) -> int:
    from config import load_settings
    from logging_config import setup_logging

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    log = logging.getLogger("market_mapper")

    if argv[:1] == ["build"]:
        return await _cli_build(log)
    if argv[:1] == ["show"]:
        return await _cli_show(log, argv[1] if len(argv) > 1 else "")
    log.error('Usage: python market_mapper.py build | show ["title filter"]')
    return 2


if __name__ == "__main__":
    #   python market_mapper.py build              discover + write data/market_map.json
    #   python market_mapper.py show "Rhode Island"  live oracle prices for mapped markets
    import sys

    raise SystemExit(asyncio.run(_main(sys.argv[1:])))
