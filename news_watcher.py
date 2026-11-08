"""RSS news watcher - an internal trigger only.

When a new headline mentions a race we trade (built automatically from the market
map: "New Hampshire" + "Senate", "NH-01", "control of the House", ...), it calls
`on_match`, which main.py uses to start the next strategy cycle immediately instead
of waiting out the normal interval. That is the entire job: no notifications, no
sentiment scoring - headlines are far too slow and noisy to trade on directly
against real-money oracle prices, but they are a good hint to re-poll NOW.

A false positive costs one extra cycle (~3 s of reads), so matching errs broad.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import aiohttp
import feedparser

from market_mapper import MarketMap, Race, _STATE_NAMES

logger = logging.getLogger(__name__)

# Free, key-less political feeds (all verified reachable 2026-09-30).
# Politico retired politics-news.xml (404 since 2026-09-30); congress.xml replaces it.
DEFAULT_FEEDS = (
    "https://thehill.com/homenews/campaign/feed/",
    "https://rss.politico.com/congress.xml",
    "https://feeds.npr.org/1014/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "https://moxie.foxnews.com/google-publisher/politics.xml",
    "https://www.cbsnews.com/latest/rss/politics",
    "https://feeds.nbcnews.com/nbcnews/public/politics",
    "https://www.theguardian.com/us-news/us-politics/rss",
)

POLL_INTERVAL_S = 60
TRIGGER_COOLDOWN_S = 20  # at most one early cycle per this many seconds
REQUEST_TIMEOUT_S = 15
MAX_SEEN_ENTRIES = 5000
LOG_FEED_FAILURE_EVERY = 30  # log the 1st, 31st, 61st... consecutive failure of a feed

_OFFICE_PATTERNS = {
    "senate": re.compile(r"\b(senate|senator|sen\.)", re.IGNORECASE),
    "governor": re.compile(r"\b(governor|gov\.|gubernatorial)", re.IGNORECASE),
}
_CONTROL_PATTERNS = {
    "senate_control": re.compile(
        r"\b(control of the senate|senate (control|majority)|battle for (the )?senate)\b", re.IGNORECASE
    ),
    "house_control": re.compile(
        r"\b(control of the house|house (control|majority)|battle for (the )?house)\b", re.IGNORECASE
    ),
}
_HTML_TAG = re.compile(r"<[^>]+>")


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass(frozen=True)
class NewsMatch:
    race_keys: tuple[str, ...]
    title: str
    link: str
    feed: str


class KeywordIndex:
    """Maps free text to the race keys (Race.key) it mentions."""

    def __init__(self, market_map: MarketMap) -> None:
        races = {m.race.key: m.race for m in market_map}
        self._state_races: list[tuple[str, re.Pattern[str], re.Pattern[str]]] = []
        self._direct: list[tuple[str, re.Pattern[str]]] = []
        for key, race in sorted(races.items()):
            self._add(key, race)

    def _add(self, key: str, race: Race) -> None:
        if race.office in _CONTROL_PATTERNS:
            self._direct.append((key, _CONTROL_PATTERNS[race.office]))
        elif race.office == "house":
            # "NH-01", "NH-1", "New Hampshire's 1st (congressional) district"
            district = int(race.district)
            state = re.escape(_STATE_NAMES[race.state])
            pattern = rf"\b{race.state}-0?{district}\b|\b{state}'s {_ordinal(district)} (congressional )?district\b"
            self._direct.append((key, re.compile(pattern, re.IGNORECASE)))
        else:
            # Statewide race: the state name AND the office must both appear.
            state = re.compile(rf"\b{re.escape(_STATE_NAMES[race.state])}\b", re.IGNORECASE)
            self._state_races.append((key, state, _OFFICE_PATTERNS[race.office]))

    def match(self, text: str) -> tuple[str, ...]:
        keys = [key for key, pattern in self._direct if pattern.search(text)]
        keys += [key for key, state, office in self._state_races if state.search(text) and office.search(text)]
        return tuple(sorted(set(keys)))


OnMatch = Callable[[NewsMatch], Awaitable[None] | None]


class NewsWatcher:
    def __init__(
        self,
        market_map: MarketMap,
        on_match: OnMatch,
        feeds: tuple[str, ...] = DEFAULT_FEEDS,
        poll_interval_s: float = POLL_INTERVAL_S,
    ) -> None:
        self._index = KeywordIndex(market_map)
        self._on_match = on_match
        self._feeds = feeds
        self._poll_interval_s = poll_interval_s
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._validators: dict[str, dict[str, str]] = {}  # feed -> ETag / Last-Modified
        self._failures: dict[str, int] = {}
        self._last_trigger = 0.0
        self._pending: list[NewsMatch] = []

    async def run(self, stop: asyncio.Event) -> None:
        """Poll until `stop` is set. The first pass only primes `seen` - old news triggers nothing."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "prediction-cup-bot/1.0"}) as session:
            await self._poll_all(session, prime=True)
            logger.info("News watcher started", extra={"feeds": len(self._feeds), "seen": len(self._seen)})
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_interval_s)
                except asyncio.TimeoutError:
                    pass
                if stop.is_set():
                    break
                matches = await self._poll_all(session, prime=False)
                await self._maybe_trigger(matches)

    async def scan_once(self) -> list[NewsMatch]:
        """Match everything currently in the feeds (ignores `seen`) - for the sanity check."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "prediction-cup-bot/1.0"}) as session:
            return await self._poll_all(session, prime=False)

    async def _poll_all(self, session: aiohttp.ClientSession, prime: bool) -> list[NewsMatch]:
        results = await asyncio.gather(*(self._poll_feed(session, feed, prime) for feed in self._feeds))
        return [match for feed_matches in results for match in feed_matches]

    async def _poll_feed(self, session: aiohttp.ClientSession, feed: str, prime: bool) -> list[NewsMatch]:
        try:
            async with session.get(feed, headers=self._validators.get(feed, {})) as response:
                if response.status == 304:
                    return []  # unchanged since last poll
                response.raise_for_status()
                raw = await response.read()
                validators = {}
                if etag := response.headers.get("ETag"):
                    validators["If-None-Match"] = etag
                if modified := response.headers.get("Last-Modified"):
                    validators["If-Modified-Since"] = modified
                self._validators[feed] = validators
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            count = self._failures[feed] = self._failures.get(feed, 0) + 1
            if count % LOG_FEED_FAILURE_EVERY == 1:
                logger.warning("News feed unavailable", extra={"feed": feed, "consecutive_failures": count, "error": repr(exc)})
            return []
        self._failures[feed] = 0

        matches = []
        for entry in feedparser.parse(raw).entries:
            entry_id = entry.get("id") or entry.get("link") or entry.get("title")
            if not entry_id or (not prime and entry_id in self._seen):
                continue
            self._remember(entry_id)
            if prime:
                continue
            title = entry.get("title", "")
            text = f"{title} {_HTML_TAG.sub(' ', entry.get('summary', ''))}"
            race_keys = self._index.match(text)
            if race_keys:
                matches.append(NewsMatch(race_keys, title, entry.get("link", ""), feed))
        return matches

    def _remember(self, entry_id: str) -> None:
        self._seen[entry_id] = None
        self._seen.move_to_end(entry_id)
        while len(self._seen) > MAX_SEEN_ENTRIES:
            self._seen.popitem(last=False)

    async def _maybe_trigger(self, matches: list[NewsMatch]) -> None:
        for match in matches:
            logger.info("News match", extra={"races": match.race_keys, "title": match.title, "link": match.link})
        self._pending.extend(matches)
        if not self._pending or time.monotonic() - self._last_trigger < TRIGGER_COOLDOWN_S:
            return  # pending matches fire on a later poll, once the cooldown has passed

        combined = NewsMatch(
            race_keys=tuple(sorted({key for m in self._pending for key in m.race_keys})),
            title=self._pending[-1].title,
            link=self._pending[-1].link,
            feed=f"{len(self._pending)} headline(s)",
        )
        self._pending = []
        self._last_trigger = time.monotonic()
        result = self._on_match(combined)
        if inspect.isawaitable(result):
            await result


async def _smoke_test() -> int:
    from config import load_settings
    from logging_config import setup_logging

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    log = logging.getLogger("news_watcher.smoke_test")

    watcher = NewsWatcher(MarketMap.load(), on_match=lambda match: None)
    matches = await watcher.scan_once()
    for match in matches:
        log.info(match.title, extra={"races": match.race_keys, "feed": match.feed})
    log.info("Scan done", extra={"feeds": len(DEFAULT_FEEDS), "matching_headlines": len(matches)})
    return 0


if __name__ == "__main__":
    # Sanity check: `python news_watcher.py` - one pass over all feeds, prints the
    # headlines that WOULD trigger an early cycle. Read-only, no orders.
    raise SystemExit(asyncio.run(_smoke_test()))
