"""With every pilot flag at its default (off), the quoter behaves EXACTLY as before the pilot flags
were added: same orders, cancels, book reads, cycle reports, journal and log records.

tests/golden/quoter_cycle_golden.json was recorded with the quoter code before the pilot flags
(commit 76d4c92) by run_scenario() below. Regenerate it only for an intended behaviour change:
    python -c "import sys; sys.path[:0] = ['.', 'tests']; import test_quoter_default_identical as t; t.write_golden()"

Run from the project root:  python -m pytest tests/test_quoter_default_identical.py -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import quoter as Q
from markout import Markout
from strategy import Holding, SigBook
from test_quoter import NOW, FakeExchange, _ctx, _open_orders

GOLDEN = Path(__file__).resolve().parent / "golden" / "quoter_cycle_golden.json"
_STANDARD = set(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None).__dict__) | {"message", "asctime"}
# What strategy.py copies from a QuoterReport into the cycle report ("Cycle done").
REPORT_FIELDS = ("markets", "placed", "cancels", "requotes", "fills", "killed")


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        extra = {k: v for k, v in record.__dict__.items() if k not in _STANDARD}
        self.records.append({"level": record.levelname, "message": record.getMessage(), "extra": json.loads(json.dumps(extra, default=str, sort_keys=True))})


def _cycle_report(report: Q.QuoterReport) -> dict:
    return {**{f: getattr(report, f) for f in REPORT_FIELDS},
            "blocked": {k.removeprefix("blocked_"): v for k, v in report.__dict__.items() if k.startswith("blocked_") and v}, "skipped": report.skipped}


def run_scenario(tmp: Path, mode: str) -> dict:
    """Seven cycles: several markets, tail and spread blocks, a book move (requote), a fill, three
    adverse markouts (kill), a fair jump, a carry position in the race, and quote expiry."""
    exchange = FakeExchange()
    capture = _Capture()
    log = logging.getLogger("quoter")
    old_level = log.level
    log.addHandler(capture)
    log.setLevel(logging.DEBUG)
    journal_path = tmp / f"quoter-{mode}.jsonl"
    out: dict = {"cycles": []}
    try:
        manager = Q.QuoterManager(exchange, journal=Q.QuoterJournal(journal_path), clock=lambda: NOW - 1000, dry_run_mode=mode)
        wide = SigBook(yes_bid=0.45, yes_ask=0.55)
        books = {"1": wide, "2": wide, "3": wide, "4": SigBook(yes_bid=0.49, yes_ask=0.50), "5": wide}
        fair = {"1": 0.50, "2": 0.97, "3": 0.50, "4": 0.50, "5": 0.50}
        steps = []
        steps.append(dict(now=NOW, fair=fair, books=books))
        steps.append(dict(now=NOW + 15, fair=fair, books={**books, "1": SigBook(yes_bid=0.455, yes_ask=0.55)}))
        steps.append(dict(now=NOW + 30, fair=fair, books={**books, "1": SigBook(yes_bid=0.455, yes_ask=0.55)}, fill=True))
        steps.append(dict(now=NOW + 45, fair=fair, books=books, markouts=[Markout("quoter", "3", f"f{i}", "o", NOW, 300, 0.5, -3.0) for i in range(3)]))
        carry = {"2": Holding("no", 400, 380.0, 360.0)}
        steps.append(dict(now=NOW + 60, fair={**fair, "5": 0.56}, books=books, holdings=carry))
        steps.append(dict(now=NOW + 75, fair={**fair, "5": 0.56}, books=books, holdings=carry))
        steps.append(dict(now=NOW + 200, fair=fair, books=books, holdings=carry))
        for step in steps:
            seen = (len(exchange.batches), len(exchange.cancel_alls), len(exchange.book_reads))
            fills = None
            if step.get("fill"):
                yes_id = next(oid for oid, o in manager.orders.items() if o.exchange_id == "1" and o.side == "yes")
                fills = [{"id": "fill-1", "orderId": yes_id, "exchangeId": "1", "side": "yes", "price": 0.455, "quantity": 60, "filledAt": "2027-01-15T08:00:20.000Z"}]
            ctx = replace(_ctx(step["fair"], now=step["now"], books=step["books"], holdings=step.get("holdings")),
                          open_orders=[] if manager.virtual else _open_orders(manager), fills=fills, markouts=step.get("markouts", []))
            report = asyncio.run(manager.run_cycle(ctx))
            out["cycles"].append({
                "report": _cycle_report(report),
                "batches": [{"rank": rank, "orders": [asdict(o) for o in orders]} for orders, rank in exchange.batches[seen[0]:]],
                "cancel_alls": exchange.cancel_alls[seen[1]:], "book_reads": exchange.book_reads[seen[2]:],
                "resting": sorted([o.exchange_id, o.side, o.price, o.quantity] for o in manager.orders.values()),
            })
        out["positions"] = {eid: asdict(p) for eid, p in sorted(manager.positions.items())}
        out["journal"] = [{k: v for k, v in json.loads(line).items() if k != "ts"} for line in journal_path.read_text(encoding="utf-8").splitlines()]
        out["logs"] = capture.records
    finally:
        log.removeHandler(capture)
        log.setLevel(old_level)
    return json.loads(json.dumps(out, default=str))


def write_golden() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        golden = {mode: run_scenario(Path(tmp), mode) for mode in ("inherit", "true")}
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mode", ["inherit", "true"])
def test_default_flags_reproduce_the_recorded_quoter_cycles(tmp_path, mode):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[mode]
    got = run_scenario(tmp_path, mode)
    for key in ("cycles", "positions", "journal", "logs"):
        assert got[key] == golden[key], key


def test_the_scenario_exercises_every_path():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["inherit"]
    reports = [c["report"] for c in golden["cycles"]]
    assert sum(r["requotes"] for r in reports) >= 2 and sum(r["fills"] for r in reports) == 1 and sum(r["killed"] for r in reports) == 1
    assert any(c["cancel_alls"] for c in golden["cycles"]) and any(r["blocked"] for r in reports)
    messages = {r["message"] for r in golden["logs"]}
    assert {"Quoter enabled", "Quoter summary", "Quoter jump guard", "Quoter market killed: consecutive adverse markouts", "Quoter quotes withdrawn"} <= messages
