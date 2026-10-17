from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .core import Result
from .llm import LLMError
from .toolbox import DOC_TOOLS, SQL_TOOLS

CATEGORIES = ("docs", "sql", "hybrid", "unanswerable")
GROUPS = {"docs": DOC_TOOLS, "sql": SQL_TOOLS}
NOT_COVERED = (
    "not cover", "no information", "not mention", "no mention", "does not say", "do not say",
    "not specified", "not available", "cannot", "can't", "unable", "no data", "not contain",
    "does not contain", "do not contain", "not include", "doesn't", "don't have", "no details",
    "not found", "no relevant", "not provide",
)


@dataclass
class Case:
    id: str
    category: str
    question: str
    needs: list[str]
    expect: list[str] = field(default_factory=list)
    sql: str | None = None


@dataclass
class Outcome:
    case_id: str
    category: str
    answer: str
    correct: bool
    tools_ok: bool
    finished: bool
    steps: int
    llm_calls: int
    seconds: float
    error: str = ""
    trace: list[dict] = field(default_factory=list)


def load_cases(path: str | Path) -> list[Case]:
    return [Case(**item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


def _normalize(text: str) -> str:
    # drop thousands separators so "1,234.50" matches 1234.50
    return re.sub(r"(?<=\d),(?=\d{3})", "", text).lower()


def _present(token: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(token)}(?!\d|\.\d)", text) is not None


def value_matches(value, text: str) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        candidates = {f"{value:.2f}"}
        if round(value, 1) == round(value, 2):
            candidates.add(f"{value:.1f}")
        if abs(value - round(value)) < 0.005:
            candidates.add(str(round(value)))
        return any(_present(c, text) for c in candidates)
    if isinstance(value, int):
        return _present(str(value), text)
    return str(value).lower() in text


def answer_matches(answer: str, rows: list[tuple]) -> bool:
    text = _normalize(answer)
    return all(value_matches(v, text) for row in rows for v in row)


def tools_correct(case: Case, result: Result) -> bool:
    """Every source the question needs was used successfully at least once."""
    used = {s.tool for s in result.steps if not s.error}
    return all(used & set(GROUPS[need]) for need in case.needs)


def is_correct(case: Case, result: Result, db_path: str | Path) -> bool:
    if not result.finished:
        return False
    text = _normalize(result.answer)
    if case.category == "unanswerable":
        return any(marker in text for marker in NOT_COVERED)
    if not all(e.lower() in text for e in case.expect):
        return False
    if case.sql:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            rows = db.execute(case.sql).fetchall()
        return answer_matches(result.answer, rows)
    return True


def evaluate(agent, cases: list[Case], db_path: str | Path, on_case=None, max_failures: int = 3) -> list[Outcome]:
    outcomes = []
    failures = 0
    for case in cases:
        start = time.perf_counter()
        try:
            result = agent.run(case.question)
            error = ""
        except LLMError as exc:
            # one failing model call should not throw away the other results
            result, error = Result(f"model error: {exc}", [], False, 0), str(exc)
        seconds = time.perf_counter() - start
        outcome = Outcome(
            case.id, case.category, result.answer, is_correct(case, result, db_path),
            tools_correct(case, result), result.finished, len(result.steps), result.llm_calls,
            round(seconds, 2), error,
            [{"role": s.role, "tool": s.tool, "args": s.args, "error": s.error} for s in result.steps],
        )
        outcomes.append(outcome)
        if on_case:
            on_case(outcome)
        # a model that is down fails every question, so stop instead of recording noise
        failures = failures + 1 if error else 0
        if failures >= max_failures:
            raise LLMError(f"{failures} model errors in a row, last: {error[:200]}")
    return outcomes


def summarize(outcomes: list[Outcome]) -> dict:
    def block(items: list[Outcome]) -> dict:
        n = len(items) or 1
        return {
            "cases": len(items),
            "correct": sum(o.correct for o in items),
            "tools_ok": sum(o.tools_ok for o in items),
            "avg_steps": round(sum(o.steps for o in items) / n, 2),
            "avg_llm_calls": round(sum(o.llm_calls for o in items) / n, 2),
            "avg_seconds": round(sum(o.seconds for o in items) / n, 2),
        }

    summary = {"all": block(outcomes)}
    for category in CATEGORIES:
        subset = [o for o in outcomes if o.category == category]
        if subset:
            summary[category] = block(subset)
    return summary


def format_report(results: dict[str, list[Outcome]]) -> str:
    lines = ["| variant | correct | tools ok | steps | LLM calls | seconds |", "|---|---|---|---|---|---|"]
    for name, outcomes in results.items():
        s = summarize(outcomes)["all"]
        lines.append(
            f"| {name} | {s['correct']}/{s['cases']} | {s['tools_ok']}/{s['cases']} | "
            f"{s['avg_steps']} | {s['avg_llm_calls']} | {s['avg_seconds']} |"
        )
    lines += ["", "| variant | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for name, outcomes in results.items():
        summary = summarize(outcomes)
        cells = [f"{summary[c]['correct']}/{summary[c]['cases']}" if c in summary else "-" for c in CATEGORIES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(path: str | Path, meta: dict, results: dict[str, list[Outcome]]) -> None:
    payload = {
        "meta": meta,
        "summary": {name: summarize(o) for name, o in results.items()},
        "outcomes": {name: [asdict(o) for o in outcomes] for name, outcomes in results.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
