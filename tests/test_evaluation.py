import sqlite3
from pathlib import Path


from agentlab.core import Result, Step
from agentlab.evaluation import (
    Case, answer_matches, evaluate, format_report, is_correct, load_cases, summarize, tools_correct,
)
from agentlab.llm import LLMError

CASES = "data/eval/questions.json"


def result(answer, *tools, finished=True):
    return Result(answer, [Step(t, {}, "", e) for t, e in tools], finished, len(tools) + 1)


def test_question_file_is_consistent(db_path):
    cases = load_cases(CASES)
    assert len({c.id for c in cases}) == len(cases)
    assert {c.category for c in cases} == {"docs", "sql", "hybrid", "unanswerable"}
    with sqlite3.connect(db_path) as db:
        for case in cases:
            assert case.needs and set(case.needs) <= {"docs", "sql"}
            if case.sql:
                assert db.execute(case.sql).fetchall(), case.id
            if case.category == "sql":
                assert case.sql and case.needs == ["sql"]
            if case.category == "hybrid":
                assert case.needs == ["docs", "sql"]


def test_expected_doc_facts_are_in_the_documents():
    docs = " ".join(p.read_text() for p in Path("data/docs").glob("*.md")).lower()
    for case in load_cases(CASES):
        for word in case.expect:
            assert word.lower() in docs, (case.id, word)


def test_number_matching():
    assert answer_matches("Revenue is 406,391.00 EUR", [(406391.0,)])
    assert answer_matches("There are 60 customers", [(60,)])
    assert not answer_matches("There are 160 customers", [(60,)])
    assert answer_matches("Laptop 16 costs 1299", [("Laptop 16", 1299.0)])


def test_correctness_rules(db_path):
    docs = Case("d", "docs", "q", ["docs"], ["30"])
    assert is_correct(docs, result("30 days"), db_path)
    assert not is_correct(docs, result("14 days"), db_path)
    assert not is_correct(docs, result("30 days", finished=False), db_path)
    sql = Case("s", "sql", "q", ["sql"], [], "SELECT COUNT(*) FROM customers")
    assert is_correct(sql, result("60 customers"), db_path)
    assert not is_correct(sql, result("59 customers"), db_path)
    none = Case("u", "unanswerable", "q", ["docs"])
    assert is_correct(none, result("The documents do not cover this."), db_path)
    assert not is_correct(none, result("It is 12 weeks."), db_path)


def test_tool_use_needs_a_successful_call_per_source():
    both = Case("h", "hybrid", "q", ["docs", "sql"])
    assert tools_correct(both, result("x", ("search_docs", False), ("run_sql", False)))
    assert not tools_correct(both, result("x", ("search_docs", False), ("run_sql", True)))
    assert not tools_correct(both, result("x", ("search_docs", False)))


class FakeAgent:
    def __init__(self, replies):
        self.replies = replies

    def run(self, question):
        reply = self.replies[question]
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_evaluate_keeps_going_after_a_model_error(db_path):
    cases = [Case("a", "docs", "qa", ["docs"], ["30"]), Case("b", "docs", "qb", ["docs"], ["30"])]
    agent = FakeAgent({"qa": LLMError("down"), "qb": result("30 days", ("search_docs", False))})
    outcomes = evaluate(agent, cases, db_path)
    assert [o.correct for o in outcomes] == [False, True] and outcomes[0].error == "down"
    summary = summarize(outcomes)
    assert summary["all"]["correct"] == 1 and summary["docs"]["cases"] == 2
    assert "| fake | 1/2 |" in format_report({"fake": outcomes})


def test_evaluate_stops_when_the_model_keeps_failing(db_path):
    cases = [Case(str(i), "docs", f"q{i}", ["docs"], ["30"]) for i in range(5)]
    agent = FakeAgent({f"q{i}": LLMError("down") for i in range(5)})
    with pytest.raises(LLMError, match="3 model errors in a row"):
        evaluate(agent, cases, db_path)
