import pytest
from conftest import final, tool

from agentlab.core import ActionError, Run, parse_action, run_loop
from agentlab.llm import ScriptedLLM

DOCS = ("search_docs", "list_docs", "read_doc")


def make_run(replies, toolbox, max_steps=8, allowed=DOCS):
    llm = ScriptedLLM(replies)
    return Run(llm, "system", toolbox, allowed, allowed, "question", "agent", max_steps), llm


def test_parse_action_variants():
    assert parse_action('```json\n{"tool": "run_sql", "args": {"query": "x"}}\n```') == {
        "tool": "run_sql", "args": {"query": "x"}}
    assert parse_action('{"final": "ok"} trailing text') == {"final": "ok"}
    assert parse_action('{"tool": "list_docs"}') == {"tool": "list_docs", "args": {}}


@pytest.mark.parametrize("reply", ["no json", '{"x": 1}', '{"tool": "a", "args": 3}', "[1]", '{"tool": '])
def test_parse_action_rejects(reply):
    with pytest.raises(ActionError):
        parse_action(reply)


def test_tool_then_final(toolbox):
    run, llm = make_run([tool("search_docs", query="return window"), final("30 days")], toolbox)
    result = run_loop(run).result()
    assert result.finished and result.answer == "30 days"
    assert [s.tool for s in result.steps] == ["search_docs"] and result.llm_calls == 2
    assert "Observation:" in llm.calls[1][1][-1]["content"]


def test_answer_without_tool_use_is_refused(toolbox):
    run, _ = make_run([final("from memory"), tool("list_docs"), final("done")], toolbox)
    result = run_loop(run).result()
    assert result.answer == "done"
    assert result.steps[0].tool == "(early answer)" and result.steps[0].error


def test_repeated_call_is_rejected(toolbox):
    call = tool("search_docs", query="shipping")
    run, _ = make_run([call, call, final("x")], toolbox)
    steps = run_loop(run).result().steps
    assert not steps[0].error and steps[1].error and "already ran" in steps[1].observation


def test_tool_outside_the_role_is_rejected(toolbox):
    run, _ = make_run([tool("run_sql", query="SELECT 1"), tool("list_docs"), final("x")], toolbox)
    steps = run_loop(run).result().steps
    assert steps[0].error and "unknown tool" in steps[0].observation


def test_invalid_reply_is_reported_back(toolbox):
    run, llm = make_run(["oops", tool("list_docs"), final("x")], toolbox)
    result = run_loop(run).result()
    assert result.steps[0].tool == "(invalid reply)" and result.finished


def test_step_limit_ends_the_loop(toolbox):
    run, _ = make_run(["oops"] * 3, toolbox, max_steps=3)
    result = run_loop(run).result()
    assert not result.finished and "3 steps" in result.answer and result.llm_calls == 3
