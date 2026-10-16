import pytest
from conftest import final, tool

from agentlab.agents import GraphAgent, LoopAgent, MultiAgent, build_agent
from agentlab.llm import ScriptedLLM

COUNT = tool("run_sql", query="SELECT COUNT(*) FROM customers")
SEARCH = tool("search_docs", query="return window")


@pytest.mark.parametrize("cls", [LoopAgent, GraphAgent])
def test_single_agent_uses_both_sources(toolbox, cls):
    llm = ScriptedLLM([SEARCH, COUNT, final("30 days and 60 customers")])
    result = cls(llm, toolbox).run("q")
    assert result.finished and [s.tool for s in result.steps] == ["search_docs", "run_sql"]
    assert result.llm_calls == 3


def test_graph_behaves_like_the_loop(toolbox):
    # the graph only changes how the loop is wired, so the same replies must give the same trace
    script = ["garbage", final("early"), SEARCH, SEARCH, final("done")]
    results = [cls(ScriptedLLM(script), toolbox).run("q") for cls in (LoopAgent, GraphAgent)]
    traces = [[(s.tool, s.error) for s in r.steps] for r in results]
    assert traces[0] == traces[1] and results[0].answer == results[1].answer == "done"
    assert results[0].llm_calls == results[1].llm_calls


@pytest.mark.parametrize("cls", [LoopAgent, GraphAgent])
def test_single_agent_gives_up_at_the_step_limit(toolbox, cls):
    result = cls(ScriptedLLM(["oops"] * 3), toolbox, max_steps=3).run("q")
    assert not result.finished and result.llm_calls == 3


def test_multi_agent_both_route(toolbox):
    llm = ScriptedLLM([
        '{"route": "both"}',
        SEARCH, final("The window is 30 days."),
        COUNT, final("There are 60 customers."),
        final("30 days, 60 customers"),
    ])
    result = MultiAgent(llm, toolbox).run("q")
    assert result.finished and result.answer == "30 days, 60 customers" and result.llm_calls == 6
    assert [(s.role, s.tool) for s in result.steps] == [("docs", "search_docs"), ("sql", "run_sql")]
    synth_input = llm.calls[-1][1][0]["content"]
    assert "docs specialist: The window is 30 days." in synth_input and "sql specialist: There are 60" in synth_input


def test_multi_agent_sql_route_skips_the_docs_specialist(toolbox):
    llm = ScriptedLLM(['{"route": "sql"}', COUNT, final("60"), final("60 customers")])
    result = MultiAgent(llm, toolbox).run("q")
    assert [s.role for s in result.steps] == ["sql"] and result.llm_calls == 4


def test_multi_agent_unusable_router_reply_falls_back_to_both(toolbox):
    llm = ScriptedLLM(["no idea", SEARCH, final("a"), COUNT, final("b"), final("c")])
    result = MultiAgent(llm, toolbox).run("q")
    assert {s.role for s in result.steps} == {"docs", "sql"}


def test_specialists_only_see_their_own_tools(toolbox):
    llm = ScriptedLLM(['{"route": "docs"}', COUNT, SEARCH, final("a"), final("b")])
    result = MultiAgent(llm, toolbox).run("q")
    assert result.steps[0].error and "unknown tool" in result.steps[0].observation
    assert "run_sql" not in llm.calls[1][0] and "search_docs" in llm.calls[1][0]


def test_multi_agent_synthesizer_gets_one_retry(toolbox):
    llm = ScriptedLLM(['{"route": "docs"}', SEARCH, final("a"), "nonsense", final("fixed")])
    assert MultiAgent(llm, toolbox).run("q").answer == "fixed"
    llm = ScriptedLLM(['{"route": "docs"}', SEARCH, final("a"), "nonsense", "again nonsense"])
    result = MultiAgent(llm, toolbox).run("q")
    assert not result.finished


def test_multi_agent_marks_an_unfinished_specialist(toolbox):
    llm = ScriptedLLM(['{"route": "docs"}'] + ["oops"] * 2 + [final("not covered")])
    result = MultiAgent(llm, toolbox, max_steps=2).run("q")
    assert "could not finish" in llm.calls[-1][1][0]["content"] and result.finished


def test_build_agent():
    with pytest.raises(ValueError):
        build_agent("nope", ScriptedLLM([]), None)
