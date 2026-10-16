from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .core import (
    ActionError,
    Result,
    Run,
    act,
    decide,
    describe_tools,
    parse_action,
    run_loop,
    schema_text,
)
from .llm import LLM
from .toolbox import DOC_TOOLS, SQL_TOOLS, Toolbox

VARIANTS = ("loop", "graph", "multi")


class LoopAgent:
    """A: the loop written by hand."""

    name = "loop"

    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 8):
        self.llm, self.toolbox, self.max_steps = llm, toolbox, max_steps
        tools = DOC_TOOLS + SQL_TOOLS
        self.tools = tools
        self.system = prompts.SINGLE.format(tools=describe_tools(toolbox, tools), schema=schema_text(toolbox))

    def new_run(self, question: str) -> Run:
        return Run(self.llm, self.system, self.toolbox, self.tools, self.tools, question, "agent", self.max_steps)

    def run(self, question: str) -> Result:
        return run_loop(self.new_run(question)).result()


class _LoopState(TypedDict):
    run: Run


class GraphAgent(LoopAgent):
    """B: the same prompt, tools and limits, but the loop is a LangGraph graph (decide <-> act)."""

    name = "graph"

    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 8):
        super().__init__(llm, toolbox, max_steps)
        graph = StateGraph(_LoopState)
        graph.add_node("decide", lambda s: {"run": decide(s["run"])})
        graph.add_node("act", lambda s: {"run": act(s["run"])})
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide", lambda s: "act" if s["run"].pending is not None else (END if s["run"].done else "decide")
        )
        graph.add_edge("act", "decide")
        self.graph = graph.compile()

    def run(self, question: str) -> Result:
        state = self.graph.invoke({"run": self.new_run(question)}, {"recursion_limit": 4 * self.max_steps + 10})
        return state["run"].result()


class _TeamState(TypedDict):
    question: str
    route: str
    notes: dict
    steps: list
    llm_calls: int
    answer: str | None


class MultiAgent:
    """C: router, then a documents and a data specialist, then a synthesizer."""

    name = "multi"

    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 6):
        self.llm, self.toolbox, self.max_steps = llm, toolbox, max_steps
        schema = schema_text(toolbox)
        self.docs_system = prompts.DOCS.format(tools=describe_tools(toolbox, DOC_TOOLS))
        self.sql_system = prompts.SQL.format(tools=describe_tools(toolbox, SQL_TOOLS), schema=schema)
        self.router_system = prompts.ROUTER.format()
        self.synth_system = prompts.SYNTH.format()
        graph = StateGraph(_TeamState)
        graph.add_node("router", self._router)
        graph.add_node("docs", self._specialist("docs", self.docs_system, DOC_TOOLS, ("search_docs", "read_doc")))
        graph.add_node("sql", self._specialist("sql", self.sql_system, SQL_TOOLS, ("run_sql",)))
        graph.add_node("synth", self._synth)
        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", lambda s: "sql" if s["route"] == "sql" else "docs")
        graph.add_conditional_edges("docs", lambda s: "sql" if s["route"] == "both" else "synth")
        graph.add_edge("sql", "synth")
        graph.add_edge("synth", END)
        self.graph = graph.compile()

    def _router(self, state: _TeamState) -> dict:
        reply = self.llm.chat(self.router_system, [{"role": "user", "content": state["question"]}])
        route = "both"  # when the reply is unusable, asking both specialists is the safe choice
        try:
            value = json.JSONDecoder().raw_decode(reply[reply.find("{"):])[0].get("route")
            if value in ("docs", "sql", "both"):
                route = value
        except (ValueError, AttributeError):
            pass
        return {"route": route, "llm_calls": state["llm_calls"] + 1}

    def _specialist(self, role: str, system: str, allowed: tuple, require: tuple):
        def node(state: _TeamState) -> dict:
            run = run_loop(Run(self.llm, system, self.toolbox, allowed, require, state["question"], role, self.max_steps))
            note = run.answer if run.finished else "(could not finish)"
            return {
                "notes": {**state["notes"], role: note},
                "steps": state["steps"] + run.steps,
                "llm_calls": state["llm_calls"] + run.llm_calls,
            }

        return node

    def _synth(self, state: _TeamState) -> dict:
        notes = "\n".join(f"- {role} specialist: {text}" for role, text in state["notes"].items())
        messages = [{"role": "user", "content": f"Question: {state['question']}\n\nNotes:\n{notes}"}]
        calls = state["llm_calls"]
        for _ in range(2):
            reply = self.llm.chat(self.synth_system, messages)
            calls += 1
            try:
                action = parse_action(reply)
            except ActionError as error:
                messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": str(error)}]
                continue
            if "final" in action:
                return {"answer": action["final"], "llm_calls": calls}
        return {"answer": None, "llm_calls": calls}

    def run(self, question: str) -> Result:
        start: _TeamState = {"question": question, "route": "", "notes": {}, "steps": [], "llm_calls": 0, "answer": None}
        state = self.graph.invoke(start, {"recursion_limit": 25})
        answer = state["answer"]
        if answer is None:
            return Result("I could not finish the answer.", state["steps"], False, state["llm_calls"])
        return Result(answer, state["steps"], True, state["llm_calls"])


def build_agent(variant: str, llm: LLM, toolbox: Toolbox, max_steps: int | None = None):
    classes = {"loop": LoopAgent, "graph": GraphAgent, "multi": MultiAgent}
    if variant not in classes:
        raise ValueError(f"unknown variant: {variant}")
    kwargs = {"max_steps": max_steps} if max_steps else {}
    return classes[variant](llm, toolbox, **kwargs)
