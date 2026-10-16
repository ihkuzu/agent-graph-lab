from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLM
from .toolbox import Toolbox


class ActionError(ValueError):
    pass


@dataclass
class Step:
    tool: str
    args: dict
    observation: str
    error: bool = False
    role: str = ""


@dataclass
class Result:
    answer: str
    steps: list[Step] = field(default_factory=list)
    finished: bool = True
    llm_calls: int = 0


def parse_action(reply: str) -> dict:
    text = re.sub(r"```(?:json)?", "", reply).strip()
    start = text.find("{")
    if start < 0:
        raise ActionError("Reply with a single JSON object.")
    try:
        action, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise ActionError(f"Invalid JSON ({error.msg}). Reply with a single JSON object.") from error
    if not isinstance(action, dict):
        raise ActionError("Reply with a single JSON object.")
    if "final" in action:
        return {"final": str(action["final"])}
    if isinstance(action.get("tool"), str):
        args = action.get("args") or {}
        if not isinstance(args, dict):
            raise ActionError("'args' must be an object.")
        return {"tool": action["tool"], "args": args}
    raise ActionError('The object needs a "tool" or a "final" key.')


def describe_tools(toolbox: Toolbox, names: tuple[str, ...]) -> str:
    lines = []
    for name in names:
        spec = toolbox.specs.get(name)
        if spec:
            params = ", ".join(f"{k}: {v}" for k, v in spec.parameters.items()) or "no arguments"
            lines.append(f"- {spec.name}({params}): {spec.description}")
    return "\n".join(lines)


def schema_text(toolbox: Toolbox) -> str:
    """Builds the database schema from the MCP tools list_tables and describe_table."""
    raw, failed = toolbox.call("list_tables", {})
    # a list result arrives as one text block per item
    tables = [] if failed else [line.strip() for line in raw.splitlines() if line.strip()]
    if not tables:
        return "(schema unavailable, use list_tables and describe_table)"
    lines = []
    for table in tables:
        text, failed = toolbox.call("describe_table", {"table": table})
        try:
            info = json.loads(text)
            columns = ", ".join(f"{c['name']} {c['type']}" for c in info["columns"])
            refs = "".join(f" [{k['column']} -> {k['references']}]" for k in info.get("foreign_keys", []))
            lines.append(f"{table}({columns}){refs}")
        except (ValueError, KeyError, TypeError):
            lines.append(table)
    return "\n".join(lines)


@dataclass
class Run:
    """State of one tool loop. A role is a prompt plus the tools it may use."""

    llm: LLM
    system: str
    toolbox: Toolbox
    allowed: tuple[str, ...]
    require: tuple[str, ...]
    question: str
    role: str = "agent"
    max_steps: int = 8
    messages: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    seen: set = field(default_factory=set)
    llm_calls: int = 0
    decisions: int = 0
    pending: dict | None = None
    answer: str | None = None

    def __post_init__(self):
        if not self.messages:
            self.messages = [{"role": "user", "content": self.question}]

    @property
    def done(self) -> bool:
        return self.answer is not None or self.decisions >= self.max_steps

    @property
    def finished(self) -> bool:
        return self.answer is not None

    def result(self) -> Result:
        answer = self.answer if self.answer is not None else f"I could not finish within {self.max_steps} steps."
        return Result(answer, self.steps, self.finished, self.llm_calls)

    def _record(self, step: Step) -> None:
        step.role = self.role
        self.steps.append(step)
        self.messages.append({"role": "user", "content": f"Observation: {step.observation}"})


def decide(run: Run) -> Run:
    """Asks the model for its next action and either stores it or records why it was refused."""
    run.pending = None
    run.decisions += 1
    reply = run.llm.chat(run.system, run.messages)
    run.llm_calls += 1
    run.messages.append({"role": "assistant", "content": reply})
    try:
        action = parse_action(reply)
    except ActionError as error:
        run._record(Step("(invalid reply)", {}, str(error), error=True))
        return run
    if "final" in action:
        if any(s.tool in run.require and not s.error for s in run.steps):
            run.answer = action["final"]
        else:
            # small models like to answer from memory, so make them use a tool first
            text = "You have not used the tools successfully yet. Check the sources before you answer."
            run._record(Step("(early answer)", {}, text, error=True))
        return run
    key = json.dumps(action, sort_keys=True)
    if action["tool"] not in run.allowed:
        text = f"unknown tool '{action['tool']}'. Available: {', '.join(run.allowed)}"
        run._record(Step(action["tool"], action["args"], text, error=True))
    elif key in run.seen:
        text = "You already ran this exact call. Try something different or give the final answer."
        run._record(Step(action["tool"], action["args"], text, error=True))
    else:
        run.seen.add(key)
        run.pending = action
    return run


def act(run: Run) -> Run:
    action, run.pending = run.pending, None
    text, failed = run.toolbox.call(action["tool"], action["args"])
    run._record(Step(action["tool"], action["args"], text, failed))
    return run


def run_loop(run: Run) -> Run:
    while not run.done:
        decide(run)
        if run.pending is not None:
            act(run)
    return run
