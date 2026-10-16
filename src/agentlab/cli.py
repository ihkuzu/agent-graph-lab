from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents import VARIANTS, build_agent
from .evaluation import evaluate, format_report, load_cases, save_results
from .llm import LLMError, get_llm
from .toolbox import Toolbox

DEFAULT_DB = "data/shop.db"
DEFAULT_DOCS = "data/docs"
DEFAULT_CASES = "data/eval/questions.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentlab", description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite file served by the MCP server")
    parser.add_argument("--docs", default=DEFAULT_DOCS, help="folder with the markdown documents")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the sample shop database")

    def model_options(p):
        p.add_argument("--llm", choices=["ollama", "gemini"], help="default: AGENTLAB_LLM or ollama")
        p.add_argument("--model", help="model name, default depends on --llm")
        p.add_argument("--max-steps", type=int, help="tool calls allowed per loop")

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--variant", choices=VARIANTS, default="multi")
    model_options(ask)

    ev = sub.add_parser("eval", help="run the eval for one or more variants")
    ev.add_argument("--variants", default=",".join(VARIANTS), help="comma separated")
    ev.add_argument("--cases", default=DEFAULT_CASES)
    ev.add_argument("--limit", type=int, help="only the first N questions")
    ev.add_argument("--out", help="write the full results as JSON")
    model_options(ev)
    return parser


def _init_db(path: str) -> None:
    from mcpkb.sampledata import build_sample_db

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    build_sample_db(Path(path))
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-db":
        _init_db(args.db)
        return 0
    if not Path(args.db).exists():
        print(f"{args.db} does not exist, run: agentlab init-db", file=sys.stderr)
        return 2
    try:
        llm = get_llm(args.llm, args.model)
    except (LLMError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    with Toolbox.stdio(args.db, args.docs) as toolbox:
        if args.command == "ask":
            agent = build_agent(args.variant, llm, toolbox, args.max_steps)
            try:
                result = agent.run(args.question)
            except LLMError as error:
                print(f"model error: {error}", file=sys.stderr)
                return 1
            for step in result.steps:
                flag = " (error)" if step.error else ""
                print(f"[{step.role}] {step.tool} {step.args}{flag}", file=sys.stderr)
            print(result.answer)
            return 0 if result.finished else 1

        cases = load_cases(args.cases)[: args.limit]
        results = {}
        for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
            agent = build_agent(name, llm, toolbox, args.max_steps)
            print(f"== {name}", file=sys.stderr)

            def show(o):
                print(f"{o.case_id:4} {'ok ' if o.correct else 'bad'} tools={'ok ' if o.tools_ok else 'bad'} "
                      f"steps={o.steps} calls={o.llm_calls} {o.seconds}s", file=sys.stderr)

            results[name] = evaluate(agent, cases, args.db, show)
        print(format_report(results))
        if args.out:
            meta = {"llm": args.llm or "ollama", "model": args.model or "default", "cases": len(cases)}
            save_results(args.out, meta, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
