# agent-graph-lab

The same 20 questions, answered three ways, on the same tools and the same eval:

| variant | what it is |
|---|---|
| `loop` | one agent, a tool loop I wrote by hand |
| `graph` | the same agent, prompt and limits, but the loop is a [LangGraph](https://github.com/langchain-ai/langgraph) graph |
| `multi` | a LangGraph team: a router, a documents specialist, a data specialist and a synthesizer |

The tools are not written here. They come from my [mcp-knowledge-server](https://github.com/ihkuzu/mcp-knowledge-server) over MCP (stdio subprocess): `search_docs`, `read_doc`, `list_docs`, `list_tables`, `describe_table` and `run_sql` for a fictional online shop (10 policy documents, a SQLite database with customers, products and orders).

The point is the comparison. A framework or a team of agents should earn its extra model calls, so the eval counts correctness, tool use, steps, model calls and latency per variant instead of assuming that more structure is better.

## Architecture

```mermaid
flowchart LR
    Q[question] --> R{router}
    R -->|docs| D[documents specialist]
    R -->|sql| S[data specialist]
    R -->|both| D
    D -->|both| S
    D --> Y[synthesizer]
    S --> Y
    Y --> A[answer]
    D -. MCP .-> T[(mcp-knowledge-server)]
    S -. MCP .-> T
```

`loop` and `graph` use one agent with all six tools. In `multi` each specialist only sees its own tools (documents or database) and its own prompt, and runs the same bounded tool loop. The synthesizer sees the question and the specialists' notes, nothing else.

All three share the code in `core.py`: one `decide` step (ask the model, parse the JSON action) and one `act` step (call the tool). `loop` calls them in a `while`, `graph` wires them as two nodes with a conditional edge. So any difference between those two is orchestration overhead, not a different prompt.

Rules every loop enforces, because small models need them:

- one JSON object per reply: `{"tool": ..., "args": ...}` or `{"final": ...}`; invalid replies are reported back as an observation
- a step limit per loop
- an identical repeated tool call is refused
- a final answer before a successful tool call is refused (the model has to look at the sources first)
- a tool outside the role's tool set is refused

The MCP client is async, the model clients and graph nodes are sync, so `Toolbox` runs the client in one task on a background event loop and exposes a plain blocking `call`.

## The eval

`data/eval/questions.json` has 20 questions in four groups:

| group | count | needs | example |
|---|---|---|---|
| docs | 6 | documents | "From which order value is shipping free?" |
| sql | 6 | database | "How many customers live in Munich?" |
| hybrid | 5 | both | "What is our total revenue according to the company's definition of revenue?" |
| unanswerable | 3 | documents (to find out there is nothing) | "What is the company's parental leave policy?" |

- **correct**: the answer contains the expected facts. SQL gold values are computed by running a reference query on the database, not typed in. For unanswerable questions the answer has to say the documents do not cover it.
- **tools ok**: every source the question needs was used successfully at least once.
- **steps / LLM calls / seconds**: average per question.

The check is substring and number matching, so a correct answer in unusual words can be marked wrong. That is the same trade-off as in my [sql-agent](https://github.com/ihkuzu/sql-agent).

## Results

Not measured yet. Run the eval on your own machine, the numbers depend on the model:

```bash
agentlab eval --llm ollama --model llama3.2:3b --out results.json
```

## Run it

```bash
pip install -e ".[dev]"
agentlab init-db                  # sample shop database in data/shop.db
agentlab ask "How many customers live in Munich?" --variant multi
agentlab eval --variants loop,graph,multi --out results.json
```

Models: Ollama (default, `llama3.2:3b`) or Gemini (`--llm gemini`, needs `GEMINI_API_KEY`). Replies are requested as JSON.

Tests need no model and no network. They script the model's replies and run against the real MCP server over stdio:

```bash
pytest -q
```

## Limits

- Documents and data are fictional and small (10 documents, 18 products, 400 orders). The numbers say something about these agents on this task, not about agents in general.
- 20 questions are enough to see big differences and not enough to rank two close variants.
- A 3B model through Ollama is weak at following the JSON protocol. That is on purpose (it shows what the guard rails catch), but it also means failures can be the model and not the architecture. The Gemini option exists to check that.
- The synthesizer can only be as good as the specialists' notes; it has no tools to check them.
- No memory between questions, no streaming, no parallel specialists.

## Layout

```
src/agentlab/
  core.py        decide/act steps, limits, parser, schema from MCP
  agents.py      LoopAgent, GraphAgent, MultiAgent
  prompts.py     system prompts per role
  toolbox.py     sync wrapper around the async MCP client
  evaluation.py  cases, correctness, tool-use check, report
  llm.py         Ollama, Gemini and a scripted model for tests
  cli.py
data/docs        the documents served by the MCP server
data/eval        the questions
tests/           34 tests, no model needed
```

MIT license.
