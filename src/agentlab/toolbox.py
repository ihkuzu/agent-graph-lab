from __future__ import annotations

import asyncio
import os
import sys
import threading
from dataclasses import dataclass

from mcp import Client, StdioServerParameters

DOC_TOOLS = ("search_docs", "list_docs", "read_doc")
SQL_TOOLS = ("list_tables", "describe_table", "run_sql")
MAX_OBSERVATION = 3000


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]


def _spec(tool) -> ToolSpec:
    schema = tool.input_schema or {}
    params = {k: v.get("type", "any") for k, v in schema.get("properties", {}).items()}
    return ToolSpec(tool.name, (tool.description or "").strip().split("\n")[0], params)


class Toolbox:
    """Sync wrapper around an async MCP client, so sync LLM code and graph nodes can call tools."""

    def __init__(self, target):
        self._target = target
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = None
        self._stop: asyncio.Future | None = None
        ready = self._submit(self._open())
        ready.result(timeout=60)
        self.specs: dict[str, ToolSpec] = {s.name: s for s in self._submit(self._list()).result(timeout=30)}

    @classmethod
    def stdio(cls, db_path: str, docs_dir: str | None = None, command: str | None = None) -> "Toolbox":
        env = {**os.environ, "MCPKB_DB_PATH": str(db_path)}
        if docs_dir:
            env["MCPKB_DOCS_DIR"] = str(docs_dir)
        params = StdioServerParameters(command=command or sys.executable, args=["-m", "mcpkb", "serve"], env=env)
        return cls(params)

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _open(self):
        # the client lives in one task so its context manager exits in the task that entered it
        started = self._loop.create_future()
        self._stop = self._loop.create_future()

        async def hold():
            async with Client(self._target) as client:
                self._client = client
                started.set_result(True)
                await self._stop

        self._holder = self._loop.create_task(hold())
        await started

    async def _list(self):
        return [_spec(t) for t in (await self._client.list_tools()).tools]

    async def _call(self, name: str, args: dict):
        return await self._client.call_tool(name, args)

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns the observation text and whether the tool reported an error."""
        try:
            result = self._submit(self._call(name, args)).result(timeout=60)
        except Exception as error:  # protocol level problems such as bad arguments
            return f"error: {error}", True
        text = "\n".join(getattr(block, "text", "") for block in result.content).strip()
        if len(text) > MAX_OBSERVATION:
            text = text[:MAX_OBSERVATION] + "\n... (cut)"
        return text, bool(result.is_error)

    def close(self) -> None:
        if self._stop is not None and not self._stop.done():
            self._loop.call_soon_threadsafe(self._stop.set_result, True)
            try:
                asyncio.run_coroutine_threadsafe(asyncio.wait_for(self._holder, 10), self._loop).result(15)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
