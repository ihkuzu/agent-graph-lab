from agentlab.core import schema_text


def test_lists_the_server_tools(toolbox):
    assert {"search_docs", "read_doc", "list_tables", "describe_table", "run_sql"} <= set(toolbox.specs)
    assert toolbox.specs["run_sql"].parameters["query"] == "string"


def test_call_returns_text(toolbox):
    text, failed = toolbox.call("run_sql", {"query": "SELECT COUNT(*) FROM customers"})
    assert not failed and "60" in text


def test_server_errors_are_flagged_not_raised(toolbox):
    text, failed = toolbox.call("run_sql", {"query": "DELETE FROM orders"})
    assert failed and "blocked" in text
    _, failed = toolbox.call("no_such_tool", {})
    assert failed


def test_schema_comes_from_the_mcp_tools(toolbox):
    schema = schema_text(toolbox)
    assert "orders(" in schema and "customer_id" in schema
