import json
from pathlib import Path

import pytest
from mcpkb.sampledata import build_sample_db

from agentlab.toolbox import Toolbox


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    return build_sample_db(tmp_path_factory.mktemp("db") / "shop.db")


@pytest.fixture(scope="session")
def toolbox(db_path):
    with Toolbox.stdio(str(db_path), str(Path(__file__).parents[1] / "data" / "docs")) as box:
        yield box


def tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def final(text):
    return json.dumps({"final": text})
