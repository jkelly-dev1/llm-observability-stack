"""The paid run's exits, tested without spending anything.

scripts/real_run.py returns 0 or 2, and the 2 is a verdict: the run refused.
These call main() directly with an explicit argv. The confirmed-run case
replaces the `anthropic` module with one that fails on construction, so no
version of this test can reach the network or bill anybody, whatever the
shell holds.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _real_run():
    spec = importlib.util.spec_from_file_location(
        "real_run", ROOT / "scripts" / "real_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def no_network(monkeypatch):
    """An `anthropic` whose client cannot be built, so nothing is sent."""
    stub = types.ModuleType("anthropic")

    def refuse(*args, **kwargs):
        raise AssertionError("the test reached the API client")

    stub.Anthropic = refuse
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")


def test_the_dry_run_sends_nothing_and_exits_zero(tmp_path, no_network):
    out = tmp_path / "result.json"
    assert _real_run().main(["--out", str(out)]) == 0
    assert not out.exists(), "a dry run wrote a results file"


def test_the_cost_cap_refuses_with_exit_two(tmp_path, no_network):
    out = tmp_path / "result.json"
    rc = _real_run().main(["--max-cost", "0.000001", "--confirm",
                           "--out", str(out)])
    assert rc == 2
    assert not out.exists()


def test_a_confirmed_run_refuses_to_overwrite_existing_evidence(
        tmp_path, no_network):
    existing = tmp_path / "real_run.json"
    existing.write_text('{"evidence": true}')
    rc = _real_run().main(["--confirm", "--out", str(existing)])
    assert rc == 2, "a paid run was allowed to start over existing evidence"
    assert existing.read_text() == '{"evidence": true}'


def test_the_default_output_is_a_new_file_not_the_shipped_evidence():
    import ast
    tree = ast.parse((ROOT / "scripts" / "real_run.py").read_text())
    defaults = [kw.value for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"
                and node.args and getattr(node.args[0], "value", "") == "--out"
                for kw in node.keywords if kw.arg == "default"]
    assert defaults, "no --out argument found; this test examined nothing"
    assert all(isinstance(d, ast.Constant) and d.value is None
               for d in defaults), (
        "--out has a fixed default, so the documented command would write "
        "over whatever file that names")
