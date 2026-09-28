"""The README figure checker, run the way CI runs it.

CI acts on the exit code, so both directions are asserted: 0 on the shipped
README, 1 on one with a derived prose figure changed.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_readme_numbers  # noqa: E402


def test_the_figure_checker_returns_a_verdict_ci_can_act_on(monkeypatch, tmp_path, capsys):
    assert check_readme_numbers.main() == 0
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "the trace is off by 33.75%" in " ".join(text.split())
    broken = tmp_path / "README.md"
    broken.write_text(text.replace("33.75%", "34.75%"), encoding="utf-8")
    monkeypatch.setattr(check_readme_numbers, "README", str(broken))
    assert check_readme_numbers.main() == 1
    assert "MISSING [prose:cache-write]" in capsys.readouterr().out


def test_the_promotion_count_is_derived_everywhere_it_is_stated(monkeypatch, tmp_path, capsys):
    """The heading, the block and the sibling paragraph each state it."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count("0 of 500") == 3
    for i in range(3):
        parts = text.split("0 of 500")
        broken = tmp_path / f"README{i}.md"
        broken.write_text("0 of 500".join(parts[:i + 1]) + "1 of 500"
                          + "0 of 500".join(parts[i + 1:]), encoding="utf-8")
        monkeypatch.setattr(check_readme_numbers, "README", str(broken))
        assert check_readme_numbers.main() == 1, f"occurrence {i} unchecked"
    assert "MISSING [promotion:" in capsys.readouterr().out


def test_every_repeated_statement_of_a_run_figure_has_its_own_row():
    """A figure the README states twice is two claims; one row per copy, or
    an edit to one copy passes on the strength of the other."""
    tags = {tag for tag, _ in check_readme_numbers.emit()}
    assert {"prose:instruction-following", "block:real-run-header",
            "prose:paid-run-scope", "prose:outcome-signal",
            "command:offline-runs", "prose:headline-calls",
            "prose:offline-runs", "prose:denominator",
            "prose:limits-cache-write"} <= tags, tags
