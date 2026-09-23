"""Re-derive every published number in README.md from audit/*.json and diff them.

A README is prose and drifts; audit/*.json is evidence and does not. This
script rebuilds each figure from the JSON and asserts the exact string is
present in the README, so a re-run that shifts a figure fails loudly instead of
leaving the document quietly wrong.

    python3 scripts/check_readme_numbers.py            check
    python3 scripts/check_readme_numbers.py --emit     print what it derives

Whitespace AND emphasis are normalized on both sides. Which cell is bold is a
choice about where a reader's eye should land and no evidence file knows it;
demanding the markup would certify the typography instead of the figures.

The count is printed whether OR NOT anything is missing, so a version of this
script that quietly stopped deriving half of them is visible rather than clean.

It checks enumerations as well as figures. A surface added to obs/runs.py
changes the "9 of 11" figures and also needs a row in the README's surface
table, so the table is rebuilt from PLANTED_SURFACES and SURFACE_LOCATORS: the
row count must match, and every surface's attribute or event name must appear.
"""

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CI runs this step with no PYTHONPATH, so the import below needs the root on
# the path. Importing obs is what makes SURFACES a derived figure instead of a
# hand-typed one.
sys.path.insert(0, ROOT)

from obs.redact import SURFACE_LOCATORS                          # noqa: E402
from obs.runs import PLANTED_SURFACES                            # noqa: E402

#: The surfaces the redaction table counts, derived from the corpus instead of
#: typed here. It is the fixed denominator: the table's "9/11" is a claim about
#: how many of a KNOWN set are still leaking, and a surface silently
#: disappearing from the corpus would otherwise improve every policy's score.
SURFACES = len(PLANTED_SURFACES)


def load(name):
    with open(os.path.join(ROOT, "audit", name), encoding="utf-8") as fh:
        return json.load(fh)


def rows_redaction():
    """What each redaction policy leaves readable, and on how many surfaces."""
    labels = {"none": "none (control)",
              "prompt_and_completion": "prompt and completion",
              "all_content_attributes": "all content attributes",
              "content_attributes_and_events": "content attributes and events",
              "every_string_in_the_span": "every string in the span"}
    out = []
    for r in load("offline.json")["redaction"]:
        leaking = len(r["surfaces_leaking"])
        total = leaking + len(r["surfaces_clean"])
        if total != SURFACES:
            out.append(("redaction:" + r["policy"],
                        "UNDERIVABLE: %d surfaces in the run, %d in the table"
                        % (total, SURFACES)))
            continue
        out.append(("redaction:" + r["policy"],
                    "| %s | %d/%d | %d/%d |"
                    % (labels[r["policy"]], r["leaked"], r["planted"],
                       leaking, SURFACES)))
    return out


def rows_sampling():
    """What each sampling policy keeps, and how many failures it keeps."""
    labels = {"keep_everything": "keep everything",
              "head_10pct": "head 10%",
              "head_1pct": "head 1%",
              "tail_any_error_span": "tail on any error span",
              "tail_run_failed": "tail on run outcome"}
    out = []
    for s in load("offline.json")["sampling"]:
        pct = 100.0 * s["failures_kept"] / s["failures_total"]
        out.append(("sampling:" + s["policy"],
                    "| %s | %d (%s%%) | %d/%d (%s%%) | %s |"
                    % (labels[s["policy"]], s["kept"], _trim(s["kept_pct"]),
                       s["failures_kept"], s["failures_total"], _trim(pct),
                       "yes" if s["buffers_traces"] else "no")))
    return out


def _trim(x):
    """100.0 prints as 100, 8.6 as 8.6, which is how the table writes them."""
    return ("%g" % round(x, 1))


def prose_figures():
    off = load("offline.json")["cost"]
    real = load("real_run.json")
    out = [("prose:offline-cost",
            "the same effect is worth %s%%, and deduping retried calls, the "
            "obvious correction, makes it %s%%"
            % (_trim(off["summed"]["error_pct"]),
               _trim(off["deduped"]["error_pct"])))]
    traced, billed = real["traced_usd"], real["billed_usd"]
    out.append(("prose:real-cost",
                "cost from the trace $%.4f cost from the usage report $%.4f "
                "the trace is off by %.2f%% cache read tokens %s"
                % (traced, billed, 100 * (traced - billed) / billed,
                   "{:,}".format(real["cache_read_tokens"]))))
    # The denominator is runs that succeeded, counted from the records.
    # `real["runs"]` is runs requested. A paid run with two API errors would
    # otherwise publish "18 out of 20", which a reader takes as two
    # non-reproductions instead of two calls that never reached the model.
    # Counting the records without an "error" key derives it from the
    # evidence and needs no new field in the shipped file.
    succeeded = sum(1 for r in real["records"] if "error" not in r)
    out.append(("prose:identifier",
                "the account number came back in the model's own text %d out "
                "of %d times" % (real["identifier_in_output"], succeeded)))
    return out


def surface_name(loc):
    """The name the README's surface table identifies a locator by.

    An "attr" locator is named by its attribute key. An "event" locator is
    named by its event name when the field it reads is the convention's
    generic `content` placeholder, as `gen_ai.user.message` does, and by its
    field name when that field is itself a dotted attribute, as `exception.message` and
    `exception.stacktrace` are. Both exception surfaces share the one
    `exception` event, so naming them by the event would make two surfaces
    indistinguishable and let either row go missing unnoticed.
    """
    if loc[0] == "event" and "." in loc[2]:
        return loc[2]
    return loc[1]


def rows_surface_table():
    """Every planted surface must have a row in the README's surface table.

    Derives the enumeration from obs/redact.SURFACE_LOCATORS rather than
    restating it, so a surface added to the corpus and not to the document is
    a failure here.
    """
    out = []
    for surface in PLANTED_SURFACES:
        out.append(("surface:" + surface,
                    "| `%s`" % surface_name(SURFACE_LOCATORS[surface])))
    return out


def surface_table_rows(readme_text):
    """Count the data rows of the surface table, by its header and separator.

    Counted from the document, so a row deleted from the table is a
    failure here even if every name still appears somewhere in the
    prose.
    """
    lines = readme_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("| surface |"):
            n = 0
            for row in lines[i + 2:]:          # skip the |---| separator
                if not row.startswith("|"):
                    break
                n += 1
            return n
    return None


def test_count():
    """The suite size, counted from the test files.

    README's Reproducing block prints "# NN tests", and a typed count drifts
    the moment a test is added. Counted by parsing the test files, so a renamed-but-not-added test cannot
    inflate it and a commented-out one cannot hide in it.
    """
    n = 0
    tests = os.path.join(ROOT, "tests")
    for name in sorted(os.listdir(tests)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        with open(os.path.join(tests, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        n += sum(1 for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test_"))
    return [("prose:test-count", "# %d tests" % n)]


def emit():
    return (rows_redaction() + rows_sampling() + prose_figures()
            + rows_surface_table() + test_count())


def _fenced_block(text, after):
    """The ``` block that follows the line `after`, without its fences."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == after:
            out = []
            for row in lines[i + 1:]:
                if row.startswith("```"):
                    return out
                out.append(row)
    return None


def check_sample_run():
    """SAMPLE_RUN.md says "Every block below is captured output". Check it.

    This re-runs the command the document names and diffs its output against
    the block. The `wrote <path>` line is excluded because the path is this
    check's temporary file rather than the audit file,
    and the elapsed time in the pytest block is excluded because it is the one
    genuinely non-deterministic character in either block; everything else must
    match exactly.

    Returns a list of complaints, empty when the document is what it claims.
    """
    import subprocess
    import tempfile

    path = os.path.join(ROOT, "SAMPLE_RUN.md")
    with open(path, encoding="utf-8") as fh:
        doc = fh.read()
    bad = []

    # -- the offline block ---------------------------------------------------
    cmd_line = ("$ .venv/bin/python scripts/offline_demo.py --runs 500 "
                "--json audit/offline.json")
    block = _fenced_block(doc, cmd_line)
    if block is None:
        bad.append("SAMPLE_RUN.md has no block for %r" % cmd_line)
    else:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            run = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts",
                                              "offline_demo.py"),
                 "--runs", "500", "--json", tmp.name],
                capture_output=True, text=True, cwd=ROOT)
        if run.returncode != 0:
            bad.append("offline_demo.py failed: %s" % run.stderr.strip()[:200])
        else:
            def strip_wrote(rows):
                return [r for r in rows if not r.startswith("wrote ")]
            want = strip_wrote(run.stdout.rstrip("\n").split("\n"))
            got = strip_wrote([r for r in block if r != cmd_line])
            if got != want:
                for i, (a, b) in enumerate(zip(got, want)):
                    if a != b:
                        bad.append("SAMPLE_RUN.md offline block line %d\n"
                                   "    document: %r\n    a fresh run: %r"
                                   % (i + 1, a, b))
                        break
                else:
                    bad.append("SAMPLE_RUN.md offline block has %d lines, a "
                               "fresh run produces %d" % (len(got), len(want)))

    # -- the pytest block ----------------------------------------------------
    # Not re-run: pytest cannot report on itself from inside itself here.
    # Instead the summary line and the number of progress dots are checked
    # against the derived test count.
    n = int(test_count()[0][1].split()[1])
    block = _fenced_block(doc, "$ .venv/bin/python -m pytest -q")
    if block is None:
        bad.append("SAMPLE_RUN.md has no pytest block")
    else:
        body = "\n".join(block)
        if ("%d passed" % n) not in body:
            bad.append("SAMPLE_RUN.md's pytest block does not say '%d passed'; "
                       "the suite has %d tests" % (n, n))
        # Count dots only on progress lines (rows made of dots, brackets,
        # digits and percent), so the "0.19s" in the summary line and any
        # prose full stop cannot be mistaken for a test result.
        dots = sum(row.count(".") for row in block
                   if set(row.strip()) <= set(". []%0123456789"))
        if dots != n:
            bad.append("SAMPLE_RUN.md's pytest block shows %d progress dots "
                       "for a %d-test suite; pytest prints one per test"
                       % (dots, n))
    return bad


def squash(text):
    return re.sub(r"\s+", " ", text.replace("**", ""))


def main():
    derived = emit()
    if "--emit" in sys.argv:
        for tag, row in derived:
            print("%s\n%s" % (tag, row))
        return 0
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        raw = fh.read()
    readme = squash(raw)
    missing = [(t, r) for t, r in derived if squash(r) not in readme]
    for tag, row in missing:
        print("MISSING [%s]\n  %s" % (tag, row))

    # The enumeration check, reported separately because "every name appears"
    # and "the table has the right number of rows" are different claims and
    # only the second one catches a row that was never added.
    rows = surface_table_rows(raw)
    bad_table = False
    if rows is None:
        print("MISSING [surface-table]\n  no '| surface |' table found in "
              "README.md; the enumeration cannot be checked")
        bad_table = True
    elif rows != SURFACES:
        print("MISSING [surface-table]\n  the surface table has %d data rows "
              "and obs.runs.PLANTED_SURFACES has %d entries" % (rows, SURFACES))
        bad_table = True
    else:
        print("surface table: %d data rows == %d planted surfaces"
              % (rows, SURFACES))

    sample_problems = check_sample_run()
    for problem in sample_problems:
        print("MISMATCH [sample-run]\n  %s" % problem)
    if not sample_problems:
        print("SAMPLE_RUN.md: both captured blocks reproduce")

    tables = sum(1 for t, _ in derived if not t.startswith("prose:"))
    print("\n%d of %d derived figures found verbatim in README.md "
          "(%d table rows, %d in prose)"
          % (len(derived) - len(missing), len(derived), tables,
             len(derived) - tables))
    return 1 if (missing or bad_table or sample_problems) else 0


if __name__ == "__main__":
    sys.exit(main())
