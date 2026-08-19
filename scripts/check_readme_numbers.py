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
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The eleven surfaces the redaction table counts. This list is the fixed
#: denominator and the run does not supply it, because the table's "9/11" is a
#: claim about how many of a
#: KNOWN set are still leaking, and a surface silently disappearing from the
#: corpus would otherwise improve every policy's score.
SURFACES = 11


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
    out.append(("prose:identifier",
                "the account number came back in the model's own text %d out "
                "of %d times" % (real["identifier_in_output"], real["runs"])))
    return out


def emit():
    return rows_redaction() + rows_sampling() + prose_figures()


def squash(text):
    return re.sub(r"\s+", " ", text.replace("**", ""))


def main():
    derived = emit()
    if "--emit" in sys.argv:
        for tag, row in derived:
            print("%s\n%s" % (tag, row))
        return 0
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = squash(fh.read())
    missing = [(t, r) for t, r in derived if squash(r) not in readme]
    for tag, row in missing:
        print("MISSING [%s]\n  %s" % (tag, row))
    tables = sum(1 for t, _ in derived if not t.startswith("prose:"))
    print("\n%d of %d derived figures found verbatim in README.md "
          "(%d table rows, %d in prose)"
          % (len(derived) - len(missing), len(derived), tables,
             len(derived) - tables))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
