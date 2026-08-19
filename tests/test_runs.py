"""The corpus is the ground truth, so it is the first thing that must be right.

Every number this repository publishes is graded against `trace.truth`. If the
truth and the spans can drift apart, every result is confident and wrong, a
defect this project's sibling repositories keep finding in themselves.
"""

from __future__ import annotations

import json

from obs.runs import PLANTED_SURFACES, corpus, make_run
from obs.redact import SURFACE_LOCATORS, _surface_text
from obs.spans import OP_CHAT, USAGE_INPUT, USAGE_OUTPUT


def test_runs_are_deterministic():
    """Same index, same trace. Otherwise no two measurements are comparable."""
    a, b = make_run(7), make_run(7)
    assert [s.span_id for s in a.spans] == [s.span_id for s in b.spans]
    assert a.truth == b.truth


def test_every_planted_surface_actually_occurs_in_the_corpus():
    """A surface nobody plants would report a perfect 0 leaks forever.

    This is the vacuity check for the whole redaction result: the denominator
    has to be real on every surface, or a policy gets credit for covering
    something that was never there.
    """
    seen = set()
    for trace in corpus(120):
        seen.update(item["surface"] for item in trace.truth["planted"])
    missing = set(PLANTED_SURFACES) - seen
    assert not missing, f"never planted, so never measurable: {missing}"


def test_every_planted_identifier_is_really_readable_where_it_was_planted():
    """The truth must describe the spans, not the intention.

    If planting says 'the email is in the db statement' and it is not, the
    unredacted control arm silently under-reports and every policy looks
    better than it is.
    """
    for trace in corpus(60):
        identity = trace.truth["identity"]
        by_id = {s.span_id: s for s in trace.spans}
        for item in trace.truth["planted"]:
            text = _surface_text(by_id[item["span_id"]], item["surface"])
            assert (identity["email"] in text
                    or identity["customer_id"] in text), (
                f"{trace.trace_id} claims {item['surface']} carries the "
                f"identity and it does not")


def test_every_surface_has_a_locator():
    """A planted surface with no locator would raise mid-measurement."""
    assert set(PLANTED_SURFACES) <= set(SURFACE_LOCATORS)


def test_the_true_usage_matches_what_the_spans_report():
    """Ground truth and telemetry must agree about the tokens that are in the
    trace. They are allowed to disagree about caching, that is the finding, so
    this asserts only the totals the conventions actually carry."""
    for trace in corpus(60):
        span_in = sum(int(s.attributes.get(USAGE_INPUT, 0))
                      for s in trace.by_kind(OP_CHAT))
        span_out = sum(int(s.attributes.get(USAGE_OUTPUT, 0))
                       for s in trace.by_kind(OP_CHAT))
        assert span_in == trace.truth["true_usage"]["input_tokens"]
        assert span_out == trace.truth["true_usage"]["output_tokens"]


def test_the_silent_failure_leaves_no_error_span():
    """The failure the whole stack exists to catch must be invisible to status.

    If a failed run also carried an ERROR span, the sampling result would be
    an artifact: 'tail on error' would look like it works.
    """
    failures = [t for t in corpus(300) if t.failed()]
    assert failures, "no failures in 300 runs; the rate constant is wrong"
    for trace in failures:
        # The span that produced the answer is the LAST chat span, whatever its
        # status. A failed run may ALSO have had a retried model call earlier,
        # which does leave an ERROR span, that is the confound the sampling
        # result is about, so the assertion is about the answering span.
        #
        # Select it by position, not by status. Filtering to non-error spans
        # first and then asserting the result is not an error is a tautology:
        # it holds however the corpus marks the answering span, so the claim
        # this test exists for, that the silent failure carries no error
        # status, would survive its own negation.
        answered = list(trace.by_kind(OP_CHAT))[-1]
        assert not answered.is_error(), (
            f"{trace.trace_id}: the answering span is marked ERROR, so this "
            "failure is not silent and the sampling result is an artifact")


def test_a_failed_run_really_answers_with_the_wrong_data():
    """The planted failure has to BE a failure, not a label on a good run."""
    for trace in corpus(300):
        if not trace.failed():
            continue
        out = trace.spans[-1].attributes.get("gen_ai.output.messages", "")
        content = json.loads(out)[0]["parts"][0]["content"]
        assert trace.truth["identity"]["customer_id"] not in content
        # and every other copy of the answer agrees with it
        for ev in trace.spans[-1].events:
            if "content" in ev.attributes:
                assert trace.truth["identity"]["customer_id"] not in \
                    ev.attributes["content"]
