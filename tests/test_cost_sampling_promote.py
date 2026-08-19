"""Cost attribution, sampling capture, and what a promoted trace cannot be."""

from __future__ import annotations

import copy

import pytest

from obs import cost, sampling
from obs.promote import promote, promote_all
from obs.redact import EVERYTHING
from obs.runs import corpus, make_run
from obs.spans import OP_CHAT, USAGE_INPUT


# -- cost --------------------------------------------------------------------

def test_summing_spans_underbills_because_cached_tokens_are_absent():
    """The headline cost claim, in the direction the PAID RUN established.

    Asserting an over-bill would require modeling a cache hit as a discount
    on tokens that are still in the trace.
    A real provider does not do that: it reports the cached prefix in a
    separate counter and EXCLUDES it from input_tokens, and no GenAI attribute
    carries that counter. Measured on 20 real calls: input_tokens 143,
    cache_read_input_tokens 2,579. So the tokens are missing rather than
    mispriced, and a trace-derived total is too low.
    """
    traces = corpus(200)
    d = cost.drift(traces)
    assert d["error_usd"] < 0, "expected an under-bill"
    gap = cost.attribution_gap(traces)
    assert gap["cached_input_tokens"] > 0
    assert abs(abs(d["error_usd"]) - gap["cached_underbill_usd"]) < 2e-4


def test_cached_tokens_are_absent_from_the_span_and_not_discounted_in_it():
    """Pins the provider semantics the correction turned on.

    If a later change puts the cached tokens back into gen_ai.usage.input_tokens
    the cost finding silently inverts, and every number in the README with it.
    """
    cached = [t for t in corpus(200)
              if t.truth["true_usage"]["cached_input_tokens"] > 0]
    assert cached, "no cached runs in 200; the rate constant is wrong"
    tr = cached[0]
    span_in = sum(int(s.attributes.get(USAGE_INPUT, 0))
                  for s in tr.by_kind(OP_CHAT))
    assert span_in == tr.truth["true_usage"]["input_tokens"]
    assert tr.truth["true_usage"]["cached_input_tokens"] not in (
        [int(s.attributes.get(USAGE_INPUT, 0)) for s in tr.by_kind(OP_CHAT)])


def test_deduping_retries_makes_the_error_worse_not_better():
    """Mutation check on a tempting fix. Deduping retried calls is the obvious
    correction, and it moves the total further from the truth: those attempts
    really were billed. Both effects now push the same way, so there is no
    cancellation to hide behind."""
    traces = corpus(200)
    summed = cost.drift(traces)
    deduped = cost.drift(traces, dedupe_retries=True)
    assert deduped["error_usd"] < summed["error_usd"] < 0
    gap = cost.attribution_gap(traces)
    assert gap["retry_input_tokens"] > 0
    # The extra error is exactly the tokens the dedupe discarded. Compared at
    # the precision drift() reports at, which is 4 decimals.
    assert abs(abs(summed["error_usd"] - deduped["error_usd"])
               - gap["retry_underbill_usd_if_deduped"]) < 2e-4


def test_retried_model_calls_are_really_in_the_trace_twice():
    retried = [t for t in corpus(200)
               if t.truth["true_usage"]["model_attempts"] > 1]
    assert retried, "no model retries in 200 runs; the rate constant is wrong"
    tr = retried[0]
    chats = list(tr.by_kind(OP_CHAT))
    assert sum(1 for s in chats if s.is_error()) >= 1
    assert all(int(s.attributes.get(USAGE_INPUT, 0)) > 0 for s in chats), (
        "a failed attempt with no input tokens would make the retry finding "
        "free rather than measured")


def test_an_unpriced_model_is_refused():
    """Guessing a price is how a cost report becomes fiction."""
    with pytest.raises(KeyError):
        cost.from_spans(make_run(0), model="model-nobody-priced")


# -- sampling ----------------------------------------------------------------

def test_head_sampling_keeps_its_rate_of_failures_and_no_more():
    rows = {r["policy"]: r for r in sampling.evaluate(corpus(500))}
    head = rows["head_10pct"]
    assert head["failure_capture_pct"] <= 25.0, (
        "head sampling cannot preferentially keep failures; it decides before "
        "anything has gone wrong")


def test_the_intuitive_tail_policy_keeps_retries_rather_than_failures():
    """The trap, asserted. 'Keep any trace with an error span' sounds like the
    fix and is not: retries leave ERROR spans on runs that succeeded, and the
    silent wrong-answer failure has no error span at all."""
    rows = {r["policy"]: r for r in sampling.evaluate(corpus(500))}
    err = rows["tail_any_error_span"]
    assert err["kept"] > err["failures_kept"] * 3, (
        "the error-span policy should be dominated by successful retried runs")
    assert err["failure_capture_pct"] < 100.0


def test_only_an_outcome_signal_captures_every_failure():
    rows = {r["policy"]: r for r in sampling.evaluate(corpus(500))}
    assert rows["tail_run_failed"]["failure_capture_pct"] == 100.0
    assert rows["tail_run_failed"]["buffers_traces"] is True


# -- promotion ---------------------------------------------------------------

def test_no_trace_promotes_into_a_runnable_case():
    """A trace records what the model said. A regression case needs what it
    should have said, and nothing in telemetry carries that."""
    out = promote_all(corpus(100))
    assert out["runnable"] == 0
    assert out["cases"] == 100


def test_the_missing_expectation_is_always_the_blocker():
    case = promote(make_run(3))
    assert any("no expected output" in b for b in case.blocked_on)
    assert case.observed_output is not None, (
        "the trace does carry what the model said; only the expectation is "
        "missing, and conflating the two would overstate the problem")


def test_redacting_the_trace_breaks_the_replay_it_would_have_supported():
    """The tension this repository does not resolve: the redaction section
    recommends masking identifiers, and the promotion section needs them."""
    trace = copy.deepcopy(make_run(5))
    EVERYTHING.apply(trace)
    case = promote(trace, redacted=True)
    assert any("input is redacted" in b for b in case.blocked_on)
