"""Cost attribution, sampling capture, and what a promoted trace cannot be."""

from __future__ import annotations

import copy

import pytest

from obs import cost, sampling
from obs.promote import promote, promote_all
from obs.redact import EVERYTHING
from obs.runs import (SYSTEM_PREFIX_TOKENS, _IN_RANGE, corpus,  # noqa: PLC2701
                      make_run)
from obs.spans import OP_CHAT, USAGE_INPUT


# -- cost --------------------------------------------------------------------

def test_summing_spans_underbills_because_cached_tokens_are_absent():
    """The headline cost claim, in the direction the PAID RUN established.

    Asserting an over-bill would require modeling a cache hit as a discount
    on tokens that are still in the trace.
    A real provider does not do that: it reports the cached prefix in a
    separate counter and EXCLUDES it from its own input_tokens, and the SDK
    instrumentations copy that field onto gen_ai.usage.input_tokens unchanged.
    Measured on 20 real calls: input_tokens 143, cache_read_input_tokens 2,579.
    So the tokens are missing rather than mispriced, and a trace-derived total
    is too low.

    The conventions have defined gen_ai.usage.cache_read.input_tokens since
    2026-08-20 and ask that input_tokens include cached tokens. This corpus
    models the non-conformant collector because that is the one that ships;
    see obs/cost.py's PROMPT CACHING note.
    """
    traces = corpus(200)
    d = cost.drift(traces)
    assert d["error_usd"] < 0, "expected an under-bill"
    gap = cost.attribution_gap(traces)
    assert gap["cached_input_tokens"] > 0
    assert abs(abs(d["error_usd"]) - gap["cached_underbill_usd"]) < 2e-4


def test_cached_tokens_are_absent_from_the_span_and_not_discounted_in_it():
    """Pins the provider semantics the cost finding turns on.

    If a later change puts the cached tokens back into gen_ai.usage.input_tokens
    the cost finding silently inverts, and every number in the README with it.

    This does not compare the span to the truth, because make_run writes both:
    an edit that adds the cached prefix to the span and to
    `truth["input_tokens"]` together would move both sides and pass such a
    check.

    So the anchor is a constant neither side can quietly restate: `_IN_RANGE`,
    the declared per-call input range the generator draws from. A cached prefix
    is SYSTEM_PREFIX_TOKENS, larger than the whole range, so a span that had it
    folded in could not be inside the range any more. That holds however the
    truth is written.
    """
    cached = [t for t in corpus(200)
              if t.truth["true_usage"]["cached_input_tokens"] > 0]
    assert cached, "no cached runs in 200; the rate constant is wrong"
    assert SYSTEM_PREFIX_TOKENS > _IN_RANGE[1], (
        "this test's anchor assumes a cached prefix is bigger than any single "
        "call's uncached input; if that stops being true, re-derive it")

    for tr in cached[:20]:
        assert tr.truth["true_usage"]["cached_input_tokens"] == \
            SYSTEM_PREFIX_TOKENS
        for span in tr.by_kind(OP_CHAT):
            got = int(span.attributes.get(USAGE_INPUT, 0))
            assert _IN_RANGE[0] <= got <= _IN_RANGE[1], (
                f"{tr.trace_id}/{span.span_id}: gen_ai.usage.input_tokens is "
                f"{got}, outside the generator's declared range {_IN_RANGE}. "
                "The cached prefix has been folded into the span, which is the "
                "provider semantics this repository measured and inverts the "
                "cost finding.")


def test_the_paid_run_evidence_shows_the_provider_excluding_the_cache():
    """The semantics above, checked against evidence THIS REPOSITORY DID NOT
    WRITE.

    Every other cost assertion grades a corpus against a truth the same module
    produced. This one reads the shipped record of the paid run: on a call
    served from cache, the provider's own `input_tokens` is a small remainder
    and the cached tokens sit in a counter beside it. That is the fact the
    whole cost section rests on, and audit/real_run.json is the only place in
    this repository where it is attested rather than modeled.
    """
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    real = json.loads((root / "audit" / "real_run.json").read_text())
    served_from_cache = [r for r in real["records"]
                         if "error" not in r
                         and r["usage"]["cache_read_input_tokens"] > 0]
    assert served_from_cache, "no cache hits in the shipped paid run"
    for r in served_from_cache:
        u = r["usage"]
        assert u["input_tokens"] < u["cache_read_input_tokens"], (
            f"run {r['run']}: input_tokens {u['input_tokens']} is not the "
            f"small remainder beside cache_read {u['cache_read_input_tokens']}; "
            "the provider would be including the cached prefix after all")
        assert u["input_tokens"] <= _IN_RANGE[1], (
            f"run {r['run']}: the real uncached remainder {u['input_tokens']} "
            f"is outside the range the corpus models, {_IN_RANGE}")


def test_deduping_retries_makes_the_error_worse_not_better():
    """Mutation check on a tempting fix. Deduping retried calls is the obvious
    correction, and it moves the total further from the truth: those attempts
    really were billed. Both effects push the same way, so there is no
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

# How many seeds the head-sampling test averages over, and the tolerance in
# percentage points. 21 failures on ONE seed is far too few to say anything:
# the per-seed capture rate can only take the values 0/21, 1/21, 2/21 ... so a
# single seed lands on 0.0%, 4.8%, 9.5%, 14.3% or 19.0%, and NONE of those
# distinguishes an unbiased 10% sampler from one cheating by 2.4x.
#
# Both numbers were chosen by measurement. 200 seeds is 4,200 draws, whose
# mean has a standard error of 0.46 points, and the unbiased sampler measures
# 9.90% capture against 9.91% kept, a gap of 0.01. The cheats measure 14.00%
# (1.5x), 18.81% (2.0x) and 4.98% (0.5x). So 2.5 points is 5.4 standard
# errors from honest and at least 1.5 points clear of the nearest cheat: not
# flaky, and not slack enough to let one through.
# Neither number can be relaxed on its own. At 40 seeds the standard error is
# 1.04 points, and a bound wide enough not to be flaky there (4 points) admits
# the 1.5x cheat outright. Fewer seeds therefore require a bound that
# does not work; more seeds cost nothing measurable, the whole test runs in
# well under a second.
_SAMPLING_SEEDS = 200
_SAMPLING_TOLERANCE_PP = 2.5


def test_head_sampling_keeps_its_rate_of_failures_and_no_more():
    """The claim is TWO-SIDED, so both sides are asserted.

    "Head sampling keeps its rate of the failures AND NO MORE" says the failure
    capture rate EQUALS the sampling rate. A one-sided ceiling does not say
    that. `failure_capture_pct <= 25.0` against a 10% policy is a 2.5x ceiling,
    and a sampler that peeks at the outcome it cannot have seen --

        rng.random() < (rate * 2.4 if trace.failed() else rate)

    -- passes such a ceiling at 19.0%, which moves the README's row from 1/21
    to 4/21 with the whole suite green. A head sampler CANNOT do that; it
    decides at the root span, before anything has gone wrong. So this pins the
    equality, from both directions:

      1 the mean failure capture rate is the sampling rate, within tolerance
      2 it equals the rate at which the policy keeps EVERYTHING, which is the
        "for the same reason" clause of the README sentence

    The tolerance and the seed count are measured; see the constants above.
    """
    traces = corpus(500)
    failures = sum(1 for t in traces if t.failed())
    assert failures, "no failures in 500 runs; the rate constant is wrong"

    head_only = (sampling.head(0.10),)
    kept_failures = kept_total = 0
    for seed in range(_SAMPLING_SEEDS):
        row = sampling.evaluate(traces, policies=head_only, seed=seed)[0]
        kept_failures += row["failures_kept"]
        kept_total += row["kept"]

    capture_pct = 100.0 * kept_failures / (failures * _SAMPLING_SEEDS)
    keep_pct = 100.0 * kept_total / (len(traces) * _SAMPLING_SEEDS)

    assert abs(capture_pct - 10.0) <= _SAMPLING_TOLERANCE_PP, (
        f"head sampling captured {capture_pct:.2f}% of failures at a 10% rate. "
        "A head sampler decides at the root span, before anything has gone "
        "wrong, so it cannot preferentially keep OR drop them.")
    assert abs(capture_pct - keep_pct) <= _SAMPLING_TOLERANCE_PP, (
        f"head sampling kept {keep_pct:.2f}% of all traces but "
        f"{capture_pct:.2f}% of the failures. It keeps its rate of the "
        "failures FOR THE SAME REASON it keeps its rate of everything, so "
        "these two numbers are one number.")


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
