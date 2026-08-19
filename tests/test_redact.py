"""Redaction coverage, and the checks that keep the coverage result honest."""

from __future__ import annotations

import copy

import pytest

from obs.redact import (CONTENT_AND_EVENTS, CONTENT_ATTRS, EVERYTHING, MASK,
                        NONE, POLICIES, PROMPT_ONLY, residual, scrub,
                        _surface_text)
from obs.runs import PLANTED_SURFACES, corpus


def _leaking_surfaces(policy, n=60):
    traces = [copy.deepcopy(t) for t in corpus(n)]
    for t in traces:
        policy.apply(t)
    return {r["surface"] for t in traces for r in residual(t)}


def test_scrub_removes_each_pattern_it_claims_to():
    text = "mail dana.w@northwind.example id CUS-123456 phone +1-555-0100"
    out = scrub(text)
    for leaked in ("dana.w@northwind.example", "CUS-123456", "+1-555-0100"):
        assert leaked not in out
    assert MASK in out


def test_the_unredacted_control_leaks_every_surface():
    """THE VACUITY CHECK. If the control arm did not leak everywhere, a policy
    could score well by measuring nothing. Every result in this repository is
    a difference from this row."""
    assert _leaking_surfaces(NONE) == set(PLANTED_SURFACES)


def test_redacting_everything_leaks_nothing():
    """The other end of the scale. If this fails, either a locator is wrong or
    a surface is not reachable by any policy, and the coverage numbers in
    between mean nothing."""
    assert _leaking_surfaces(EVERYTHING) == set()


def test_broader_policies_leak_a_strict_subset():
    """Each policy is pointed at everything the previous one was, plus more, so
    its leaking surfaces must be a subset. A crossover would mean a policy
    stopped covering something it used to."""
    order = [NONE, PROMPT_ONLY, CONTENT_ATTRS, CONTENT_AND_EVENTS, EVERYTHING]
    leaks = [_leaking_surfaces(p) for p in order]
    for narrower, broader in zip(leaks, leaks[1:]):
        assert broader < narrower, "a broader policy did not strictly improve"


def test_the_prompt_only_policy_misses_the_event_convention():
    """The headline claim, asserted rather than narrated: a redactor written
    against gen_ai.input.messages never visits the event that carries the same
    text under the older revision."""
    leaking = _leaking_surfaces(PROMPT_ONLY)
    assert "input_messages_attr" not in leaking
    assert "user_message_event" in leaking
    assert "choice_event" in leaking


def test_client_and_exception_surfaces_need_the_widest_policy():
    """The URL, the SQL and the stack trace are not GenAI attributes at all,
    so no content-shaped policy reaches them."""
    for surface in ("http_url", "db_statement", "exception_message",
                    "exception_stacktrace"):
        assert surface in _leaking_surfaces(CONTENT_AND_EVENTS)
        assert surface not in _leaking_surfaces(EVERYTHING)


def test_residual_grades_the_surface_and_not_the_span():
    """MUTATION CHECK for the defect this measurement actually had. The first
    version asked 'does this span still contain the customer anywhere?', so a
    leak on one surface marked every other planted surface in the same span as
    leaking too, and three different policies produced identical numbers.

    The plan span carries two surfaces. Redact one; the other must still be
    reported, and the redacted one must not be.
    """
    trace = copy.deepcopy(corpus(1)[0])
    PROMPT_ONLY.apply(trace)
    surfaces = {r["surface"] for r in residual(trace)}
    assert "user_message_event" in surfaces
    assert "input_messages_attr" not in surfaces


def test_an_unknown_surface_is_an_error_not_a_pass():
    """A typo in a surface name must not silently score as 'clean'."""
    span = corpus(1)[0].spans[0]
    with pytest.raises(KeyError):
        _surface_text(span, "surface_that_does_not_exist")


def test_every_policy_is_measured_against_the_same_denominator():
    counts = []
    for pol in POLICIES:
        traces = [copy.deepcopy(t) for t in corpus(20)]
        for t in traces:
            pol.apply(t)
        counts.append(sum(len(t.truth["planted"]) for t in traces))
    assert len(set(counts)) == 1, "a policy changed the planted count"
