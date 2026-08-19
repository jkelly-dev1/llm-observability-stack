"""Redaction policies for spans, and the surfaces each one actually reaches.

The claim this module exists to measure. "We redact PII before it reaches the
traces" is a sentence every team says, and it is true of the place they looked
at when they wrote the redactor. A trace of one agent turn carries the same
customer identity in eleven places (obs/runs.PLANTED_SURFACES), spread across two
different content-capture conventions, an HTTP client's URL, a database
client's statement text, and an exception message that quotes the input back.

A Policy is a set of surfaces it visits, not a set of patterns it matches.
That is the design decision this module makes, and the one worth arguing with.
A regex that finds every email in the world still leaks if it is only ever
handed `gen_ai.input.messages`. So each policy below declares the span fields
it walks, and the measurement reports residual identifiers per surface.

Nothing here is a novel redaction technique. The matcher is deliberately a
plain, strong regex set: the finding is about coverage, and a weak matcher
would confound the two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .spans import (DB_STATEMENT, EVENT_ASSISTANT, EVENT_CHOICE, EVENT_EXCEPTION,
                    EVENT_USER, EXCEPTION_MESSAGE, EXCEPTION_STACKTRACE,
                    HTTP_URL, INPUT_MESSAGES, OUTPUT_MESSAGES, Span, Trace)

MASK = "[REDACTED]"

# Deliberately broad. If the measurement is going to say a surface leaked, it
# must not be because the pattern was weak on that surface.
PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),   # email
    re.compile(r"\bCUS-\d{6}\b"),                                    # cust id
    re.compile(r"\+\d[\d-]{7,}\d"),                                  # phone
)

# Every attribute key that can carry content, and the two events, so that a
# policy can be defined by which of these it is pointed at.
ATTR_CONTENT_KEYS = (INPUT_MESSAGES, OUTPUT_MESSAGES,
                     "gen_ai.tool.call.arguments", "gen_ai.tool.call.result")
ATTR_CLIENT_KEYS = (HTTP_URL, DB_STATEMENT)
EVENT_CONTENT_FIELDS = ("content",)
EVENT_ERROR_FIELDS = (EXCEPTION_MESSAGE, EXCEPTION_STACKTRACE)


def scrub(text: str) -> str:
    for pat in PATTERNS:
        text = pat.sub(MASK, text)
    return text


@dataclass(frozen=True)
class Policy:
    """Which parts of a span the redactor is pointed at."""

    name: str
    attr_keys: tuple = ()
    walk_all_attrs: bool = False
    event_content: bool = False
    event_errors: bool = False
    # A note on the name identifies WHO writes this policy in real life, so the
    # result reads as a statement about practice rather than about strawmen.
    note: str = ""

    def apply(self, trace: Trace) -> Trace:
        for span in trace.spans:
            self._span(span)
        return trace

    def _span(self, span: Span) -> None:
        keys = span.attributes.keys() if self.walk_all_attrs else self.attr_keys
        for key in list(keys):
            value = span.attributes.get(key)
            if isinstance(value, str):
                span.attributes[key] = scrub(value)
        for ev in span.events:
            if self.event_content:
                for f in EVENT_CONTENT_FIELDS:
                    if isinstance(ev.attributes.get(f), str):
                        ev.attributes[f] = scrub(ev.attributes[f])
            if self.event_errors:
                for f in EVENT_ERROR_FIELDS:
                    if isinstance(ev.attributes.get(f), str):
                        ev.attributes[f] = scrub(ev.attributes[f])


# The policies, ordered by how much of the trace they were pointed at. Each one
# is something a real team ships, not an invented weak baseline.
NONE = Policy("none", note="no redaction: the control arm")

PROMPT_ONLY = Policy(
    "prompt_and_completion",
    attr_keys=(INPUT_MESSAGES, OUTPUT_MESSAGES),
    note="redact the prompt and the completion. The instrumentation docs show "
         "these two attributes, so this is what most redactors are pointed at")

CONTENT_ATTRS = Policy(
    "all_content_attributes",
    attr_keys=ATTR_CONTENT_KEYS,
    note="the above plus tool arguments and tool results, still attributes only")

CONTENT_AND_EVENTS = Policy(
    "content_attributes_and_events",
    attr_keys=ATTR_CONTENT_KEYS, event_content=True,
    note="adds the older convention's message events, which a redactor written "
         "against the newer revision never visits")

EVERYTHING = Policy(
    "every_string_in_the_span",
    walk_all_attrs=True, event_content=True, event_errors=True,
    note="walk every string attribute and every event field. The only policy "
         "that covers the client and exception surfaces")

POLICIES = (NONE, PROMPT_ONLY, CONTENT_ATTRS, CONTENT_AND_EVENTS, EVERYTHING)


# Where each planted surface lives. The measurement grades the SURFACE, not the
# span. Asking "does this span still contain the customer anywhere?" Marks
# every planted item in a span as leaking the moment any one of them does, and
# makes three different policies produce identical numbers. A per-surface
# locator is the difference between a measurement and a mood.
SURFACE_LOCATORS = {
    "input_messages_attr": ("attr", INPUT_MESSAGES),
    "output_messages_attr": ("attr", OUTPUT_MESSAGES),
    "tool_call_arguments": ("attr", "gen_ai.tool.call.arguments"),
    "tool_result": ("attr", "gen_ai.tool.call.result"),
    "http_url": ("attr", HTTP_URL),
    "db_statement": ("attr", DB_STATEMENT),
    "user_message_event": ("event", EVENT_USER, "content"),
    "choice_event": ("event", EVENT_CHOICE, "content"),
    "assistant_message_event": ("event", EVENT_ASSISTANT, "content"),
    "exception_message": ("event", EVENT_EXCEPTION, EXCEPTION_MESSAGE),
    "exception_stacktrace": ("event", EVENT_EXCEPTION, EXCEPTION_STACKTRACE),
}


def _surface_text(span: Span, surface: str) -> str:
    loc = SURFACE_LOCATORS.get(surface)
    if loc is None:
        raise KeyError(f"no locator for surface {surface!r}")
    if loc[0] == "attr":
        value = span.attributes.get(loc[1])
        return value if isinstance(value, str) else ""
    parts = []
    for ev in span.events:
        if ev.name == loc[1]:
            value = ev.attributes.get(loc[2])
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def residual(trace: Trace) -> list[dict]:
    """Planted identifiers still readable on the surface they were planted in.

    Grades against `trace.truth["planted"]`, so a leak is reported as "this
    identifier, in this span, on this surface" rather than as a count. A count
    is what makes a redaction result unarguable and useless.
    """
    out = []
    by_id = {s.span_id: s for s in trace.spans}
    identity = trace.truth["identity"]
    for item in trace.truth["planted"]:
        span = by_id.get(item["span_id"])
        if span is None:
            continue
        text = _surface_text(span, item["surface"])
        if identity["email"] in text or identity["customer_id"] in text:
            out.append(item)
    return out


def _contains_identity(span: Span, identity: dict) -> bool:
    """Is ANY strong identifier for this customer still readable in the span.

    Used by the whole-span check in the tests, not by the per-surface
    measurement. Deliberately checks the email and the customer id rather than
    the name: names are not reliably identifying, and counting them would
    inflate every leak number in this repository, which is the wrong direction
    for a result to err in.
    """
    parts = [str(v) for v in span.attributes.values() if isinstance(v, str)]
    for ev in span.events:
        parts.extend(str(v) for v in ev.attributes.values()
                     if isinstance(v, str))
    parts.append(span.name)
    hay = "\n".join(parts)
    return identity["email"] in hay or identity["customer_id"] in hay
