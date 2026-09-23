"""A span model shaped like the OpenTelemetry GenAI semantic conventions.

Why a model and not the SDK. The measurement here is about where content ends
up in a trace, and that is a property of the conventions, not of any exporter.
Modeling it in the standard library keeps the measurement exact, free, and
runnable without an OTLP collector, and a reader can check every name below
against the spec rather than against this code. Scripts/real_run.py emits the
same shape from real API responses.

The part that matters for this repository. Message content has been captured in
TWO different places, and WHICH ONE a library emits depends on the revision it
was written against:

    SUPERSEDED     span EVENTS: gen_ai.system.message, gen_ai.user.message,
    revision       gen_ai.assistant.message, gen_ai.choice, each with a
                   `content` field
    CURRENT        span ATTRIBUTES: gen_ai.input.messages,
    revision       gen_ai.output.messages

The event names are no longer in the conventions, and this model emits them
anyway, because that is the finding: semantic-conventions v1.37.0 (released
2025-08-25) removed all four, directing instrumentations to the two attributes
instead. Removing a name from a specification does not remove it from the
libraries already deployed, the collectors already parsing it, or the backends
already storing it, so a trace arriving today can carry either shape and a
redactor pointed at one does not see the other.

The two groups below are therefore labeled separately and tested separately:
tests/test_conventions.py asserts the current names against the current
registry and asserts the superseded names are spelled the way the revision
that had them spelled them. Renaming the events to the current shape would
delete the finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

# -- attribute names, straight from the current registry ---------------------
# The provider identity. semantic-conventions v1.37.0 renamed `gen_ai.system`
# to this, so `gen_ai.system` is a deprecated spelling, listed as one in
# tests/test_conventions.py and not emitted here.
PROVIDER_NAME = "gen_ai.provider.name"
OPERATION = "gen_ai.operation.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
FINISH_REASONS = "gen_ai.response.finish_reasons"
USAGE_INPUT = "gen_ai.usage.input_tokens"
USAGE_OUTPUT = "gen_ai.usage.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
# The tool leg's content. These two are current registry names, and they are
# constants because a name that appears only as a literal is one
# tests/test_conventions.py cannot pin.
TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_CALL_RESULT = "gen_ai.tool.call.result"

# Content capture, current revision: attributes.
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"

# Content capture, superseded revision: events. Removed from the conventions in
# v1.37.0 and still emitted by deployed instrumentations, so they are modeled
# here. See the module docstring.
EVENT_USER = "gen_ai.user.message"
EVENT_SYSTEM = "gen_ai.system.message"
EVENT_ASSISTANT = "gen_ai.assistant.message"
EVENT_CHOICE = "gen_ai.choice"
# Not a GenAI name: the general-purpose exception event.
EVENT_EXCEPTION = "exception"

# Operations, from the convention's enum.
OP_CHAT = "chat"
OP_EMBEDDINGS = "embeddings"
OP_EXECUTE_TOOL = "execute_tool"

# Attributes that are NOT in the GenAI conventions but end up in the same trace
# because the tool leg is an ordinary HTTP or database client. They are in this
# model because leaving them out is how a redaction measurement flatters
# itself: these carry customer data as often as the prompt does.
HTTP_URL = "url.full"
DB_STATEMENT = "db.query.text"
EXCEPTION_MESSAGE = "exception.message"
EXCEPTION_STACKTRACE = "exception.stacktrace"


@dataclass
class Event:
    """A span event. `attributes` carries the convention's `content` field."""

    name: str
    attributes: dict = field(default_factory=dict)


@dataclass
class Span:
    """One span. Deliberately flat: the exporter shape, not the SDK's."""

    name: str
    kind: str                       # the gen_ai.operation.name, or "internal"
    trace_id: str
    span_id: str
    parent_id: str | None = None
    attributes: dict = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    status: str = "OK"              # "OK" or "ERROR"
    duration_ms: int = 0
    # Set by the run generator, never by an exporter. The ground truth for
    # "this span is an attempt that was retried", which the cost attribution
    # has to get right and a naive per-span sum does not.
    attempt: int = 1

    def is_error(self) -> bool:
        return self.status == "ERROR"


@dataclass
class Trace:
    """Every span emitted by one agent run, plus what really happened.

    `truth` is not telemetry. It is what the generator knows and the trace does
    not: the true per-call token usage, where each planted identifier was put,
    and whether the run actually failed. Keeping it on the trace object rather
    than in a parallel structure is deliberate. The measurements compare the
    trace against the truth constantly, and a truth that can drift out of sync
    with the trace it describes is a defect waiting to be published.
    """

    trace_id: str
    spans: list[Span] = field(default_factory=list)
    truth: dict = field(default_factory=dict)

    def add(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    def by_kind(self, kind: str) -> Iterator[Span]:
        return (s for s in self.spans if s.kind == kind)

    def failed(self) -> bool:
        """Did the RUN fail, as opposed to any span inside it.

        A retried call leaves an ERROR span behind and the run still succeeds.
        Sampling policies that key on 'any error span' therefore keep far more
        than the failures, which is measured in obs/sampling.py rather than
        asserted here.
        """
        return bool(self.truth.get("run_failed"))
