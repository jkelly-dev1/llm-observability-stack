"""A span model shaped like the OpenTelemetry GenAI semantic conventions.

Why a model and not the SDK. The measurement here is about where content ends
up in a trace, and that is a property of the conventions, not of any exporter.
Modeling it in the standard library keeps the measurement exact, free, and
runnable without an OTLP collector, and every attribute name below is the one
the convention specifies, so a reader can check the claim against the spec
rather than against this code. Scripts/real_run.py emits the same shape from
real API responses.

The part that matters for this repository. The conventions capture message
content in TWO different places depending on which revision a library targets:

    older revision   span EVENTS: gen_ai.user.message, gen_ai.assistant.message,
                     gen_ai.choice, each with a `content` field
    newer revision   span ATTRIBUTES: gen_ai.input.messages,
                     gen_ai.output.messages

Both are opt-in, both are widely deployed, and a process that walks one of them
does not see the other. That is not a subtlety of this model; it is the fact
the redaction measurement rests on, and it is why a redactor can be correct,
tested, and still leak everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

# -- attribute names, straight from the convention ---------------------------
SYSTEM = "gen_ai.system"
OPERATION = "gen_ai.operation.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
FINISH_REASONS = "gen_ai.response.finish_reasons"
USAGE_INPUT = "gen_ai.usage.input_tokens"
USAGE_OUTPUT = "gen_ai.usage.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"

# Content capture, newer revision: attributes.
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"

# Content capture, older revision: events.
EVENT_USER = "gen_ai.user.message"
EVENT_SYSTEM = "gen_ai.system.message"
EVENT_ASSISTANT = "gen_ai.assistant.message"
EVENT_CHOICE = "gen_ai.choice"
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
