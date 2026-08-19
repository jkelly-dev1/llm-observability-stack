"""Turn a sampled trace into an eval case, and report what is missing.

The promise this path makes. A production run went wrong, you have its trace,
so you turn it into a regression case and it never goes wrong again. That is
what ties an observability stack to an eval harness, and the reason G was built
next to llm-eval-gate.

The part nobody mentions. A trace is telemetry, not a fixture. To replay a run
you need the inputs; to GRADE a replay you need the expected output, and the
trace records what the model SAID, not what it should have said. Promoting a
trace gives you a case whose expectation is the very output you are trying to
stop reproducing.

So this module does two things and refuses to do a third: it extracts what the
trace really contains, it states exactly which fields are missing, and it will
not invent an expectation. `promote()` returns a case with
`expected_output=None` and a `blocked_on` list, and the measurement reports how
many promoted cases are runnable, which, for the failure this repository cares
about, is none of them without a human.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .spans import (INPUT_MESSAGES, OP_CHAT, OP_EXECUTE_TOOL, OUTPUT_MESSAGES,
                    TOOL_NAME, Trace)


@dataclass
class EvalCase:
    """A regression case built from one trace."""

    case_id: str
    input_messages: list = field(default_factory=list)
    tools_called: list = field(default_factory=list)
    observed_output: str | None = None
    expected_output: str | None = None
    # Everything the trace could not supply. A case with a non-empty
    # blocked_on is not a test, it is a to-do with provenance.
    blocked_on: list = field(default_factory=list)

    def runnable(self) -> bool:
        return not self.blocked_on

    def as_dict(self) -> dict:
        return {"case_id": self.case_id, "input_messages": self.input_messages,
                "tools_called": self.tools_called,
                "observed_output": self.observed_output,
                "expected_output": self.expected_output,
                "blocked_on": list(self.blocked_on)}


def promote(trace: Trace, *, redacted: bool = False) -> EvalCase:
    """Build the case a trace can support, and name what it cannot.

    `redacted` is not a detail. If the trace went through the redaction the
    previous section recommends, the customer identifier in the input is now
    [REDACTED] and the case cannot be replayed as it happened. That is a real
    tension between two things this repository recommends, and nothing here
    resolves it.
    """
    case = EvalCase(case_id=f"case-{trace.trace_id}")

    for span in trace.by_kind(OP_CHAT):
        raw = span.attributes.get(INPUT_MESSAGES)
        if raw and not case.input_messages:
            try:
                case.input_messages = json.loads(raw)
            except json.JSONDecodeError:
                case.blocked_on.append("input messages were not parseable")
        out = span.attributes.get(OUTPUT_MESSAGES)
        if out:
            try:
                parts = json.loads(out)[0]["parts"]
                case.observed_output = parts[0]["content"]
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    case.tools_called = [s.attributes.get(TOOL_NAME)
                         for s in trace.by_kind(OP_EXECUTE_TOOL)
                         if not s.is_error()]

    if not case.input_messages:
        # The older convention puts content in events, so a promoter written
        # against the newer one finds nothing at all here.
        case.blocked_on.append(
            "no gen_ai.input.messages attribute: content capture was off, or "
            "this run used the event-based convention")
    if case.observed_output is None:
        case.blocked_on.append("no output content captured")

    # The one that never goes away.
    case.blocked_on.append(
        "no expected output: the trace records what the model said, and this "
        "case exists because that was wrong")

    if redacted and _looks_redacted(case.input_messages):
        case.blocked_on.append(
            "input is redacted: the identifiers the run turned on are masked, "
            "so the replay is not the run")

    # Tool results are not in the case at all: replaying needs the tool to
    # return what it returned that day, and nothing in a trace pins that.
    if case.tools_called:
        case.blocked_on.append(
            "tool responses are not reproducible from the trace: the case "
            "needs a recorded fixture per tool call")
    return case


def _looks_redacted(messages) -> bool:
    return "[REDACTED]" in json.dumps(messages)


def promote_all(traces, *, redacted: bool = False) -> dict:
    """Promote a corpus and report how much of it is actually runnable."""
    cases = [promote(t, redacted=redacted) for t in traces]
    blockers: dict = {}
    for c in cases:
        for b in c.blocked_on:
            key = b.split(":")[0]
            blockers[key] = blockers.get(key, 0) + 1
    return {
        "cases": len(cases),
        "runnable": sum(1 for c in cases if c.runnable()),
        "blockers": dict(sorted(blockers.items(), key=lambda kv: -kv[1])),
    }
