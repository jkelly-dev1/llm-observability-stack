"""Synthetic agent runs with known ground truth, and known planted identifiers.

Why synthetic. Every measurement in this repository is "what did the trace fail
to record, or record twice?", and answering that needs a run whose true token
usage and true identifier placement are known by construction. A trace captured
from a real agent gives you neither: you would be grading the telemetry against
the telemetry.

What a run looks like. A support agent answering one customer question:

    chat (plan)  ->  execute_tool (lookup_customer)  ->  chat (answer)
                     execute_tool (search_orders)

with retries, tool failures and a rare hard failure mixed in at measured rates.
Each run plants ONE customer identity in the places a real deployment would put
it, and records exactly where. That placement list is the ground truth the
redaction measurement grades against: see PLANTED_SURFACES.

The planting is the experiment, so it is not arbitrary. Each surface below is
in the list because a real agent stack puts customer data there, and each is
named so a reader can disagree with a specific one rather than with the total.
"""

from __future__ import annotations

import json
import random

from .spans import (DB_STATEMENT, EVENT_ASSISTANT, EVENT_CHOICE,
                    EVENT_EXCEPTION, EVENT_USER, EXCEPTION_MESSAGE,
                    EXCEPTION_STACKTRACE, FINISH_REASONS, HTTP_URL,
                    INPUT_MESSAGES, OP_CHAT, OP_EXECUTE_TOOL, OPERATION,
                    OUTPUT_MESSAGES, REQUEST_MODEL, RESPONSE_ID,
                    RESPONSE_MODEL, SYSTEM, TOOL_CALL_ID, TOOL_NAME,
                    USAGE_INPUT, USAGE_OUTPUT, Event, Span, Trace)

# The eleven places one customer's identity lands in one trace. The redaction
# measurement reports coverage per surface, so this list IS THE DENOMINATOR,
# which is exactly why a surface the corpus emits and this tuple omits makes
# the measurement flatter itself. obs/redact.py says so in its own words.
PLANTED_SURFACES = (
    "input_messages_attr",    # newer convention: content in a span attribute
    "user_message_event",     # older convention: content in a span event
    "output_messages_attr",   # the model's answer, quoting the customer back
    "choice_event",           # the same answer under the older convention
    "tool_call_arguments",    # what the agent passed to lookup_customer
    "tool_result",            # what came back, which is the whole record
    "http_url",               # the tool's own outbound call, id in the path
    "db_statement",           # the query the tool ran, id inline
    "exception_message",      # the failure text, which quotes the input
    "exception_stacktrace",   # the frame arguments, which quote it again
    "assistant_message_event",  # the answer a THIRD time, older convention
)

_FIRST = ("Dana", "Priya", "Marcus", "Elena", "Tomas", "Aisha", "Ruth", "Ivan")
_LAST = ("Whitfield", "Nakamura", "Osei", "Lindqvist", "Barros", "Kaur",
         "Petrov", "Mensah")
_DOMAINS = ("northwind.example", "acmesupply.example", "lakeside.example")
_QUESTIONS = (
    "where is my order",
    "why was I charged twice",
    "can I change the delivery address",
    "is this part still under warranty",
)

# Prices are irrelevant to the offline measurement, cost.py takes them as an
# argument, but the TOKEN COUNTS below are load-bearing, so they are ranges a
# real support turn actually produces rather than round numbers.
_IN_RANGE = (700, 2400)
_OUT_RANGE = (80, 600)

# Rates, chosen once and stated because every sampling number depends on them.
P_TOOL_RETRY = 0.18      # a tool call that fails once and succeeds on retry
P_MODEL_RETRY = 0.09     # a 429/overload on the model leg, then succeeds
P_RUN_FAILS = 0.04       # the run itself fails: this is what tracing is for
P_CACHED = 0.35          # the request hit the provider's prompt cache

# The shared prefix, in tokens. Measured from scripts/real_run.py's own system
# prompt, which the paid run reported as 2,579 cached tokens against 143
# uncached ones.
SYSTEM_PREFIX_TOKENS = 2579


def _identity(rng: random.Random) -> dict:
    first, last = rng.choice(_FIRST), rng.choice(_LAST)
    return {
        "name": f"{first} {last}",
        "email": f"{first.lower()}.{last.lower()}@{rng.choice(_DOMAINS)}",
        "customer_id": f"CUS-{rng.randint(100000, 999999)}",
        "phone": f"+1-555-{rng.randint(1000, 9999)}",
    }


def make_run(index: int) -> Trace:
    """Build run `index`. Deterministic: same index, same trace and truth."""
    rng = random.Random(f"obs-run-{index}")
    who = _identity(rng)
    question = rng.choice(_QUESTIONS)
    trace_id = f"T{index:05d}"
    tr = Trace(trace_id=trace_id)

    planted: list[dict] = []

    def plant(surface: str, span_id: str, value: str) -> None:
        planted.append({"surface": surface, "span_id": span_id,
                        "value": value})

    prompt = (f"Customer {who['name']} ({who['email']}, {who['customer_id']}) "
              f"asks: {question}")
    answer = (f"Hello {who['name']}, your order for account "
              f"{who['customer_id']} shipped yesterday.")

    # -- the planning call -----------------------------------------------
    plan_in = rng.randint(*_IN_RANGE)
    plan_out = rng.randint(*_OUT_RANGE)
    cached = rng.random() < P_CACHED
    # How a provider actually reports a cache hit, taken from the paid run
    # rather than assumed. Anthropic reports `input_tokens` EXCLUDING the
    # cached prefix and puts the cached tokens in a separate counter. The
    # measured run: input_tokens 143, cache_read_input_tokens 2,579: the
    # conventional attribute saw 5% of the tokens the request actually
    # processed.
    #
    # Putting the whole input on the Span and modeling the cache as a discount
    # is the intuitive model and it points the wrong way: it makes a trace
    # OVER-bill. The real semantics make it UNDER-bill, and by far more. See
    # audit/real_run.json. The shared system prefix an agent sends on every
    # turn: tools, policy, format rules. This is what gets cached in
    # production, and it is a constant rather than a fraction of the turn, so
    # a cached turn loses most of its tokens from the trace. Sized from the
    # paid run's own prefix (2,579 tokens).
    cache_read = SYSTEM_PREFIX_TOKENS if cached else 0
    plan_uncached = plan_in
    plan = tr.add(Span(
        name=f"{OP_CHAT} claude-sonnet-5", kind=OP_CHAT, trace_id=trace_id,
        span_id=f"{trace_id}-plan",
        attributes={
            SYSTEM: "anthropic", OPERATION: OP_CHAT,
            REQUEST_MODEL: "claude-sonnet-5",
            RESPONSE_MODEL: "claude-sonnet-5",
            RESPONSE_ID: f"msg_{index:05d}",
            FINISH_REASONS: ["tool_use"],
            # Only the uncached input reaches the convention attribute.
            USAGE_INPUT: plan_uncached, USAGE_OUTPUT: plan_out,
            # Newer convention: the whole conversation, in an attribute.
            INPUT_MESSAGES: json.dumps(
                [{"role": "user", "parts": [{"type": "text",
                                             "content": prompt}]}]),
        },
        events=[Event(EVENT_USER, {"content": prompt})],
        duration_ms=rng.randint(400, 2600)))
    plant("input_messages_attr", plan.span_id, prompt)
    plant("user_message_event", plan.span_id, prompt)

    # -- the tool leg, with a retry some of the time ------------------------
    tool_args = json.dumps({"customer_id": who["customer_id"],
                            "email": who["email"]})
    tool_result = json.dumps({"customer": who, "orders": 3})
    attempts = 2 if rng.random() < P_TOOL_RETRY else 1
    for attempt in range(1, attempts + 1):
        failed = attempt < attempts
        sp = tr.add(Span(
            name=f"{OP_EXECUTE_TOOL} lookup_customer", kind=OP_EXECUTE_TOOL,
            trace_id=trace_id, span_id=f"{trace_id}-tool{attempt}",
            parent_id=plan.span_id,
            attributes={
                OPERATION: OP_EXECUTE_TOOL, TOOL_NAME: "lookup_customer",
                TOOL_CALL_ID: f"call_{index:05d}",
                "gen_ai.tool.call.arguments": tool_args,
                HTTP_URL: ("https://crm.internal.example/v2/customers/"
                           f"{who['customer_id']}?email={who['email']}"),
                DB_STATEMENT: ("SELECT * FROM customers WHERE email = "
                               f"'{who['email']}'"),
            },
            status="ERROR" if failed else "OK",
            duration_ms=rng.randint(30, 900), attempt=attempt))
        plant("tool_call_arguments", sp.span_id, tool_args)
        plant("http_url", sp.span_id, who["email"])
        plant("db_statement", sp.span_id, who["email"])
        if failed:
            msg = f"upstream 503 while fetching {who['email']}"
            sp.events.append(Event(EVENT_EXCEPTION, {
                EXCEPTION_MESSAGE: msg,
                EXCEPTION_STACKTRACE: (
                    "Traceback (most recent call last):\n"
                    "  File \"crm_client.py\", line 88, in fetch\n"
                    f"    raise UpstreamError({who['email']!r})"),
            }))
            plant("exception_message", sp.span_id, msg)
            plant("exception_stacktrace", sp.span_id, who["email"])
        else:
            sp.attributes["gen_ai.tool.call.result"] = tool_result
            plant("tool_result", sp.span_id, tool_result)

    # -- the answering call, also retried some of the time -----------------
    ans_in = rng.randint(*_IN_RANGE)
    ans_out = rng.randint(*_OUT_RANGE)
    model_attempts = 2 if rng.random() < P_MODEL_RETRY else 1
    for attempt in range(1, model_attempts + 1):
        failed = attempt < model_attempts
        sp = tr.add(Span(
            name=f"{OP_CHAT} claude-sonnet-5", kind=OP_CHAT,
            trace_id=trace_id, span_id=f"{trace_id}-answer{attempt}",
            parent_id=plan.span_id,
            attributes={
                SYSTEM: "anthropic", OPERATION: OP_CHAT,
                REQUEST_MODEL: "claude-sonnet-5",
                RESPONSE_MODEL: "claude-sonnet-5",
                USAGE_INPUT: ans_in,
                USAGE_OUTPUT: 0 if failed else ans_out,
                FINISH_REASONS: ["error"] if failed else ["end_turn"],
            },
            status="ERROR" if failed else "OK",
            duration_ms=rng.randint(600, 5200), attempt=attempt))
        if failed:
            sp.events.append(Event(EVENT_EXCEPTION, {
                EXCEPTION_MESSAGE: "overloaded_error: retry after 1s",
                EXCEPTION_STACKTRACE: "anthropic.OverloadedError",
            }))
        else:
            sp.attributes[OUTPUT_MESSAGES] = json.dumps(
                [{"role": "assistant",
                  "parts": [{"type": "text", "content": answer}]}])
            sp.events.append(Event(EVENT_CHOICE, {"content": answer}))
            sp.events.append(Event(EVENT_ASSISTANT, {"content": answer}))
            plant("output_messages_attr", sp.span_id, answer)
            plant("choice_event", sp.span_id, answer)
            plant("assistant_message_event", sp.span_id, answer)

    run_failed = rng.random() < P_RUN_FAILS
    if run_failed:
        # The failure this whole stack exists to catch: the agent answers
        # without the customer's data at all. No span is ERROR, nothing
        # retries, and the run looks perfect in every latency dashboard.
        #
        # The wrong answer has to replace every copy of the right one, and the
        # planted record has to stop claiming those surfaces carry the
        # identity. Rewriting only the attribute leaves the event holding the
        # correct answer and the ground truth describing a value that is no
        # longer there, which the corpus test catches on 9 of 300 runs.
        wrong = "Hello, your order shipped yesterday."
        last = tr.spans[-1]
        last.attributes[OUTPUT_MESSAGES] = json.dumps(
            [{"role": "assistant",
              "parts": [{"type": "text", "content": wrong}]}])
        for ev in last.events:
            if "content" in ev.attributes:
                ev.attributes["content"] = wrong
        planted = [p for p in planted
                   if not (p["span_id"] == last.span_id
                           and p["surface"] in ("output_messages_attr",
                                                "choice_event",
                                                "assistant_message_event"))]

    tr.truth = {
        "identity": who,
        "planted": planted,
        "run_failed": run_failed,
        "true_usage": {
            # What the spans carry: uncached input only, matching the provider.
            "input_tokens": plan_uncached + ans_in * model_attempts,
            "output_tokens": plan_out + ans_out,
            # Billed, and in no GenAI attribute anywhere.
            "cached_input_tokens": cache_read,
            "model_attempts": model_attempts,
            "tool_attempts": attempts,
        },
        "cached": cached,
    }
    # The provider bills cached input at a discount, and the trace does not say
    # so anywhere: no GenAI attribute carries a cache flag. That omission is
    # the whole cost finding, so it is recorded in the truth and deliberately
    # NOT written onto the span.
    return tr


def corpus(n: int) -> list[Trace]:
    """`n` deterministic runs."""
    return [make_run(i) for i in range(n)]
