"""The attribute names, pinned as literal strings.

Why a test of constants exists here. The README's argument is that a redactor
written against one revision of the GenAI conventions never visits the other,
and the whole measurement is expressed in these names. Nothing else in the
suite reads them: the corpus writes an attribute and the policy reads the same
constant back, so producer and consumer move together and a rename is
invisible to them: renaming `gen_ai.usage.input_tokens` to the deprecated
`gen_ai.usage.prompt_tokens` would move no published number.

These are therefore asserted against the spelling the conventions actually
use, not against themselves. A test that compared the constant to the constant would be
the same tautology this file exists to remove.

The names are in three groups, because semantic-conventions v1.37.0
(2025-08-25) renamed `gen_ai.system` to `gen_ai.provider.name` and removed all
four per-message events:

    CURRENT     in the registry today. A rename upstream should break these.
    SUPERSEDED  removed from the conventions, still emitted by deployed
                instrumentations. This repository models them because they are
                the older half of the two-revision split the redaction finding
                rests on, and they are pinned so that renaming them to the
                current shape cannot delete the finding.
    DEPRECATED  must appear nowhere. `gen_ai.system` is in this group.

A name that exists only as a bare literal cannot be pinned here, so the
tool-call names are constants in obs/spans.py and are in CURRENT below.
"""

from __future__ import annotations

from obs import spans


# CURRENT: the GenAI registry names this model emits, spelled out rather than
# imported, so a rename has to be made in two places by someone who meant it.
GENAI_ATTRIBUTES = {
    "PROVIDER_NAME": "gen_ai.provider.name",
    "OPERATION": "gen_ai.operation.name",
    "REQUEST_MODEL": "gen_ai.request.model",
    "RESPONSE_MODEL": "gen_ai.response.model",
    "RESPONSE_ID": "gen_ai.response.id",
    "FINISH_REASONS": "gen_ai.response.finish_reasons",
    "USAGE_INPUT": "gen_ai.usage.input_tokens",
    "USAGE_OUTPUT": "gen_ai.usage.output_tokens",
    "TOOL_NAME": "gen_ai.tool.name",
    "TOOL_CALL_ID": "gen_ai.tool.call.id",
    "TOOL_CALL_ARGUMENTS": "gen_ai.tool.call.arguments",
    "TOOL_CALL_RESULT": "gen_ai.tool.call.result",
    "INPUT_MESSAGES": "gen_ai.input.messages",
    "OUTPUT_MESSAGES": "gen_ai.output.messages",
}

# SUPERSEDED: the per-message events. Present in the conventions through
# semantic-conventions v1.36.0 (released 2025-07-05) and REMOVED in v1.37.0
# (released 2025-08-25), whose release notes say to use `gen_ai.input.messages`
# and `gen_ai.output.messages` instead. Deployed instrumentations still emit
# them, so obs/spans.py models them and the spelling is pinned here: the redaction finding is that a redactor pointed at the current
# attributes never visits these.
SUPERSEDED_EVENTS = {
    "EVENT_USER": "gen_ai.user.message",
    "EVENT_SYSTEM": "gen_ai.system.message",
    "EVENT_ASSISTANT": "gen_ai.assistant.message",
    "EVENT_CHOICE": "gen_ai.choice",
}

GENAI_OPERATIONS = {
    "OP_CHAT": "chat",
    "OP_EMBEDDINGS": "embeddings",
    "OP_EXECUTE_TOOL": "execute_tool",
}

# NOT GenAI names, and this asserts that they are not. They belong to the HTTP,
# database and exception conventions, and they are in this model because the
# tool leg is an ordinary client that carries customer data as often as the
# prompt does. Only the broadest policy reaches them, which is the finding.
NON_GENAI_ATTRIBUTES = {
    "HTTP_URL": "url.full",
    "DB_STATEMENT": "db.query.text",
    "EXCEPTION_MESSAGE": "exception.message",
    "EXCEPTION_STACKTRACE": "exception.stacktrace",
    "EVENT_EXCEPTION": "exception",
}


def test_every_genai_attribute_is_spelled_the_way_the_conventions_spell_it():
    for name, want in GENAI_ATTRIBUTES.items():
        assert getattr(spans, name) == want, (
            f"spans.{name} is {getattr(spans, name)!r}, conventions say {want!r}")


def test_every_superseded_event_name_matches_the_revision_that_had_it():
    """These four are not current, and asserting them is still right.

    A reader who checks them against today's registry will not find them: they
    left in v1.37.0. They are pinned against the spelling the superseded
    revision used, because the measurement's claim is that a trace arriving
    today can still carry this shape, and renaming these four constants to the
    current shape would make the prompt-only policy look complete.
    """
    for name, want in SUPERSEDED_EVENTS.items():
        assert getattr(spans, name) == want, (
            f"spans.{name} is {getattr(spans, name)!r}; the superseded "
            f"revision spells it {want!r}")


def test_the_superseded_events_are_not_mistaken_for_current_attributes():
    """The two groups must stay disjoint, or the finding collapses.

    The redaction result is a DIFFERENCE between what the current-attribute
    policy reaches and what the event-carrying trace holds. If an event name
    ever equalled a current attribute name, the prompt-only policy would score
    as covering a surface it never visits.
    """
    assert not set(SUPERSEDED_EVENTS.values()) & set(GENAI_ATTRIBUTES.values())


def test_operation_names_come_from_the_conventions_enum():
    for name, want in GENAI_OPERATIONS.items():
        assert getattr(spans, name) == want


def test_the_non_genai_attributes_are_named_and_are_not_genai():
    """The other half, and the one the README's argument rests on: these four
    are not GenAI names. If one silently became a `gen_ai.*` name the 'only
    the broadest policy reaches them' finding would stop meaning what it
    says."""
    for name, want in NON_GENAI_ATTRIBUTES.items():
        got = getattr(spans, name)
        assert got == want, f"spans.{name} is {got!r}, expected {want!r}"
        assert not got.startswith("gen_ai."), (
            f"spans.{name} is a GenAI name; the redaction argument depends on "
            "it not being one")


def test_no_genai_name_is_a_deprecated_spelling():
    """The renames that would look harmless and are not."""
    deprecated = {
        "gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens",
        "gen_ai.prompt", "gen_ai.completion",
        "http.url", "db.statement",
        # Renamed to gen_ai.provider.name in semantic-conventions v1.37.0
        # (2025-08-25).
        "gen_ai.system",
    }
    live = {v for v in GENAI_ATTRIBUTES.values()} | {v for v in NON_GENAI_ATTRIBUTES.values()}
    assert not (live & deprecated), f"deprecated names in use: {live & deprecated}"
    # Absent from the module as well as from the table above, so a
    # constant that is emitted but never tabulated cannot reintroduce one.
    emitted = {v for k, v in vars(spans).items()
               if k.isupper() and isinstance(v, str)}
    assert not (emitted & deprecated), (
        f"obs/spans.py still defines deprecated names: {emitted & deprecated}")


def test_every_genai_name_the_model_emits_is_in_one_of_the_three_groups():
    """No `gen_ai.*` constant may be untested.

    A name used only as a bare literal elsewhere could be renamed everywhere
    with this file green, so the check is structural: anything `gen_ai.*`
    that obs/spans.py defines must appear in CURRENT or in SUPERSEDED, and a
    new constant that is in neither fails here.
    """
    declared = set(GENAI_ATTRIBUTES.values()) | set(SUPERSEDED_EVENTS.values())
    emitted = {v for k, v in vars(spans).items()
               if k.isupper() and isinstance(v, str) and v.startswith("gen_ai.")}
    assert emitted <= declared, (
        f"gen_ai.* constants in obs/spans.py that no group pins: "
        f"{sorted(emitted - declared)}")
