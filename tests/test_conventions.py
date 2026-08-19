"""The attribute names, pinned as literal strings.

Why a test of constants exists here. The README's argument is that a redactor
written against one revision of the GenAI conventions never visits the other,
and the whole measurement is expressed in these names. Nothing else in the
suite reads them: the corpus writes an attribute and the policy reads the same
constant back, so producer and consumer move together and a rename is
invisible. Renaming `gen_ai.usage.input_tokens` to the deprecated
`gen_ai.usage.prompt_tokens` left the whole suite green and every published
number identical.

So these are asserted against the spelling the conventions actually use, not
against themselves. A test that compared the constant to the constant would be
the same tautology this file exists to remove.
"""

from __future__ import annotations

from obs import spans


# The GenAI registry names this model emits, spelled out rather than imported,
# so a rename has to be made in two places by someone who meant it.
GENAI_ATTRIBUTES = {
    "SYSTEM": "gen_ai.system",
    "OPERATION": "gen_ai.operation.name",
    "REQUEST_MODEL": "gen_ai.request.model",
    "RESPONSE_MODEL": "gen_ai.response.model",
    "RESPONSE_ID": "gen_ai.response.id",
    "FINISH_REASONS": "gen_ai.response.finish_reasons",
    "USAGE_INPUT": "gen_ai.usage.input_tokens",
    "USAGE_OUTPUT": "gen_ai.usage.output_tokens",
    "TOOL_NAME": "gen_ai.tool.name",
    "TOOL_CALL_ID": "gen_ai.tool.call.id",
    "INPUT_MESSAGES": "gen_ai.input.messages",
    "OUTPUT_MESSAGES": "gen_ai.output.messages",
}

GENAI_EVENTS = {
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


def test_every_genai_event_name_matches_the_older_revision():
    for name, want in GENAI_EVENTS.items():
        assert getattr(spans, name) == want, (
            f"spans.{name} is {getattr(spans, name)!r}, conventions say {want!r}")


def test_operation_names_come_from_the_conventions_enum():
    for name, want in GENAI_OPERATIONS.items():
        assert getattr(spans, name) == want


def test_the_non_genai_attributes_are_named_and_are_not_genai():
    """The other half, and the one the README's argument rests on: these four
    are deliberately NOT GenAI names. If one silently became a `gen_ai.*` name
    the 'only the broadest policy reaches them' finding would stop meaning
    what it says."""
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
    }
    live = {v for v in GENAI_ATTRIBUTES.values()} | {v for v in NON_GENAI_ATTRIBUTES.values()}
    assert not (live & deprecated), f"deprecated names in use: {live & deprecated}"
