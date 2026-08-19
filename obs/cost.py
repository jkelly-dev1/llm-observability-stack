"""Per-step cost attribution from a trace, and the three ways it drifts.

What this is for. Someone asks what a feature costs. The trace is the only
per-request record, so cost gets computed from spans: sum the usage attributes,
multiply by list prices. The invoice then disagrees, and the gap is not noise;
it is three specific, nameable effects, each of which is a MISSING ATTRIBUTE
rather than a bug in the arithmetic:

  RETRIES        a retried call emits two spans. Only one of them was billed
                 for output, but both carry input tokens, and both were billed.
                 An attribution that dedupes by call id under-bills; one that
                 sums every span over-bills unless the failed attempt really
                 did consume input, which it did.
  PROMPT CACHING the provider reports cached input in a SEPARATE counter and
                 excludes it from input_tokens. The GENAI conventions have no
                 attribute for that counter, so those tokens are absent from
                 the trace entirely, not mispriced but absent. Attribution from
                 spans alone UNDER-bills every cached request. Measured on a
                 real run: input_tokens 143, cache_read_input_tokens 2,579.
  REASONING      on providers that bill thinking tokens as output, the output
                 count includes tokens no one ever sees. Nothing marks them.

The honest framing, stated before the numbers. Offline, this module is graded
against a truth this repository generated, so it can only demonstrate that the
attribution handles the three effects, not that they occur at any particular
rate in the wild. Scripts/real_run.py is what tests it against a provider's own
usage report, which is the only ground truth that is not ours.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spans import OP_CHAT, USAGE_INPUT, USAGE_OUTPUT, Trace

# USD per million tokens. Dated list prices, and a dated claim: re-check
# before quoting. A price that was right when it was written stops being true
# with nothing in this file changing, and it fails silently: every number
# downstream stays plausible. PRICES_VERIFIED below is the date these were
# read off the providers' own pricing pages; if it is stale, so is any figure
# this module prints.
PRICES = {
    "claude-sonnet-5": {"in": 2.00, "out": 10.00, "cached_in": 0.20},
    "claude-opus-5": {"in": 5.00, "out": 25.00, "cached_in": 0.50},
    "gpt-5.4": {"in": 2.50, "out": 15.00, "cached_in": 0.25},
}
PRICES_VERIFIED = "2026-08-11"


@dataclass
class Attribution:
    """What a trace says a run cost, and how it got there."""

    usd: float
    input_tokens: int
    output_tokens: int
    spans_counted: int
    method: str

    def as_dict(self) -> dict:
        return {"usd": round(self.usd, 6), "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "spans_counted": self.spans_counted, "method": self.method}


def _price(model: str) -> dict:
    if model not in PRICES:
        raise KeyError(f"no price for {model!r}; add it rather than guessing")
    return PRICES[model]


def from_spans(trace: Trace, model: str = "claude-sonnet-5",
               *, dedupe_retries: bool = False) -> Attribution:
    """Attribute cost using ONLY what the trace records.

    This is the corrected form of the naive method: it uses every attribute the
    GenAI conventions actually define. It cannot see prompt caching, because no
    such attribute exists.
    """
    p = _price(model)
    chats: list = []
    tin = tout = 0
    counted = 0
    for span in trace.by_kind(OP_CHAT):
        chats.append(span)
    if dedupe_retries:
        # Keyed on the span name plus parent, which is all a collector has to
        # recognize "the same logical call, tried twice". Keeps the last
        # attempt, which is the charitable implementation: keeping the FIRST
        # would keep the failed attempt and throw away the successful one's
        # output tokens entirely, turning a 4% error into a much larger one for
        # a reason that has nothing to do with the finding.
        last_by_key = {}
        for span in chats:
            last_by_key[(span.name, span.parent_id)] = span
        chats = list(last_by_key.values())
    for span in chats:
        tin += int(span.attributes.get(USAGE_INPUT, 0))
        tout += int(span.attributes.get(USAGE_OUTPUT, 0))
        counted += 1
    usd = tin / 1e6 * p["in"] + tout / 1e6 * p["out"]
    return Attribution(usd, tin, tout, counted,
                       "spans_deduped" if dedupe_retries else "spans_summed")


def from_truth(trace: Trace, model: str = "claude-sonnet-5") -> Attribution:
    """What the run actually cost, from the generator's ground truth.

    `input_tokens` is the uncached input, exactly as the provider reports it,
    so the cached tokens are ADDED here rather than discounted out of it.
    """
    p = _price(model)
    t = trace.truth["true_usage"]
    cached = t["cached_input_tokens"]
    usd = (t["input_tokens"] / 1e6 * p["in"] + cached / 1e6 * p["cached_in"]
           + t["output_tokens"] / 1e6 * p["out"])
    return Attribution(usd, t["input_tokens"] + cached, t["output_tokens"], 0,
                       "truth")


def drift(traces, model: str = "claude-sonnet-5",
          *, dedupe_retries: bool = False) -> dict:
    """Traced total against true total across a corpus.

    Reports the SIGNED error, because the direction is the useful part: a
    method that over-bills makes a feature look unaffordable and one that
    under-bills gets discovered by finance instead of by engineering.
    """
    traced = sum(from_spans(t, model, dedupe_retries=dedupe_retries).usd
                 for t in traces)
    true = sum(from_truth(t, model).usd for t in traces)
    return {
        "method": "spans_deduped" if dedupe_retries else "spans_summed",
        "traced_usd": round(traced, 4),
        "true_usd": round(true, 4),
        "error_usd": round(traced - true, 4),
        "error_pct": round((traced - true) / true * 100, 2) if true else 0.0,
        "n_traces": len(traces),
    }


def attribution_gap(traces, model: str = "claude-sonnet-5") -> dict:
    """Split the total drift into the effects that cause it.

    Each entry is what the traced total would lose if that one effect were
    corrected in isolation, so the parts do not have to sum to the whole and
    are not reported as if they did.
    """
    p = _price(model)
    cached_tokens = sum(t.truth["true_usage"]["cached_input_tokens"]
                        for t in traces)
    retry_input = 0
    for t in traces:
        u = t.truth["true_usage"]
        if u["model_attempts"] > 1:
            # The failed attempt's input tokens: billed, and present in the
            # trace, so summing spans is right and deduping is wrong.
            retry_input += sum(
                int(s.attributes.get(USAGE_INPUT, 0))
                for s in t.by_kind(OP_CHAT) if s.is_error())
    return {
        "cached_input_tokens": cached_tokens,
        "cached_underbill_usd": round(cached_tokens / 1e6 * p["cached_in"], 4),
        "retry_input_tokens": retry_input,
        "retry_underbill_usd_if_deduped": round(
            retry_input / 1e6 * p["in"], 4),
        "note": "Cached input has no GenAI attribute and is excluded from "
                "gen_ai.usage.input_tokens, so those tokens are missing from "
                "the trace rather than mispriced. Both effects push the same "
                "way: a trace-derived total is too LOW.",
    }
