#!/usr/bin/env python3
"""The paid half: trace a real agent turn and check the trace against the bill.

    ENV_FILE=~/.secrets/ai.env python scripts/real_run.py
    ENV_FILE=~/.secrets/ai.env python scripts/real_run.py --confirm

WITHOUT --confirm THIS SPENDS NOTHING. The default prints the plan and the cost
estimate and exits.

Why this run exists, given the offline half already measures cost drift. The
offline number is graded against a ground truth this repository generated, so
it can only show that the attribution handles caching and retries, not that a
real provider's numbers behave that way. Here the ground truth is the
provider's own usage report, which nobody here wrote.

Two things are measured, and the second is the one that could embarrass the
offline result:

  1 COST      cost computed from the GenAI convention attributes alone, against
              cost computed from the provider's full usage object. The
              conventions define gen_ai.usage.input_tokens and
              .output_tokens and nothing for cache reads, so a request served
              from cache is billed at the full rate by any attribution built on
              the conventions. This run makes that happen on purpose, with a
              shared prefix long enough to be cacheable.

  2 SURFACES  whether a real model actually reproduces the customer identifier
              into its answer. The redaction measurement assumes the output is
              a leak surface; if a real model never quotes the identifier back,
              that assumption is generous and the offline leak counts are
              inflated. This checks it rather than assuming it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from obs.cost import PRICES, PRICES_VERIFIED                    # noqa: E402
from obs.redact import PATTERNS                                 # noqa: E402
from obs.runs import make_run                                   # noqa: E402
from obs.spans import (FINISH_REASONS, INPUT_MESSAGES, OP_CHAT,  # noqa: E402
                       OPERATION, OUTPUT_MESSAGES, REQUEST_MODEL,
                       RESPONSE_MODEL, SYSTEM, USAGE_INPUT, USAGE_OUTPUT,
                       Span, Trace)

# A long, stable system prompt. Its ONLY job is to exceed the provider's
# minimum cacheable prefix so that runs after the first are served from cache
# and the discount shows up in the bill but not in the trace. Padding is honest
# filler rather than hidden instructions. An unstated instruction here would
# change what the model does and confound the surface measurement.
_POLICY_LINE = (
    "Support policy: answer only from the customer record supplied by the "
    "lookup tool, never invent an order status, and keep replies to two "
    "sentences. Escalate billing disputes to a human. Do not offer refunds. ")
SYSTEM_PROMPT = ("You are a support agent for a parts distributor.\n"
                 + _POLICY_LINE * 40)

ANSWER_PROMPT = """\
Using the customer record below, answer the customer's question in two
sentences. Address them by name and include their account identifier.

{record}

Question: {question}"""


def _api_key(name: str = "ANTHROPIC_API_KEY") -> str:
    """Read `name` from the file ENV_FILE points at.

    Never falls back to a key already in the environment: an accidental ambient
    key is how a run gets billed to the wrong account.
    """
    env_file = os.environ.get("ENV_FILE")
    if not env_file:
        raise RuntimeError(
            "ENV_FILE is not set. Run with:\n"
            "    ENV_FILE=~/.secrets/ai.env python scripts/real_run.py")
    path = Path(env_file).expanduser()
    if not path.is_file():
        raise RuntimeError(f"ENV_FILE points at {path}, which does not exist")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    raise RuntimeError(f"no {name} line in {path}")


def _span_from_response(trace_id: str, span_id: str, model: str, resp,
                        prompt: str, text: str) -> tuple[Span, dict]:
    """Build a convention-shaped span, and return the FULL usage separately.

    The split is the experiment. Everything that goes on the span is something
    the GenAI conventions define. The cache counters have no convention
    attribute, so they go in the second return value: exactly where a real
    collector loses them.
    """
    usage = resp.usage
    full = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(
            usage, "cache_read_input_tokens", 0) or 0,
    }
    span = Span(
        name=f"{OP_CHAT} {model}", kind=OP_CHAT, trace_id=trace_id,
        span_id=span_id,
        attributes={
            SYSTEM: "anthropic", OPERATION: OP_CHAT,
            REQUEST_MODEL: model, RESPONSE_MODEL: resp.model,
            FINISH_REASONS: [resp.stop_reason],
            # The conventions carry these two and only these two.
            USAGE_INPUT: full["input_tokens"],
            USAGE_OUTPUT: full["output_tokens"],
            INPUT_MESSAGES: json.dumps(
                [{"role": "user", "parts": [{"type": "text",
                                             "content": prompt}]}]),
            OUTPUT_MESSAGES: json.dumps(
                [{"role": "assistant", "parts": [{"type": "text",
                                                  "content": text}]}]),
        })
    return span, full


def _cost_from_span(span: Span, price: dict) -> float:
    return (int(span.attributes[USAGE_INPUT]) / 1e6 * price["in"]
            + int(span.attributes[USAGE_OUTPUT]) / 1e6 * price["out"])


def _cost_from_usage(full: dict, price: dict) -> float:
    """What the provider actually bills, cache reads at the cached rate.

    Anthropic reports cache reads SEPARATELY from input_tokens, so the billed
    full-rate input is input_tokens as reported, plus cache writes at a
    premium, plus cache reads at the discount.
    """
    return (full["input_tokens"] / 1e6 * price["in"]
            + full["cache_creation_input_tokens"] / 1e6 * price["in"] * 1.25
            + full["cache_read_input_tokens"] / 1e6 * price["cached_in"]
            + full["output_tokens"] / 1e6 * price["out"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--model", default="claude-sonnet-5", choices=sorted(PRICES))
    ap.add_argument("--max-cost", type=float, default=1.00)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("audit/real_run.json"))
    args = ap.parse_args()

    price = PRICES[args.model]
    # Measured from one built prompt plus a typical two-sentence answer, and
    # rounded up so the estimate errs high.
    est_in = len(SYSTEM_PROMPT) // 3 + 400
    est_out = 220
    # One model call per run, which is what the loop below actually makes.
    # Saying two would err high and still be wrong: an estimate nobody can
    # reconcile against the loop is not a safeguard.
    calls = args.runs
    cost = (calls * est_in / 1e6 * price["in"]
            + calls * est_out / 1e6 * price["out"])

    print(f"model            {args.model}")
    print(f"design           {args.runs} agent runs x 1 model call = "
          f"{calls} calls")
    print(f"system prompt    {len(SYSTEM_PROMPT):,} chars, cached after the "
          f"first call")
    print(f"estimated tokens {calls * est_in:,} in / {calls * est_out:,} out")
    print(f"ESTIMATED COST   ${cost:.2f}  (list prices verified "
          f"{PRICES_VERIFIED}, and this estimate treats every call as a cache "
          f"MISS, so it errs high)")

    if cost > args.max_cost:
        print(f"\nREFUSING TO START: ${cost:.2f} exceeds --max-cost "
              f"${args.max_cost:.2f}.")
        return 2
    if not args.confirm:
        print("\nDry run. Nothing was sent and nothing was billed.")
        print("Re-run with --confirm to spend the amount above.")
        return 0

    import anthropic                                   # noqa: PLC0415
    client = anthropic.Anthropic(api_key=_api_key())

    records = []
    traced_usd = billed_usd = 0.0
    cache_reads = 0
    id_in_output = 0
    t0 = time.time()

    for i in range(args.runs):
        synthetic = make_run(i)
        who = synthetic.truth["identity"]
        question = "where is my order"
        record = json.dumps({"customer": who, "orders": [{"id": "SO-4471",
                                                          "status": "shipped"}]})
        trace = Trace(trace_id=f"R{i:05d}")
        prompt = ANSWER_PROMPT.format(record=record, question=question)

        try:
            resp = client.messages.create(
                model=args.model, max_tokens=400,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}])
        except Exception as e:                          # noqa: BLE001
            records.append({"run": i, "error": repr(e)})
            print(f"  run {i:03d}  CALL FAILED: {e}")
            continue

        text = "".join(b.text for b in resp.content if b.type == "text")
        span, full = _span_from_response(trace.trace_id, f"{trace.trace_id}-a",
                                         args.model, resp, prompt, text)
        trace.add(span)
        traced = _cost_from_span(span, price)
        billed = _cost_from_usage(full, price)
        traced_usd += traced
        billed_usd += billed
        cache_reads += full["cache_read_input_tokens"]

        # Surface check: did the model put the identifier in its answer?
        leaked = [p.pattern for p in PATTERNS if p.search(text)]
        if who["customer_id"] in text or who["email"] in text:
            id_in_output += 1

        records.append({
            "run": i, "usage": full,
            "traced_usd": round(traced, 6), "billed_usd": round(billed, 6),
            "identifier_in_output": who["customer_id"] in text
                                    or who["email"] in text,
            "patterns_matched_in_output": leaked,
            "output": text,
        })
        print(f"  run {i:03d}  cache_read {full['cache_read_input_tokens']:>6}  "
              f"traced ${traced:.5f}  billed ${billed:.5f}")

    elapsed = time.time() - t0
    ok = [r for r in records if "error" not in r]
    print(f"\n{len(ok)}/{args.runs} runs succeeded in {elapsed:.0f}s")
    print(f"cache read tokens          {cache_reads:,}")
    print(f"cost from the trace        ${traced_usd:.4f}")
    print(f"cost from the usage report ${billed_usd:.4f}")
    if billed_usd:
        err = (traced_usd - billed_usd) / billed_usd * 100
        print(f"the trace is off by        {err:+.2f}%")
    print(f"\nidentifier reproduced in the model's own answer: "
          f"{id_in_output}/{len(ok)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "adapter": "anthropic", "model": args.model,
        "runs": args.runs, "elapsed_s": round(elapsed, 1),
        "prices_verified": PRICES_VERIFIED,
        "note": "Cost from the trace uses ONLY gen_ai.usage.* attributes, "
                "which is everything the GenAI conventions define. Cost from "
                "the usage report uses the provider's cache counters, which "
                "have no convention attribute.",
        "traced_usd": round(traced_usd, 6),
        "billed_usd": round(billed_usd, 6),
        "cache_read_tokens": cache_reads,
        "identifier_in_output": id_in_output,
        "records": records,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
