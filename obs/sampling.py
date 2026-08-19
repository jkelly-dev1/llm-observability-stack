"""Sampling policies, and what each one keeps of the failures you traced for.

The point of the stack is the rare bad run. Latency dashboards are fed by
metrics, which are cheap and complete. Traces exist for the individual run that
went wrong, and traces are expensive, so they get sampled. Almost always
head-based at a fixed rate, because that is the default in every SDK and the
only policy that needs no buffering.

What that costs, exactly. Head sampling decides at the root span, before
anything has gone wrong, so it keeps rate p of the failures for the same reason
it keeps rate p of everything. Tail sampling decides after the run, so it can
keep every failure. At the cost of buffering every unfinished trace in memory,
which is a real operational cost this module does NOT pretend away.

The trap this module measures, which is not the obvious one. "Keep every trace
containing an error span" sounds like the tail policy that solves this. It is
not, because retried calls leave ERROR spans behind on runs that SUCCEEDED, and
the failure this repository cares about, the agent answering with the wrong
customer's data, has no error span at all. So the error-span policy keeps a
pile of successful runs and misses the failure. That is measured below, not
asserted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .spans import Trace


@dataclass(frozen=True)
class Policy:
    name: str
    decide: Callable[[Trace, random.Random], bool]
    buffers_traces: bool
    note: str = ""


def head(rate: float) -> Policy:
    """Decide at the root span, on a coin flip. The SDK default."""
    def decide(trace: Trace, rng: random.Random) -> bool:
        return rng.random() < rate
    return Policy(f"head_{int(rate * 100)}pct", decide, False,
                  "decided before anything happened; keeps `rate` of "
                  "everything, failures included")


def tail_on_error_span() -> Policy:
    """Keep a trace if any span in it has ERROR status."""
    def decide(trace: Trace, rng: random.Random) -> bool:
        return any(s.is_error() for s in trace.spans)
    return Policy("tail_any_error_span", decide, True,
                  "the intuitive tail policy, and the one that keeps retries "
                  "rather than failures")


def tail_on_run_outcome() -> Policy:
    """Keep a trace if the RUN failed, which needs an outcome signal.

    Requires something the trace does not carry by default: whether the answer
    was right. In a real system that is a judge, a user thumbs-down, or a
    downstream error: always a second source. A tail sampler cannot invent it.
    """
    def decide(trace: Trace, rng: random.Random) -> bool:
        return trace.failed()
    return Policy("tail_run_failed", decide, True,
                  "keeps exactly the failures, and needs an outcome signal "
                  "the trace does not contain")


def always() -> Policy:
    return Policy("keep_everything", lambda t, r: True, False,
                  "the baseline: no sampling at all")


DEFAULT_POLICIES = (
    always(), head(0.10), head(0.01), tail_on_error_span(),
    tail_on_run_outcome(),
)


def evaluate(traces, policies=DEFAULT_POLICIES, seed: int = 0) -> list[dict]:
    """For each policy: what fraction of traces kept, and of the failures.

    `failures_kept` is the number that matters. `kept_pct` is the bill.
    """
    out = []
    total = len(traces)
    failures = [t for t in traces if t.failed()]
    for pol in policies:
        rng = random.Random(f"obs-sample-{seed}-{pol.name}")
        kept = [t for t in traces if pol.decide(t, rng)]
        kept_fail = [t for t in kept if t.failed()]
        out.append({
            "policy": pol.name,
            "kept": len(kept),
            "kept_pct": round(len(kept) / total * 100, 1) if total else 0.0,
            "failures_kept": len(kept_fail),
            "failures_total": len(failures),
            "failure_capture_pct": (round(len(kept_fail) / len(failures) * 100, 1)
                                    if failures else 0.0),
            "buffers_traces": pol.buffers_traces,
            "note": pol.note,
        })
    return out
