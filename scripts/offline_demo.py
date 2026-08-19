#!/usr/bin/env python3
"""The whole offline measurement. No network, no cost, no API key.

    python scripts/offline_demo.py
    python scripts/offline_demo.py --runs 500 --json audit/offline.json

Four questions, one corpus of synthetic agent runs:

  1 REDACTION   how much of one customer's identity survives each redaction
                policy, reported per surface rather than as a total
  2 COST        what a trace says a run cost against what it really cost, and
                which missing attribute causes each part of the gap
  3 SAMPLING    what fraction of the failures each sampling policy keeps
  4 PROMOTION   how many traces can become runnable eval cases

What this measures and what it does not. Every number here is exact, because
the corpus is constructed and its ground truth is known. None of them is a
measurement of any real deployment's rates. The retry, cache and failure rates
are stated constants in obs/runs.py, and every result scales with them.
Scripts/real_run.py replaces the cost half with a provider's own usage report,
which is the only ground truth here that this repository did not write.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from obs import cost, sampling                                  # noqa: E402
from obs.promote import promote_all                             # noqa: E402
from obs.redact import POLICIES, residual                       # noqa: E402
from obs.runs import (P_CACHED, P_MODEL_RETRY, P_RUN_FAILS,     # noqa: E402
                      P_TOOL_RETRY, PLANTED_SURFACES, corpus)


def redaction_table(traces) -> list[dict]:
    planted = sum(len(t.truth["planted"]) for t in traces)
    rows = []
    for pol in POLICIES:
        copies = [copy.deepcopy(t) for t in traces]
        for t in copies:
            pol.apply(t)
        leaks = [r for t in copies for r in residual(t)]
        by_surface = Counter(r["surface"] for r in leaks)
        rows.append({
            "policy": pol.name,
            "leaked": len(leaks),
            "planted": planted,
            "leaked_pct": round(len(leaks) / planted * 100, 1),
            "surfaces_leaking": sorted(by_surface),
            "surfaces_clean": sorted(set(PLANTED_SURFACES) - set(by_surface)),
            "by_surface": dict(sorted(by_surface.items())),
            "note": pol.note,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    traces = corpus(args.runs)
    failures = sum(1 for t in traces if t.failed())

    print("llm-observability-stack -- offline measurement")
    print(f"adapter: none (no model was called)   runs: {args.runs}")
    print(f"planted rates: tool retry {P_TOOL_RETRY:.0%}, model retry "
          f"{P_MODEL_RETRY:.0%}, cached {P_CACHED:.0%}, run failure "
          f"{P_RUN_FAILS:.0%}")
    print(f"runs that actually failed: {failures}\n")

    # -- 1. redaction ------------------------------------------------------
    red = redaction_table(traces)
    print("1. REDACTION: identifiers still readable, by policy")
    print(f"  {'policy':<32} {'leaked':>12} {'of surfaces':>12}")
    for row in red:
        print(f"  {row['policy']:<32} {row['leaked']:>6}/{row['planted']:<5} "
              f"{len(row['surfaces_leaking']):>4}/{len(PLANTED_SURFACES)}")
    worst = red[1]
    print(f"\n  The policy most stacks ship -- {worst['policy']} -- leaves "
          f"{len(worst['surfaces_leaking'])} of {len(PLANTED_SURFACES)} "
          f"surfaces intact:")
    for s in worst["surfaces_leaking"]:
        print(f"    {s}")

    # -- 2. cost -----------------------------------------------------------
    print("\n2. COST: what the trace says against what it cost")
    summed = cost.drift(traces)
    deduped = cost.drift(traces, dedupe_retries=True)
    print(f"  {'method':<18} {'traced':>10} {'true':>10} {'error':>10} "
          f"{'error %':>9}")
    for d in (summed, deduped):
        print(f"  {d['method']:<18} {d['traced_usd']:>10.4f} "
              f"{d['true_usd']:>10.4f} {d['error_usd']:>+10.4f} "
              f"{d['error_pct']:>+8.2f}%")
    gap = cost.attribution_gap(traces)
    print(f"  cached input tokens absent from the trace: "
          f"{gap['cached_input_tokens']:,} "
          f"(${gap['cached_underbill_usd']:.4f} never billed to anyone)")
    print(f"  retry input tokens a dedupe drops:        "
          f"{gap['retry_input_tokens']:,} "
          f"(${gap['retry_underbill_usd_if_deduped']:.4f} under-billed)")
    print("  Both push the same way: a trace-derived total is too LOW.")

    # -- 3. sampling -------------------------------------------------------
    print("\n3. SAMPLING: what each policy keeps of the failures")
    rows = sampling.evaluate(traces)
    print(f"  {'policy':<22} {'kept':>12} {'failures kept':>16} {'buffers':>8}")
    for r in rows:
        print(f"  {r['policy']:<22} {r['kept']:>5} ({r['kept_pct']:>4.1f}%) "
              f"{r['failures_kept']:>7}/{r['failures_total']:<4} "
              f"({r['failure_capture_pct']:>5.1f}%) {str(r['buffers_traces']):>8}")

    # -- 4. promotion ------------------------------------------------------
    print("\n4. PROMOTION: traces that can become runnable eval cases")
    plain = promote_all(traces)
    print(f"  runnable cases: {plain['runnable']} of {plain['cases']}")
    for blocker, n in plain["blockers"].items():
        print(f"    {n:>5}  {blocker}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "adapter": "none",
            "note": "Constructed runs with known ground truth. Measures the "
                    "telemetry, not any deployment's rates.",
            "runs": args.runs,
            "planted_rates": {
                "tool_retry": P_TOOL_RETRY, "model_retry": P_MODEL_RETRY,
                "cached": P_CACHED, "run_failure": P_RUN_FAILS},
            "prices_verified": cost.PRICES_VERIFIED,
            "redaction": red,
            "cost": {"summed": summed, "deduped": deduped, "gap": gap},
            "sampling": rows,
            "promotion": plain,
        }, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
