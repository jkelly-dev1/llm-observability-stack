# llm-observability-stack

OpenTelemetry GenAI tracing for agent runs, and a measurement of what the
trace quietly copies, what it silently under-bills, and which failures it never
keeps.

A personal learning project. The premise is that you instrument an agent so you
can see what it did. This repository measures what that instrumentation does
when nobody is looking at it: where the customer's data ends up, whether the
cost figure derived from it is true, and whether the run you actually needed is
still in the store when you go looking.

Every `gen_ai.*` name here is one the GenAI semantic conventions specify, and
`tests/test_conventions.py` pins the literal spelling of each. Four attributes
here are not GenAI names, `url.full`, `db.query.text`, `exception.message`,
`exception.stacktrace`, because the tool leg is an ordinary HTTP or database
client, and those surfaces carry customer data as often as the prompt does.
Leaving them out is how a redaction measurement flatters itself. The measurement
itself uses only the standard library.

## The one-sentence result

Redacting the prompt and the completion, what the instrumentation docs show you,
and what most stacks ship, leaves the same customer identifier readable on 9
of 11 surfaces in the same trace.

And a second one, from 20 real calls: cost computed from the conventional
attributes was 44% below the provider's own usage report, because a cached
prompt prefix is reported in a counter the conventions do not define. Measured:
`input_tokens 143`, `cache_read_input_tokens 2,579`.

## Why a trace is a second copy of your prompt data

One support turn. One customer. Here is where their email and account number
end up, and which of those places a redactor is usually pointed at:

| surface | what puts it there | reached by "redact the prompt"? |
|---|---|---|
| `gen_ai.input.messages` | newer convention, content in an attribute | yes |
| `gen_ai.output.messages` | the model quoting the customer back | yes |
| `gen_ai.user.message` event | **older convention, same text, different place** | no |
| `gen_ai.choice` event | older convention, the answer | no |
| `gen_ai.tool.call.arguments` | what the agent passed to the lookup tool | no |
| `gen_ai.tool.call.result` | the record that came back | no |
| `url.full` | the tool's own HTTP call, id in the query string | no |
| `db.query.text` | the SQL the tool ran, email inline | no |
| `exception.message` | the failure text, quoting the input | no |
| `exception.stacktrace` | frame arguments, quoting it again | no |

The two conventions are the trap. Content capture moved from span *events* to
span *attributes* between revisions. Both are deployed. A redactor written
against one never visits the other, and it is not obvious from either the code
or the spec that a second copy exists.

Measured over 500 runs, 4,872 planted identifiers:

| policy | still readable | surfaces left |
|---|---|---|
| none (control) | 4872/4872 | 11/11 |
| **prompt and completion** | **3893/4872** | **9/11** |
| all content attributes | 2806/4872 | 7/11 |
| content attributes and events | 1348/4872 | 4/11 |
| every string in the span | 0/4872 | 0/11 |

The last policy is the only one that reaches the URL, the SQL and the stack
trace, because those are not GenAI attributes at all; they belong to the HTTP
and database clients sitting in the same trace.

A real model does put the identifier in its answer. That is an assumption the
table above depends on, so it was checked against a real model: in 20 real calls
the account number came back in the model's own text 20 out of 20 times. The
output surface is real.

Read that check narrowly. The prompt says "Address them by name and include
their account identifier", so 20 of 20 measures INSTRUCTION-FOLLOWING, not
whether a model volunteers the identifier unprompted. It establishes that the
output surface can carry the identity, which is what the table needs. It does
not establish how often an unprompted assistant would put it there.

## The cost figure derived from a trace is too low

Someone asks what a feature costs. The trace is the only per-request record, so
the number gets computed from spans. The GenAI conventions define
`gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens`, and nothing else
about usage.

```
20 real calls, claude-sonnet-5, a cached system prefix:
  cost from the trace         $0.0204
  cost from the usage report  $0.0366
  the trace is off by          -44.38%
  cache read tokens           49,001
```

A provider reports a cache hit in a separate counter and excludes it from
`input_tokens`. There is no convention attribute for that counter, so those
tokens are not mispriced in the trace; they are absent from it. On a request
whose prefix is cached, the conventional attribute saw 143 of the 2,722 tokens
the request actually processed.

Offline, on 500 constructed runs, the same effect is worth -1.5%, and deduping
retried calls, the obvious correction, makes it -3.8%, because those attempts
really were billed. The spread between -1.5% and -44% is entirely how much of
the prompt is a cached shared prefix, which is a property of your agent rather
than of your tracing.

This repository had the sign backwards until the paid run. The offline model
treated a cache hit as a discount on tokens still present in the trace, which
made attribution over-bill. Twenty real calls said otherwise. The generator
now models what the provider does, the tests assert the corrected direction, and
the wrong version is written up in the bug log rather than quietly deleted.

## The sampling policy keeps the runs you did not need

Traces are expensive, so they are sampled, almost always head-based at a fixed
rate because that is the SDK default and the only policy needing no buffering.

| policy | traces kept | failures kept | buffers? |
|---|---|---|---|
| keep everything | 500 (100%) | 21/21 (100%) | no |
| **head 10%** | 43 (8.6%) | **1/21 (4.8%)** | no |
| head 1% | 7 (1.4%) | 0/21 (0%) | no |
| tail on any error span | 126 (25.2%) | 7/21 (33.3%) | yes |
| tail on run outcome | 21 (4.2%) | 21/21 (100%) | yes |

Head sampling decides at the root span, before anything has gone wrong, so it
keeps its rate of the failures for the same reason it keeps its rate of
everything.

The intuitive tail policy is the interesting row. "Keep any trace with an error
span" sounds like the fix. It keeps a quarter of all traces and still misses two
thirds of the failures, because retried calls leave ERROR spans on runs that
succeeded, and the failure this stack exists to catch, the agent answering
with the wrong customer's data, has no error span at all. It is a 200 OK with a
wrong answer in it.

Only the last policy captures every failure, and it needs an outcome signal the
trace does not contain: a judge, a thumbs-down, a downstream error. A tail
sampler cannot invent one.

## Trace-to-eval-case promotion, and why 0 of 500 promote

The path that ties this to an eval harness: a production run went wrong, you
have its trace, so you turn it into a regression case.

```
runnable cases: 0 of 500
    500  no expected output
    500  tool responses are not reproducible from the trace
```

A trace records what the model said. A regression case needs what it should have
said, and that is exactly the thing the case exists because you did not get.
`promote()` returns the case a trace can support and refuses to invent an
expectation, listing what is missing instead.

The zero is structural rather than measured, and that is a statement about the
design rather than a hedge. "no expected output" is appended unconditionally, so
`runnable()` is False for every trace this or any other corpus could produce.
The 500 is a denominator, not an outcome that came close to differing. Read the
row as an argument about what a trace is, not as a rate that might improve with
better instrumentation. The second row IS contingent: it counts the traces that
called a tool, and it would fall to zero on a corpus without tool use.

There is also a direct conflict between two sections of this repository: the
redaction section recommends masking identifiers, and a redacted trace cannot
be replayed as the run that happened. That tension is reported, not resolved.

## Claims backed by tests

| Claim | Test |
| --- | --- |
| Runs are deterministic, so two measurements are comparable | `tests/test_runs.py::test_runs_are_deterministic` |
| Every planted surface actually occurs in the corpus, so no policy gets credit for covering something that was never there | `tests/test_runs.py::test_every_planted_surface_actually_occurs_in_the_corpus`, `::test_every_surface_has_a_locator` |
| The identifier is really readable on the surface the ground truth says carries it | `tests/test_runs.py::test_every_planted_identifier_is_really_readable_where_it_was_planted` |
| The spans and the ground truth agree about the tokens the conventions do carry | `tests/test_runs.py::test_the_true_usage_matches_what_the_spans_report` |
| The failure this stack exists to catch leaves no error span on the answering call | `tests/test_runs.py::test_the_silent_failure_leaves_no_error_span` |
| A failed run really does answer with the wrong customer's data | `tests/test_runs.py::test_a_failed_run_really_answers_with_the_wrong_data` |
| The scrubber removes every pattern it claims to | `tests/test_redact.py::test_scrub_removes_each_pattern_it_claims_to` |
| The unredacted control leaks on all ten surfaces, so every policy is a difference from a real baseline | `tests/test_redact.py::test_the_unredacted_control_leaks_every_surface` |
| Redacting every string in the span reaches all ten, so none is unreachable by construction | `tests/test_redact.py::test_redacting_everything_leaks_nothing` |
| Redacting the prompt and the completion never visits the event carrying the same text under the older revision | `tests/test_redact.py::test_the_prompt_only_policy_misses_the_event_convention` |
| A broader policy leaks a strict subset of a narrower one, with no crossover | `tests/test_redact.py::test_broader_policies_leak_a_strict_subset` |
| The URL, the SQL and both exception surfaces are reached by no content-shaped policy | `tests/test_redact.py::test_client_and_exception_surfaces_need_the_widest_policy` |
| Coverage is graded per surface, not per span | `tests/test_redact.py::test_residual_grades_the_surface_and_not_the_span` (mutation-checked: the per-span version made three different policies produce identical numbers) |
| A surface name that does not exist is an error, not a silent pass | `tests/test_redact.py::test_an_unknown_surface_is_an_error_not_a_pass` |
| Every policy is scored against the same denominator | `tests/test_redact.py::test_every_policy_is_measured_against_the_same_denominator` |
| Every `gen_ai.*` attribute is spelled the way the conventions spell it, and the four non-GenAI ones are not GenAI names | `tests/test_conventions.py` (mutation-checked: rename any constant to a deprecated spelling and it fails) |
| The silent failure leaves no error span, selected by position rather than by status so the assertion cannot hold vacuously | `tests/test_runs.py::test_the_silent_failure_leaves_no_error_span` (mutation-checked: mark the answering span ERROR and it fails) |
| Summing spans under-bills, because the cached tokens are absent from the trace rather than mispriced in it | `tests/test_cost_sampling_promote.py::test_summing_spans_underbills_because_cached_tokens_are_absent` (the direction the paid run established, after this repository had the sign backwards) |
| Put the cached tokens back into `input_tokens` and the finding inverts, so the provider semantics are pinned | `tests/test_cost_sampling_promote.py::test_cached_tokens_are_absent_from_the_span_and_not_discounted_in_it` |
| Deduping retried calls, the obvious correction, moves the total further from the truth | `tests/test_cost_sampling_promote.py::test_deduping_retries_makes_the_error_worse_not_better` (mutation-checked on a tempting fix) |
| Retried model calls really are in the trace twice, each carrying tokens that were billed | `tests/test_cost_sampling_promote.py::test_retried_model_calls_are_really_in_the_trace_twice` |
| An unpriced model is refused rather than guessed at | `tests/test_cost_sampling_promote.py::test_an_unpriced_model_is_refused` |
| Head sampling keeps its rate of the failures and no more, because it decides before anything has gone wrong | `tests/test_cost_sampling_promote.py::test_head_sampling_keeps_its_rate_of_failures_and_no_more` |
| "Keep any trace with an error span" is dominated by successful retried runs and still misses failures | `tests/test_cost_sampling_promote.py::test_the_intuitive_tail_policy_keeps_retries_rather_than_failures` |
| Only a policy keyed on the run outcome captures every failure, and it has to buffer | `tests/test_cost_sampling_promote.py::test_only_an_outcome_signal_captures_every_failure` |
| No trace promotes into a runnable eval case | `tests/test_cost_sampling_promote.py::test_no_trace_promotes_into_a_runnable_case` |
| The blocker is always the missing expectation, never the missing answer: the trace does carry what the model said | `tests/test_cost_sampling_promote.py::test_the_missing_expectation_is_always_the_blocker` |
| Redacting a trace breaks the replay it would have supported, which is the conflict this repository reports rather than resolves | `tests/test_cost_sampling_promote.py::test_redacting_the_trace_breaks_the_replay_it_would_have_supported` |

What the table pins is direction and coverage, not the counts. 9 of 11, 3893 of
4872, 1 failure of 21 at head 10%: those come from `scripts/offline_demo.py`
over a constructed corpus whose retry, cache and failure rates are stated
constants in `obs/runs.py`, and every one of them scales with those constants.
The tests assert what does not move when the constants do, which surfaces each
policy reaches, that broader is strictly better, that the cost error is negative
and gets worse under dedupe, that only an outcome signal reaches 100%.

The paid run is absent for a different reason: it is evidence, not an
invariant. The 44% under-bill and the 20-of-20 identifier check are what 20
calls did on one day, a test that re-ran them would cost money on every commit,
and the cache ratio that produces 44% is a property of the prompt rather than
of the tracing. Raw outcomes are in `audit/`, and the semantics that run
established are pinned by the two cost tests above.

## Reproducing

```
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                              # 32 tests
.venv/bin/python scripts/offline_demo.py --runs 500 --json audit/offline.json
```

All of the above is free and needs no API key. The paid run prints its plan and
exits unless you pass `--confirm`:

```
ENV_FILE=~/.secrets/ai.env .venv/bin/python scripts/real_run.py
ENV_FILE=~/.secrets/ai.env .venv/bin/python scripts/real_run.py --confirm
```

| run | model | calls | cost | result |
|---|---|---|---|---|
| cost and surface check | claude-sonnet-5 | 20 | $0.037 | trace under-bills by 44%; identifier in output 20/20 |

## What this does not measure

- **Any deployment's rates.** The retry, cache and failure rates in
  `obs/runs.py` are stated constants. Every offline number scales with them,
  and none of them is a claim about production.
- **PII detection quality.** The matcher is a deliberately strong regex set,
  because the finding is about *coverage* and a weak matcher would confound
  the two.
- **Erasure or deletion propagation.** That is a sibling repository's subject
  and this one does not duplicate it: the question here is what the telemetry
  path copies, not what the data path retains.
- **Collector or exporter behavior.** Spans are modeled, not exported. A
  sampling decision taken in a collector rather than an SDK changes the
  operational cost and not the capture arithmetic.

## Layout

```
obs/spans.py      the GenAI convention attribute names and a flat span model
obs/runs.py       synthetic agent runs, planted identifiers, known usage truth
obs/redact.py     redaction policies as SURFACES VISITED, and per-surface grading
obs/cost.py       attribution from spans, and the gap against the truth
obs/sampling.py   head and tail policies, and what each keeps of the failures
obs/promote.py    trace to eval case, and what it refuses to invent
scripts/offline_demo.py   all four measurements, free
scripts/real_run.py       the paid leg: trace against the provider's own usage
```

## Related repositories

One of several small projects, each measuring one thing and publishing where it
fails:
[vlm-extraction-integrity](https://github.com/jkelly-dev1/vlm-extraction-integrity),
[prompt-injection-benchmark](https://github.com/jkelly-dev1/prompt-injection-benchmark),
[hardened-mcp-server](https://github.com/jkelly-dev1/hardened-mcp-server),
[ai-data-boundary-proxy](https://github.com/jkelly-dev1/ai-data-boundary-proxy),
[federated-retrieval-router](https://github.com/jkelly-dev1/federated-retrieval-router),
[least-privilege-agent](https://github.com/jkelly-dev1/least-privilege-agent),
[llm-eval-gate](https://github.com/jkelly-dev1/llm-eval-gate),
[citation-abstention-rag](https://github.com/jkelly-dev1/citation-abstention-rag),
[agentic-review-gate](https://github.com/jkelly-dev1/agentic-review-gate),
[typed-agent-service](https://github.com/jkelly-dev1/typed-agent-service),
[temporal-multi-agent](https://github.com/jkelly-dev1/temporal-multi-agent),
[ai-compliance-checker](https://github.com/jkelly-dev1/ai-compliance-checker),
[airgapped-ai-bundle](https://github.com/jkelly-dev1/airgapped-ai-bundle),
[agent-sandbox-escape](https://github.com/jkelly-dev1/agent-sandbox-escape),
[parser-eval](https://github.com/jkelly-dev1/parser-eval).

Two are worth reading directly against this one.
[llm-eval-gate](https://github.com/jkelly-dev1/llm-eval-gate) is the other end
of the trace-to-eval-case path: this repository measures why 0 of 500 traces
become runnable cases, and that one measures what happens once you have cases
and start trusting a judge to grade them.
[ai-data-boundary-proxy](https://github.com/jkelly-dev1/ai-data-boundary-proxy)
covers the DATA path: what a system retains and whether an erasure really erased
it. This one covers the TELEMETRY path, which is a second copy of the same
customer data that no erasure story usually mentions.

## License

MIT. See `LICENSE`.
