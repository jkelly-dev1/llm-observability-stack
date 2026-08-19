# SAMPLE_RUN

Every block below is captured output. Regenerate the free half with
`scripts/offline_demo.py`; the paid half needs `--confirm` and an API key.

## Tests

```
$ .venv/bin/python -m pytest -q
...........................                                              [100%]
32 passed in 0.17s
```

## The offline measurement (free, no model called)

```
$ .venv/bin/python scripts/offline_demo.py --runs 500 --json audit/offline.json
llm-observability-stack -- offline measurement
adapter: none (no model was called)   runs: 500
planted rates: tool retry 18%, model retry 9%, cached 35%, run failure 4%
runs that actually failed: 21

1. REDACTION: identifiers still readable, by policy
  policy                                 leaked  of surfaces
  none                               4872/4872    11/11
  prompt_and_completion              3893/4872     9/11
  all_content_attributes             2806/4872     7/11
  content_attributes_and_events      1348/4872     4/11
  every_string_in_the_span              0/4872     0/11

  The policy most stacks ship -- prompt_and_completion -- leaves 9 of 11 surfaces intact:
    choice_event
    db_statement
    exception_message
    exception_stacktrace
    http_url
    tool_call_arguments
    tool_result
    user_message_event

2. COST: what the trace says against what it cost
  method                 traced       true      error   error %
  spans_summed           6.6691     6.7712    -0.1021    -1.51%
  spans_deduped          6.5172     6.7712    -0.2540    -3.75%
  cached input tokens absent from the trace: 510,642 ($0.1021 never billed to anyone)
  retry input tokens a dedupe drops:        75,950 ($0.1519 under-billed)
  Both push the same way: a trace-derived total is too LOW.

3. SAMPLING: what each policy keeps of the failures
  policy                         kept    failures kept  buffers
  keep_everything          500 (100.0%)      21/21   (100.0%)    False
  head_10pct                43 ( 8.6%)       1/21   (  4.8%)    False
  head_1pct                  7 ( 1.4%)       0/21   (  0.0%)    False
  tail_any_error_span      126 (25.2%)       7/21   ( 33.3%)     True
  tail_run_failed           21 ( 4.2%)      21/21   (100.0%)     True

4. PROMOTION: traces that can become runnable eval cases
  runnable cases: 0 of 500
      500  no expected output
      500  tool responses are not reproducible from the trace
```

Read the four blocks together, because each one is a different way the same
trace is not what it looks like:

  1 the identifier survives on 9 of 11 surfaces under the policy most stacks
  ship, and the two conventions, attributes and events, are why 2 the cost
  derived from the trace is too LOW, and both effects push that way 3 head
  sampling keeps 4.8% of the failures; the intuitive tail policy keeps a quarter
  of everything and still misses two thirds of them 4 nothing promotes into a
  runnable eval case

## The paid run (20 calls, claude-sonnet-5, $0.037)

```
$ ENV_FILE=~/.secrets/ai.env .venv/bin/python scripts/real_run.py --confirm
model            claude-sonnet-5
design           20 agent runs x 1 model call = 20 calls
system prompt    8,329 chars, cached after the first call
estimated tokens 63,520 in / 4,400 out
ESTIMATED COST   $0.17  (list prices verified 2026-08-11, and this estimate treats every call as a cache MISS, so it errs high)
  run 000  cache_read      0  traced $0.00113  billed $0.00758
  run 001  cache_read   2579  traced $0.00096  billed $0.00147
  run 002  cache_read   2579  traced $0.00092  billed $0.00143
  ...
20/20 runs succeeded in 88s
cache read tokens          49,001
cost from the trace        $0.0204
cost from the usage report $0.0366
the trace is off by        -44.38%

identifier reproduced in the model's own answer: 20/20

wrote audit/real_run.json
```

This run corrected the repository; it did not confirm it. The offline model had
treated a cache hit as a discount applied to tokens that were still in the
trace, which made a trace-derived cost too HIGH. A real provider excludes the
cached prefix from `input_tokens` and reports it in a counter the GenAI
conventions do not define, so the tokens are absent rather than mispriced and
the error runs the other way: by 44% here.

Per call, the numbers that make it concrete:

```
  run 000  input_tokens 144  cache_creation 2579  cache_read     0
  run 001  input_tokens 143  cache_creation    0  cache_read  2579
  run 002  input_tokens 143  cache_creation    0  cache_read  2579
```

The conventional attribute saw 143 of the 2,722 tokens run 001 processed.

And the check that could have embarrassed the redaction result: the model
reproduced the customer's account identifier in its own answer 20 times out of
20. The output surface in the redaction table is real, not an assumption.

Sample answer, unedited:

```
Hi Elena Lindqvist (Account: CUS-365092), your order SO-4471 currently has a
status of "shipped." If you need more detailed tracking information, please
let us know and we can assist further.
```
