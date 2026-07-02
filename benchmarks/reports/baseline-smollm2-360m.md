# Benchmark report — suite `baseline`

Run `bench-20260702-102043-f159b1` · started 2026-07-02T10:20:43.1002809Z · state **done** · backend `torch:cuda` on B_IG_B_OY_ (Microsoft Windows NT 10.0.26200.0)
Decoding: greedy · repeats 1 (median; 1 warmup discarded) · timings are wall-clock on `torch:cuda`

## Models

| model | step | params | config | sha256 |
|---|---|---|---|---|
| `smollm2-360m` | 0 | 361,821,120 | d960·L32·h15/5·ctx8192 | `24013f78b73a…` |

## Aggregates

| model | bpb ↓ | median tok/s | check pass | n | stop mix |
|---|---|---|---|---|---|
| `smollm2-360m` | 1.0035 | 45.85 | 23% | 20 | maxTokens:20 |

*bpb = bits per UTF-8 byte over the suite's held-out corpus (lower is better; comparable across tokenizers). Check pass rates over n cases — treat small n with caution. `contains` checks are floor signals, not grades.*

## Cases

| case | `smollm2-360m` |
|---|---|
| capital-fr | ~ 50% · 45.4 tok/s · maxTokens |
| arith-add | ~ 50% · 44.0 tok/s · maxTokens |
| arith-mul | ✗ 0% · 45.6 tok/s · maxTokens |
| count-words | ✗ 0% · 46.5 tok/s · maxTokens |
| opposite | ✗ 0% · 47.4 tok/s · maxTokens |
| complete-proverb | ~ 50% · 46.0 tok/s · maxTokens |
| capital-jp | ~ 50% · 43.7 tok/s · maxTokens |
| days-week | ✗ 0% · 44.9 tok/s · maxTokens |
| color-mix | ~ 50% · 46.3 tok/s · maxTokens |
| planet | ~ 50% · 45.2 tok/s · maxTokens |
| h2o | ~ 50% · 45.4 tok/s · maxTokens |
| list-3-fruits | ✗ 0% · 46.9 tok/s · maxTokens |
| json-object | ✗ 0% · 46.7 tok/s · maxTokens |
| rhyme | ✗ 0% · 46.6 tok/s · maxTokens |
| translate-es | ~ 50% · 42.1 tok/s · maxTokens |
| explain-loop | ✗ 0% · 47.1 tok/s · maxTokens |
| summarize | ✗ 0% · 45.7 tok/s · maxTokens |
| antonym-up | ~ 50% · 45.3 tok/s · maxTokens |
| throughput-short | ✗ 0% · 47.3 tok/s · maxTokens |
| throughput-long | ✗ 0% · 48.2 tok/s · maxTokens |

