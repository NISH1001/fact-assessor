# FactAssessor eval: LLMJudge, no claim filter (openai:gpt-5-nano)

27 texts, 282 sentences, 285 atoms. Evidence: recorded (ddg search, crawl4ai). Per-claim timeout 15s. Warm-up (not counted): load 0.0s, judge 1.32s then 1.51s, full checks 2.7s, 4.6s, 2.6s.

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| all | 285 | 94% | 96% | 91% | 5% | 2% | 0.05 | 3.9s | 1.3s |
| short (2–4 sent.) | 30 | 93% | 96% | 90% | 0% | 7% | 0.04 | 3.3s | 1.4s |
| medium (5–10 sent.) | 70 | 94% | 97% | 91% | 6% | 0% | 0.05 | 3.6s | 1.3s |
| long (15–25 sent.) | 185 | 94% | 96% | 90% | 6% | 2% | 0.05 | 4.8s | 1.3s |
| true texts | 97 | 93% | 97% | 90% | – | 3% | 0.09 | 3.9s | 1.3s |
| false texts | 86 | 92% | 99% | 91% | 1% | – | 0.01 | 3.7s | 1.3s |
| mixed texts | 102 | 97% | 94% | 91% | 12% | 0% | 0.05 | 4.2s | 1.3s |

## Verdicts by ground truth

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 135 | 3 | 10 | 0 | 0 |
| false | 7 | 123 | 3 | 4 | 0 |
