# FactAssessor eval: comparison

Same 27 synthetic texts (282 sentences, one fact each; true, false, and mixed texts; 2–25 sentences). Recorded runs share the exact same atoms, hits, and pages, so differences come from the claim filter and judge alone, and latency is their cost (atomizer, search, and crawl replay instantly). Live runs include the whole pipeline but see different web results.

## Recorded evidence

Evidence: recorded (ddg search, crawl4ai).

### all

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 285 | 42% | 55% | 23% | 24% | 14% | 0.42 | 31.2s | 16.8s |
| laya (LayaClaimFilter + LayaJudge) | 285 | 94% | 97% | 91% | 7% | 0% | 0.04 | 1.8s | 1.3s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 285 | 94% | 96% | 91% | 5% | 2% | 0.05 | 3.9s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 285 | 99% | 98% | 97% | 4% | 0% | 0.02 | 2.9s | 1.2s |

### short (2–4 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 30 | 50% | 60% | 30% | 33% | 7% | 0.38 | 14.6s | 12.4s |
| laya (LayaClaimFilter + LayaJudge) | 30 | 100% | 100% | 100% | 0% | 0% | 0.00 | 1.3s | 1.0s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 30 | 93% | 96% | 90% | 0% | 7% | 0.04 | 3.3s | 1.4s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 30 | 100% | 100% | 100% | 0% | 0% | 0.00 | 2.5s | 1.3s |

### medium (5–10 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 70 | 40% | 43% | 17% | 32% | 14% | 0.49 | 31.2s | 16.8s |
| laya (LayaClaimFilter + LayaJudge) | 70 | 94% | 97% | 91% | 6% | 0% | 0.05 | 1.8s | 1.4s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 70 | 94% | 97% | 91% | 6% | 0% | 0.05 | 3.6s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 70 | 99% | 97% | 96% | 6% | 0% | 0.04 | 2.7s | 1.3s |

### long (15–25 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 185 | 41% | 58% | 24% | 19% | 15% | 0.38 | 201.8s | 79.3s |
| laya (LayaClaimFilter + LayaJudge) | 185 | 93% | 96% | 89% | 8% | 0% | 0.06 | 4.3s | 2.9s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 185 | 94% | 96% | 90% | 6% | 2% | 0.05 | 4.8s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 185 | 99% | 98% | 97% | 3% | 0% | 0.02 | 3.0s | 1.2s |

### true texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 97 | 40% | 59% | 24% | – | 16% | 0.43 | 20.8s | 14.8s |
| laya (LayaClaimFilter + LayaJudge) | 97 | 100% | 100% | 100% | – | 0% | 0.00 | 1.6s | 1.4s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 97 | 93% | 97% | 90% | – | 3% | 0.09 | 3.9s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 97 | 100% | 100% | 100% | – | 0% | 0.00 | 2.6s | 1.2s |

### false texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 86 | 49% | 40% | 20% | 29% | – | 0.62 | 31.2s | 11.7s |
| laya (LayaClaimFilter + LayaJudge) | 86 | 92% | 91% | 84% | 8% | – | 0.07 | 2.0s | 1.2s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 86 | 92% | 99% | 91% | 1% | – | 0.01 | 3.7s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 86 | 97% | 95% | 92% | 5% | – | 0.05 | 2.9s | 1.2s |

### mixed texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 102 | 37% | 66% | 25% | 16% | 10% | 0.19 | 38.7s | 30.4s |
| laya (LayaClaimFilter + LayaJudge) | 102 | 90% | 98% | 88% | 4% | 0% | 0.04 | 2.0s | 1.5s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 102 | 97% | 94% | 91% | 12% | 0% | 0.05 | 4.2s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 102 | 100% | 99% | 99% | 2% | 0% | 0.00 | 2.9s | 1.3s |

### Warm-up (not counted above)

- gliner: load 2.4s, claim filter 0.13s then 0.12s, judge 0.10s then 0.10s, full checks 17.2s, 17.0s
- laya: load 4.7s, claim filter 0.14s then 0.03s, judge 0.04s then 0.03s, full checks 1.9s, 1.2s
- llm-gpt5nano: load 0.0s, judge 1.32s then 1.51s, full checks 2.7s, 4.6s, 2.6s
- llm-gpt6luna: load 0.0s, judge 2.24s then 1.42s, full checks 2.9s, 1.6s

### Verdicts by ground truth: gliner

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 39 | 21 | 1 | 87 | 0 |
| false | 33 | 26 | 3 | 72 | 3 |

### Verdicts by ground truth: laya

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 147 | 0 | 1 | 0 | 0 |
| false | 9 | 112 | 4 | 8 | 4 |

### Verdicts by ground truth: llm-gpt5nano

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 135 | 3 | 10 | 0 | 0 |
| false | 7 | 123 | 3 | 4 | 0 |

### Verdicts by ground truth: llm-gpt6luna

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 148 | 0 | 0 | 0 | 0 |
| false | 5 | 129 | 0 | 3 | 0 |

## Live

Evidence: live (LLMAtomizer, Serper, crawl4ai).

### all

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| laya-live (LayaClaimFilter + LayaJudge) | 285 | 94% | 96% | 89% | 9% | 0% | 0.04 | 7.0s | 4.0s |

### short (2–4 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| laya-live (LayaClaimFilter + LayaJudge) | 30 | 97% | 100% | 97% | 0% | 0% | 0.02 | 5.2s | 3.2s |

### medium (5–10 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| laya-live (LayaClaimFilter + LayaJudge) | 70 | 96% | 96% | 91% | 9% | 0% | 0.05 | 6.8s | 4.0s |

### long (15–25 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| laya-live (LayaClaimFilter + LayaJudge) | 185 | 92% | 95% | 88% | 10% | 0% | 0.06 | 12.8s | 6.2s |

### true texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| laya-live (LayaClaimFilter + LayaJudge) | 97 | 99% | 100% | 99% | – | 0% | 0.02 | 7.0s | 4.0s |

### false texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| laya-live (LayaClaimFilter + LayaJudge) | 86 | 92% | 90% | 83% | 9% | – | 0.08 | 8.6s | 4.3s |

### mixed texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| laya-live (LayaClaimFilter + LayaJudge) | 102 | 90% | 96% | 86% | 8% | 0% | 0.03 | 6.7s | 4.0s |

### Warm-up (not counted above)

- laya-live: load 4.2s, claim filter 0.12s then 0.03s, judge 0.04s then 0.03s, full checks 3.9s, 4.0s

### Verdicts by ground truth: laya-live

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 145 | 0 | 2 | 1 | 0 |
| false | 12 | 110 | 7 | 4 | 4 |
