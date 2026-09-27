# FactAssessor eval: comparison

Same 27 synthetic texts (282 sentences, one fact each; true, false, and mixed texts; 2–25 sentences). Recorded runs share the exact same atoms, hits, and pages, so differences come from the claim filter and judge alone, and latency is their cost (atomizer, search, and crawl replay instantly). Live runs include the whole pipeline but see different web results.

## Recorded evidence

Evidence: recorded (ddg search, crawl4ai).

### all

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 285 | 37% | 59% | 22% | 16% | 14% | 0.41 | 14.3s | 4.3s |
| laya (LayaClaimFilter + LayaJudge) | 285 | 95% | 97% | 92% | 7% | 0% | 0.03 | 1.6s | 0.7s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 285 | 94% | 96% | 91% | 5% | 2% | 0.05 | 3.9s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 285 | 99% | 98% | 97% | 4% | 0% | 0.02 | 2.9s | 1.2s |

### short (2–4 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 30 | 47% | 64% | 30% | 27% | 7% | 0.38 | 7.1s | 4.1s |
| laya (LayaClaimFilter + LayaJudge) | 30 | 100% | 100% | 100% | 0% | 0% | 0.00 | 0.8s | 0.7s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 30 | 93% | 96% | 90% | 0% | 7% | 0.04 | 3.3s | 1.4s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 30 | 100% | 100% | 100% | 0% | 0% | 0.00 | 2.5s | 1.3s |

### medium (5–10 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 70 | 33% | 48% | 16% | 21% | 14% | 0.52 | 14.3s | 4.1s |
| laya (LayaClaimFilter + LayaJudge) | 70 | 94% | 97% | 91% | 6% | 0% | 0.04 | 1.6s | 0.7s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 70 | 94% | 97% | 91% | 6% | 0% | 0.05 | 3.6s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 70 | 99% | 97% | 96% | 6% | 0% | 0.04 | 2.7s | 1.3s |

### long (15–25 sent.)

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 185 | 37% | 62% | 23% | 12% | 15% | 0.35 | 37.9s | 5.8s |
| laya (LayaClaimFilter + LayaJudge) | 185 | 95% | 96% | 91% | 8% | 0% | 0.05 | 3.8s | 0.7s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 185 | 94% | 96% | 90% | 6% | 2% | 0.05 | 4.8s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 185 | 99% | 98% | 97% | 3% | 0% | 0.02 | 3.0s | 1.2s |

### true texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 97 | 37% | 56% | 21% | – | 16% | 0.45 | 15.9s | 4.1s |
| laya (LayaClaimFilter + LayaJudge) | 97 | 100% | 100% | 100% | – | 0% | 0.00 | 1.4s | 0.8s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 97 | 93% | 97% | 90% | – | 3% | 0.09 | 3.9s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 97 | 100% | 100% | 100% | – | 0% | 0.00 | 2.6s | 1.2s |

### false texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 86 | 38% | 52% | 20% | 19% | – | 0.58 | 11.5s | 4.2s |
| laya (LayaClaimFilter + LayaJudge) | 86 | 92% | 91% | 84% | 8% | – | 0.07 | 1.7s | 0.7s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 86 | 92% | 99% | 91% | 1% | – | 0.01 | 3.7s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 86 | 97% | 95% | 92% | 5% | – | 0.05 | 2.9s | 1.2s |

### mixed texts

| variant | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| gliner (GlinerClaimFilter + GlinerJudge) | 102 | 35% | 69% | 25% | 12% | 10% | 0.22 | 16.0s | 4.6s |
| laya (LayaClaimFilter + LayaJudge) | 102 | 93% | 98% | 91% | 4% | 0% | 0.03 | 1.9s | 0.7s |
| llm-gpt5nano (LLMJudge, no claim filter, openai:gpt-5-nano) | 102 | 97% | 94% | 91% | 12% | 0% | 0.05 | 4.2s | 1.3s |
| llm-gpt6luna (LLMJudge, no claim filter, openai:gpt-6-luna) | 102 | 100% | 99% | 99% | 2% | 0% | 0.00 | 2.9s | 1.3s |

### Warm-up (not counted above)

- gliner: load 1.9s, claim filter 0.13s then 0.12s, judge 0.10s then 0.10s, full checks 5.9s, 5.9s
- laya: load 2.4s, claim filter 0.25s then 0.03s, judge 0.08s then 0.03s, full checks 0.9s, 0.8s
- llm-gpt5nano: load 0.0s, judge 1.32s then 1.51s, full checks 2.7s, 4.6s, 2.6s
- llm-gpt6luna: load 0.0s, judge 2.24s then 1.42s, full checks 2.9s, 1.6s

### Verdicts by ground truth: gliner

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 34 | 21 | 1 | 92 | 0 |
| false | 22 | 28 | 1 | 83 | 3 |

### Verdicts by ground truth: laya

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 148 | 0 | 0 | 0 | 0 |
| false | 9 | 114 | 5 | 5 | 4 |

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
