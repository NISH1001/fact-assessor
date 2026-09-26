# FactAssessor eval: GlinerClaimFilter + GlinerJudge

27 texts, 282 sentences, 285 atoms. Evidence: recorded (ddg search, crawl4ai). Per-claim timeout 600s. Warm-up (not counted): load 2.4s, claim filter 0.13s then 0.12s, judge 0.10s then 0.10s, full checks 17.2s, 17.0s.

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| all | 285 | 42% | 55% | 23% | 24% | 14% | 0.42 | 31.2s | 16.8s |
| short (2–4 sent.) | 30 | 50% | 60% | 30% | 33% | 7% | 0.38 | 14.6s | 12.4s |
| medium (5–10 sent.) | 70 | 40% | 43% | 17% | 32% | 14% | 0.49 | 31.2s | 16.8s |
| long (15–25 sent.) | 185 | 41% | 58% | 24% | 19% | 15% | 0.38 | 201.8s | 79.3s |
| true texts | 97 | 40% | 59% | 24% | – | 16% | 0.43 | 20.8s | 14.8s |
| false texts | 86 | 49% | 40% | 20% | 29% | – | 0.62 | 31.2s | 11.7s |
| mixed texts | 102 | 37% | 66% | 25% | 16% | 10% | 0.19 | 38.7s | 30.4s |

## Verdicts by ground truth

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 39 | 21 | 1 | 87 | 0 |
| false | 33 | 26 | 3 | 72 | 3 |
