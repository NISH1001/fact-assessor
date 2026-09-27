# FactAssessor eval: LayaClaimFilter + LayaJudge

27 texts, 282 sentences, 285 atoms. Evidence: recorded (ddg search, crawl4ai). Per-claim timeout 15s. Warm-up (not counted): load 2.4s, claim filter 0.25s then 0.03s, judge 0.08s then 0.03s, full checks 0.9s, 0.8s.

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| all | 285 | 95% | 97% | 92% | 7% | 0% | 0.03 | 1.6s | 0.7s |
| short (2–4 sent.) | 30 | 100% | 100% | 100% | 0% | 0% | 0.00 | 0.8s | 0.7s |
| medium (5–10 sent.) | 70 | 94% | 97% | 91% | 6% | 0% | 0.04 | 1.6s | 0.7s |
| long (15–25 sent.) | 185 | 95% | 96% | 91% | 8% | 0% | 0.05 | 3.8s | 0.7s |
| true texts | 97 | 100% | 100% | 100% | – | 0% | 0.00 | 1.4s | 0.8s |
| false texts | 86 | 92% | 91% | 84% | 8% | – | 0.07 | 1.7s | 0.7s |
| mixed texts | 102 | 93% | 98% | 91% | 4% | 0% | 0.03 | 1.9s | 0.7s |

## Verdicts by ground truth

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 148 | 0 | 0 | 0 | 0 |
| false | 9 | 114 | 5 | 5 | 4 |
