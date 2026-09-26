# FactAssessor eval: LayaClaimFilter + LayaJudge

27 texts, 282 sentences, 285 atoms. Evidence: recorded (ddg search, crawl4ai). Warm-up (not counted): load 4.7s, claim filter 0.14s then 0.03s, judge 0.04s then 0.03s, full checks 1.9s, 1.2s.

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| all | 285 | 94% | 97% | 91% | 7% | 0% | 0.04 | 1.8s | 1.3s |
| short (2–4 sent.) | 30 | 100% | 100% | 100% | 0% | 0% | 0.00 | 1.3s | 1.0s |
| medium (5–10 sent.) | 70 | 94% | 97% | 91% | 6% | 0% | 0.05 | 1.8s | 1.4s |
| long (15–25 sent.) | 185 | 93% | 96% | 89% | 8% | 0% | 0.06 | 4.3s | 2.9s |
| true texts | 97 | 100% | 100% | 100% | – | 0% | 0.00 | 1.6s | 1.4s |
| false texts | 86 | 92% | 91% | 84% | 8% | – | 0.07 | 2.0s | 1.2s |
| mixed texts | 102 | 90% | 98% | 88% | 4% | 0% | 0.04 | 2.0s | 1.5s |

## Verdicts by ground truth

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 147 | 0 | 1 | 0 | 0 |
| false | 9 | 112 | 4 | 8 | 4 |
