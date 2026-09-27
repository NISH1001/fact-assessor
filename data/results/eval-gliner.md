# FactAssessor eval: GlinerClaimFilter + GlinerJudge

27 texts, 282 sentences, 285 atoms. Evidence: recorded (ddg search, crawl4ai). Per-claim timeout 15s. Warm-up (not counted): load 1.9s, claim filter 0.13s then 0.12s, judge 0.10s then 0.10s, full checks 5.9s, 5.9s.

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| all | 285 | 37% | 59% | 22% | 16% | 14% | 0.41 | 14.3s | 4.3s |
| short (2–4 sent.) | 30 | 47% | 64% | 30% | 27% | 7% | 0.38 | 7.1s | 4.1s |
| medium (5–10 sent.) | 70 | 33% | 48% | 16% | 21% | 14% | 0.52 | 14.3s | 4.1s |
| long (15–25 sent.) | 185 | 37% | 62% | 23% | 12% | 15% | 0.35 | 37.9s | 5.8s |
| true texts | 97 | 37% | 56% | 21% | – | 16% | 0.45 | 15.9s | 4.1s |
| false texts | 86 | 38% | 52% | 20% | 19% | – | 0.58 | 11.5s | 4.2s |
| mixed texts | 102 | 35% | 69% | 25% | 12% | 10% | 0.22 | 16.0s | 4.6s |

## Verdicts by ground truth

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 34 | 21 | 1 | 92 | 0 |
| false | 22 | 28 | 1 | 83 | 3 |
