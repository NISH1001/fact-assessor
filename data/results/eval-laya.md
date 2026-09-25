# FactAssessor on the synthetic set: LayaClaimFilter + LayaJudge

27 texts, 282 sentences (one fact each), 285 atoms. Texts run one at a time after the warm-up.

## Warm-up (not counted below)

Load 4.2s. Claim filter: first call 0.12s, then 0.03s. Judge: first call 0.04s, then 0.03s. Full checks: 3.9s, 4.0s.

## Overall

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| all | 285 | 94% | 96% | 89% | 9% | 0% | 0.04 | 7.0s | 4.0s |

## By length

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| short (2–4 sentences) | 30 | 97% | 100% | 97% | 0% | 0% | 0.02 | 5.2s | 3.2s |
| medium (5–10 sentences) | 70 | 96% | 96% | 91% | 9% | 0% | 0.05 | 6.8s | 4.0s |
| long (15–25 sentences) | 185 | 92% | 95% | 88% | 10% | 0% | 0.06 | 12.8s | 6.2s |

## By composition

| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |
|---|---|---|---|---|---|---|---|---|---|
| true | 97 | 99% | 100% | 99% | – | 0% | 0.02 | 7.0s | 4.0s |
| false | 86 | 92% | 90% | 83% | 9% | – | 0.08 | 8.6s | 4.3s |
| mixed | 102 | 90% | 96% | 86% | 8% | 0% | 0.03 | 6.7s | 4.0s |

## Verdicts by ground truth

| gold | supported | refuted | contested | unverified | skipped |
|---|---|---|---|---|---|
| true | 145 | 0 | 2 | 1 | 0 |
| false | 12 | 110 | 7 | 4 | 4 |

## Latency

Per text: median 7.0s, max 20.8s. Per claim (text latency / atoms): median 0.97s. First verdict: median 4.0s.
