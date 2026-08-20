# SimCSE Claim Audit

## Purpose

Every major claim in the final report is classified as one of:

- **MEASURED** — directly supported by completed project measurements.
- **SOURCE-SUPPORTED** — supported by the supplied paper/project documentation.
- **INFERENCE** — interpretation derived from multiple completed measurements.
- **LIMITATION** — explicit boundary of the evidence.
- **NOT ESTABLISHED** — deliberately excluded from the claims.

## Claim audit

| Claim | Classification | Basis |
|---|---|---|
| Implementation components were validated | MEASURED | Project validation checklist |
| Deterministic checkpoint/resume was validated | MEASURED | Project validation checklist |
| Raw BERT = 30.18% STS-B | MEASURED | Completed evaluation |
| 10K = 53.55% | MEASURED | Completed evaluation |
| 25K = 62.59% | MEASURED | Completed evaluation |
| 50K = 69.07% | MEASURED | Completed evaluation |
| Marginal gain decreases 10K→25K→50K | INFERENCE from MEASURED | +9.04 pp then +6.48 pp |
| No-dropout = 60.21% | MEASURED | Completed ablation |
| Fixed-mask = 41.07% | MEASURED | Completed ablation |
| Dropout 0.20 = 76.38% | MEASURED | Completed ablation |
| Temperature 0.01 = 68.88% | MEASURED | Completed ablation |
| Temperature 1.00 = 58.33% | MEASURED | Completed ablation |
| Independent stochastic dropout is important | INFERENCE | Controlled dropout comparison |
| Fixed-mask result supports stochastic-view interpretation | INFERENCE | Fixed-mask vs independent dropout |
| Uniformity alone is insufficient | INFERENCE | No-dropout geometry + STS-B |
| Alignment alone is insufficient | INFERENCE | Temperature 1.00 geometry + STS-B |
| Alignment–uniformity balance is useful | INFERENCE | Combined geometry/STS-B comparison |
| Official pretrained checkpoint = 82.12% STS-B locally | MEASURED | Completed local evaluation |
| Published 76.25% is Avg. STS, not STS-B | SOURCE-SUPPORTED | Paper/reference interpretation |
| Official protocol uses best-checkpoint selection | SOURCE-SUPPORTED | Official training configuration |
| Official model uses ~1M Wikipedia sentences | SOURCE-SUPPORTED | Official project documentation |
| Full 1M reproduction was completed | NOT ESTABLISHED | Explicitly false/not claimed |
| 76.38% STS-B beats published 76.25 Avg. STS | NOT ESTABLISHED | Metrics are not comparable |
| Dropout 0.20 is universally optimal | NOT ESTABLISHED | Only one local value tested |
| Temperature optimum was found | NOT ESTABLISHED | Only two alternatives + baseline |
| Pooling ablation was completed | NOT ESTABLISHED | No completed evidence |
| Reduced-data study is a full paper reproduction | NOT ESTABLISHED | Resource/protocol limitations |
