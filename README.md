# SimCSE Reproduction: Resource-Constrained Mechanism Study

A validated implementation-level reproduction of **unsupervised SimCSE** with
`bert-base-uncased`, followed by a controlled reduced-data study of training
scale, dropout, temperature, and representation geometry.

> **Scientific scope:** this project reproduces and validates the core
> unsupervised SimCSE training mechanism and studies its behavior under
> resource-constrained reduced-data training. It does **not** claim a complete
> reproduction of the original 1M-sentence paper-scale training run.

## Research questions

1. Does unsupervised SimCSE improve sentence similarity over raw BERT?
2. How does performance change from 10K → 25K → 50K training sentences?
3. Is independent stochastic dropout important to the method?
4. How sensitive is performance to contrastive temperature?
5. How do alignment and uniformity change across the completed conditions?
6. What explains the gap between the reduced-data model and the official
   pretrained checkpoint?

## Main results

| Experiment | Training size | STS-B Spearman |
|---|---:|---:|
| Raw BERT CLS | — | **30.18%** |
| SimCSE | 10K | **53.55%** |
| SimCSE | 25K | **62.59%** |
| SimCSE | 50K | **69.07%** |
| No dropout | 50K | **60.21%** |
| Fixed shared dropout mask, 0.10 | 50K | **41.07%** |
| Independent dropout, 0.20 | 50K | **76.38%** |
| Temperature 0.01 | 50K | **68.88%** |
| Temperature 1.00 | 50K | **58.33%** |
| Official pretrained checkpoint, local STS-B evaluation | — | **82.12%** |

The 10K → 25K → 50K curve improves monotonically:

```text
10K     53.55%
25K     62.59%
50K     69.07%
```

The incremental gains decrease from **+9.04 pp** to **+6.48 pp**. This is an
empirical observation over the tested range, not a universal scaling law.

## Key mechanism finding: dropout

The controlled dropout study gives:

```text
No dropout             60.21%
Fixed mask 0.10        41.07%
Independent dropout
0.20                   76.38%
```

The fixed-mask experiment is particularly informative. It keeps a nominal
dropout rate while removing independent stochastic perturbation between the
two views. Its substantially worse downstream result supports interpreting
dropout as a **view-generation mechanism**, not merely as generic
regularization.

Training loss alone is not sufficient: the fixed-mask condition achieved
lower training loss than no-dropout while producing substantially worse
STS-B performance.

## Temperature

```text
τ = 0.01     68.88%
τ = 1.00     58.33%
baseline
τ = 0.05     reference configuration
```

Temperature therefore materially affects the contrastive objective and
downstream sentence-similarity performance.

## Alignment and uniformity

The completed representation-geometry analysis is:

| Experiment | STS-B | Alignment ↓ | Uniformity ↓ |
|---|---:|---:|---:|
| Raw BERT | 30.18% | 0.1771 | -1.0560 |
| SimCSE 50K | 69.07% | 0.2924 | -2.5705 |
| No dropout | 60.21% | 0.4486 | -2.7144 |
| Fixed mask 0.10 | 41.07% | 0.4057 | -2.3718 |
| Dropout 0.20 | 76.38% | 0.1946 | -2.3814 |
| Temperature 0.01 | 68.88% | 0.1642 | -1.5738 |
| Temperature 1.00 | 58.33% | 0.1030 | -1.9329 |

Both metrics are lower-is-better.

Two important observations follow:

- **Uniformity alone is insufficient.** No-dropout has the strongest
  uniformity in this local set but only 60.21% STS-B.
- **Alignment alone is insufficient.** Temperature 1.00 has the strongest
  alignment but only 58.33% STS-B.

The strongest local STS-B result is instead associated with a useful
alignment–uniformity balance.

## Reference interpretation

The official Princeton-NLP release reports **76.25 Avg. STS** for
`unsup-simcse-bert-base-uncased`. That number is an average across seven STS
tasks, so it must not be compared directly with a local STS-B-only number.

The official repository also states that unsupervised SimCSE uses **1M
Wikipedia sentences**, batch size 64, and training with an MLP while testing
without the MLP. The repository's example training procedure evaluates during
training and saves the best checkpoint.

Accordingly, this project distinguishes:

- **implementation correctness**
- **reduced-data experimental performance**
- **official pretrained checkpoint performance**
- **paper-scale benchmark performance**

These are not interchangeable claims.

## What is validated

### Implementation

- Configuration system
- Wikipedia data pipeline
- Dynamic tokenization/collation
- Two stochastic views
- BERT-base encoder
- CLS extraction
- Training MLP
- Train-only MLP behavior
- Pooling implementations
- Similarity matrix
- Contrastive loss
- AdamW
- Gradient clipping
- Linear scheduler
- Checkpoint save/load
- RNG state save/load
- Deterministic resume

### Tests

- Configuration tests
- Experiment-directory tests
- Data-pipeline tests
- Model tests
- Pooler tests
- Loss tests
- Single-step training test
- Training-interface test
- Evaluation-embedding tests
- Checkpoint tests
- Checkpoint compatibility
- Resume test
- Final-checkpoint equivalence
- Alignment/uniformity metric tests

### Completed experiments

- Raw BERT baseline
- Official pretrained SimCSE evaluation
- 10K training
- 25K training
- 50K training
- No dropout
- Fixed shared dropout mask 0.10
- Independent dropout 0.20
- Temperature 0.01
- Temperature 1.00
- Alignment/uniformity for all completed geometry conditions

## Not completed

- Full 1M-sentence training
- Full paper-scale benchmark reproduction
- A complete pooling ablation study

The original project plan included pooling as a planned experiment, but it
is not part of the completed experimental evidence and should not be presented
as completed.

## Reproducibility principles

The research workflow deliberately separates:

```text
implementation correctness
        ↓
component validation
        ↓
controlled experiment
        ↓
downstream evaluation
        ↓
representation analysis
        ↓
discrepancy analysis
        ↓
scientific interpretation
```

The project uses the full 1,500-example STS-B validation split for the local
benchmark claims and records hardware/resource constraints. The 10K/25K/50K
experiments keep the algorithmic configuration fixed so that the data-size
comparison remains interpretable.

## Final scientific claim

> **We reproduced and validated the core unsupervised SimCSE training
> implementation and characterized its learning behavior, mechanism-level
> ablations, and representation geometry under a resource-constrained
> reduced-data training regime.**

## Deliverables

- `SimCSE_Reproduction_Research_Notes_UPDATED.md`
- `SimCSE_Final_Scientific_Synthesis.pdf`
- `SimCSE_Final_Scientific_Synthesis.md`
- `SimCSE_Final_Results_Table.csv`
- `simcse_scaling_curve.png`
- `simcse_ablation_results.png`
- `simcse_alignment_uniformity.png`
- source implementation archive

## Citation

If using the original SimCSE method, cite:

Gao, T., Yao, X., & Chen, D. (2021).
*SimCSE: Simple Contrastive Learning of Sentence Embeddings.*
Proceedings of EMNLP 2021.

## Status

**Experimental work: COMPLETE**

**Scientific synthesis: COMPLETE**

**Full 1M-sentence reproduction: NOT CLAIMED**

**Pooling ablation: NOT COMPLETED**
