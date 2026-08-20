# SimCSE Reproduction: Final Research Report

## Abstract

This project presents an implementation-level reproduction and resource-constrained
experimental study of unsupervised SimCSE using BERT-base-uncased. The work validates
the core training pipeline, evaluates raw BERT and the official pretrained SimCSE
checkpoint, and investigates reduced-data training at 10K, 25K, and 50K sentences.
Controlled experiments examine dropout, stochastic-view construction, and contrastive
temperature. Representation geometry is analyzed through alignment and uniformity.

Across the reduced-data scaling study, STS-B Spearman improves from 53.55% at 10K
sentences to 62.59% at 25K and 69.07% at 50K, with decreasing marginal gains.
The completed dropout ablations show a strong dependence on independent stochastic
view generation: no dropout reaches 60.21%, a shared fixed dropout mask reaches
41.07%, while independent dropout at 0.20 reaches 76.38%. Temperature also has a
substantial effect, with 68.88% at 0.01 and 58.33% at 1.00.

The alignment/uniformity analysis shows that neither metric alone explains downstream
performance. No-dropout achieves the strongest uniformity among the completed local
conditions but only 60.21% STS-B, while temperature 1.00 achieves the strongest
alignment but only 58.33% STS-B. The evidence therefore supports an
alignment–uniformity balance interpretation.

This is a validated reduced-data reproduction and mechanism study. It is not a claim
of completing the original 1M-sentence paper-scale training.

---

## 1. Introduction

Sentence embeddings aim to map sentences into a vector space in which semantic
similarity is reflected by geometric similarity. SimCSE uses contrastive learning
to improve sentence representations by treating two stochastic forward passes of
the same sentence as a positive pair in the unsupervised setting.

The purpose of this project was not merely to implement the loss. The project was
designed as a research reproduction loop:

```text
paper understanding
      ↓
implementation
      ↓
component validation
      ↓
controlled training
      ↓
benchmark evaluation
      ↓
ablation
      ↓
representation analysis
      ↓
discrepancy analysis
      ↓
scientific interpretation
```

The original project plan explicitly targeted the main unsupervised SimCSE result,
no-dropout and fixed-mask failures, dropout probability, temperature, pooling, and
alignment/uniformity analysis.

The completed evidence supports the first five of these areas except that a complete
pooling ablation was not established. Pooling is therefore treated as an explicit
remaining gap rather than as a completed experiment.

---

## 2. Research Questions

The completed study addresses:

1. Does unsupervised SimCSE improve STS-B performance over raw BERT?
2. How does performance change as reduced training data increases from 10K to 50K?
3. Does independent stochastic dropout matter for unsupervised SimCSE?
4. How sensitive is performance to contrastive temperature?
5. How do alignment and uniformity change across the completed experimental
   conditions?
6. What explains the remaining difference between the reduced-data model and
   the official pretrained reference?

---

## 3. Implementation and Validation

The implementation was validated component by component before expensive training.

Validated components include:

- configuration system,
- Wikipedia data pipeline,
- dynamic tokenization and collation,
- two-view input construction,
- BERT-base encoder,
- CLS extraction,
- training MLP,
- train-only MLP behavior,
- pooling implementations,
- similarity matrix,
- contrastive loss,
- diagnostics,
- AdamW,
- gradient clipping,
- linear scheduling,
- checkpoint saving/loading,
- RNG state saving/loading,
- deterministic resume.

The validation suite includes configuration, experiment-directory, data-pipeline,
model, pooler, loss, training-step, training-interface, evaluation-embedding,
checkpoint, compatibility, resume, and final-checkpoint-equivalence tests.

The research notes identify deterministic checkpoint/resume behavior as part of the
validated implementation and distinguish implementation correctness from benchmark
performance.

---

## 4. Experimental Setup

### 4.1 Evaluation

The completed benchmark experiments use the full 1,500-example STS-B validation
split used by the project.

The primary metric is Spearman correlation between predicted sentence similarity
and the STS-B gold scores.

### 4.2 Reduced-data training

The main scaling study uses:

- 10,000 sentences
- 25,000 sentences
- 50,000 sentences

The scaling experiments keep the algorithmic configuration fixed so that changes
can be interpreted primarily as a training-data effect.

### 4.3 Ablations

The completed controlled conditions include:

- no dropout,
- fixed shared dropout mask at 0.10,
- independent standard dropout at 0.20,
- temperature 0.01,
- temperature 1.00.

The existing temperature 0.05 configuration serves as the baseline/reference.

### 4.4 Representation geometry

Alignment and uniformity were computed for the completed geometry conditions.

Both metrics use normalized sentence representations and are interpreted as
lower-is-better.

---

## 5. Main Results

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

### 5.1 Raw BERT versus SimCSE

Raw BERT achieves 30.18% STS-B Spearman.

The reduced-data SimCSE models reach:

```text
10K     53.55%
25K     62.59%
50K     69.07%
```

Thus every tested SimCSE training size substantially improves over the raw BERT
baseline.

### 5.2 Training-data scaling

The absolute improvements are:

```text
Raw BERT → 10K:   +23.37 pp
10K → 25K:         +9.04 pp
25K → 50K:         +6.48 pp
Raw BERT → 50K:   +38.89 pp
```

The marginal improvement decreases across the tested intervals.

This should be described as an empirical diminishing-return pattern over 10K–50K,
not as evidence for a universal scaling law.

---

## 6. Dropout Ablation

The completed dropout conditions are:

| Condition | STS-B |
|---|---:|
| No dropout | 60.21% |
| Fixed shared mask, 0.10 | 41.07% |
| Independent dropout, 0.20 | **76.38%** |

The fixed-mask condition is especially informative because it retains a nominal
dropout rate while changing the stochastic relationship between the two views.

The completed training logs also show that the fixed-mask condition can have a
lower training loss than no-dropout while producing substantially worse
downstream STS-B.

This supports the interpretation that the quality of stochastic positive-view
generation matters, rather than treating training loss as a sufficient measure
of representation quality.

### Interpretation

The evidence supports:

> Independent stochastic dropout is an important mechanism in unsupervised
> SimCSE because it generates distinct positive views of the same sentence.

The evidence does not justify claiming that dropout probability 0.20 is universally
optimal. It is simply the strongest value among the completed local conditions.

---

## 7. Temperature Ablation

The completed temperature results are:

| Temperature | STS-B |
|---:|---:|
| 0.01 | 68.88% |
| 0.05 baseline | 69.07% |
| 1.00 | 58.33% |

Temperature therefore materially affects downstream sentence similarity.

The results do not establish a complete temperature-response curve because only
two additional values were tested.

---

## 8. Alignment and Uniformity

The completed geometry measurements are:

| Experiment | STS-B | Alignment ↓ | Uniformity ↓ |
|---|---:|---:|---:|
| Raw BERT | 30.18% | 0.1771 | -1.0560 |
| SimCSE 50K | 69.07% | 0.2924 | -2.5705 |
| No dropout | 60.21% | 0.4486 | -2.7144 |
| Fixed mask 0.10 | 41.07% | 0.4057 | -2.3718 |
| Dropout 0.20 | 76.38% | 0.1946 | -2.3814 |
| Temperature 0.01 | 68.88% | 0.1642 | -1.5738 |
| Temperature 1.00 | 58.33% | 0.1030 | -1.9329 |

### 8.1 Uniformity alone is insufficient

No-dropout has the strongest uniformity in this local set:

```text
-2.7144
```

but achieves only:

```text
60.21% STS-B
```

Therefore, stronger uniformity by itself does not guarantee better sentence
similarity performance.

### 8.2 Alignment alone is insufficient

Temperature 1.00 has the strongest alignment:

```text
0.1030
```

but achieves only:

```text
58.33% STS-B
```

Therefore, stronger alignment by itself also does not guarantee better downstream
performance.

### 8.3 Alignment–uniformity balance

The strongest local STS-B result is the independent-dropout 0.20 condition:

```text
STS-B       76.38%
Alignment    0.1946
Uniformity  -2.3814
```

The completed results therefore support the interpretation that downstream
sentence-embedding quality depends on an effective balance between bringing
positive views together and maintaining a well-spread representation space.

This is a mechanistic interpretation supported by the completed controlled
conditions. It is not a claim that either metric is a sufficient predictor of
STS-B.

---

## 9. Discrepancy Analysis

### 9.1 Paper reference terminology

A final reference check established that the published SimCSE BERT-base
**76.25%** number is the average across seven STS datasets, not STS-B alone.

Therefore the local:

```text
76.38% STS-B
```

must not be described as exceeding the paper's:

```text
76.25% Avg. STS
```

Those are different metrics.

The local official-checkpoint result:

```text
82.12% STS-B
```

is also an STS-B validation result and is not numerically equivalent to the
published seven-task average.

### 9.2 Training protocol

The official training procedure evaluates during training and selects the
best checkpoint. The local reduced-data study evaluates the resulting
one-epoch checkpoint.

This is a real protocol difference and should be reported explicitly.

### 9.3 Training scale

The official released unsupervised model uses approximately 1M Wikipedia
sentences. The local study intentionally uses 10K, 25K, and 50K because
full CPU training was computationally impractical.

The remaining performance difference should therefore be interpreted in the
context of:

- training-data scale,
- checkpoint-selection protocol,
- resource constraints,
- and any remaining implementation/configuration differences.

The completed evidence does not isolate a single cause for the entire gap.

---

## 10. Limitations

### 10.1 Reduced-data regime

The main local training study does not reproduce the full 1M-sentence training
regime.

### 10.2 Paper-scale benchmark

The completed local reduced-data study focuses on STS-B. It does not reproduce
the complete seven-task evaluation suite reported in the paper.

### 10.3 Checkpoint selection

The local reduced-data experiments do not exactly reproduce the official
best-checkpoint selection procedure.

### 10.4 Pooling

Pooling was included in the original project plan but a complete pooling
ablation is not established in the completed evidence. It must therefore remain
marked as incomplete.

### 10.5 Geometry sample

The completed geometry analysis contains seven experimental conditions.
Consequently, correlations across conditions are descriptive rather than
statistically robust estimates.

---

## 11. What the Study Establishes

The evidence strongly supports the following statements:

1. The implementation passes extensive component and deterministic-resume
   validation.
2. Unsupervised SimCSE substantially improves over raw BERT under all tested
   reduced-data training sizes.
3. Increasing training data from 10K to 50K improves STS-B over the tested range.
4. The marginal improvement decreases over that range.
5. Independent stochastic dropout is important for the completed unsupervised
   SimCSE conditions.
6. Shared dropout masks can severely damage downstream performance.
7. Contrastive temperature materially affects performance.
8. Alignment and uniformity provide useful complementary geometric information.
9. Neither alignment nor uniformity alone is sufficient to explain downstream
   STS-B performance.
10. The project provides a validated reduced-data reproduction and mechanism study.

---

## 12. What the Study Does Not Establish

The evidence does not establish:

1. Completion of the original 1M-sentence training.
2. Exact reproduction of the paper's complete seven-task benchmark.
3. A universal scaling law beyond the tested 10K–50K range.
4. A universally optimal dropout probability of 0.20.
5. A universally optimal temperature.
6. A causal law in which alignment or uniformity alone determines STS-B.
7. A completed pooling ablation.

---

## 13. Final Conclusion

The central outcome of this reproduction is not a single benchmark number.

The project establishes a complete research loop from implementation validation
through controlled experimentation and representation analysis.

The strongest scientific conclusion is:

> **We reproduced and validated the core unsupervised SimCSE training
> implementation and characterized its learning behavior, mechanism-level
> ablations, and representation geometry under a resource-constrained
> reduced-data training regime.**

Within the tested range, more training data improves performance. Independent
dropout is critical to the stochastic positive-view mechanism. Temperature
changes the behavior of the contrastive objective. Finally, the geometry
results show that good sentence embeddings require a useful alignment–
uniformity balance rather than optimization of either quantity in isolation.

The study should therefore be presented as a **validated implementation-level
reproduction plus controlled mechanism study**, with the resource and
paper-scale limitations made explicit.

---

## 14. Reproducibility Checklist

- [x] Configuration system
- [x] Dataset pipeline
- [x] Two-view construction
- [x] Encoder
- [x] Poolers
- [x] Contrastive loss
- [x] Optimizer/scheduler
- [x] Checkpointing
- [x] RNG state
- [x] Deterministic resume
- [x] Component tests
- [x] Raw BERT baseline
- [x] Official pretrained reference
- [x] 10K training
- [x] 25K training
- [x] 50K training
- [x] No-dropout ablation
- [x] Fixed-mask ablation
- [x] Dropout 0.20 ablation
- [x] Temperature 0.01
- [x] Temperature 1.00
- [x] Alignment/uniformity
- [x] Discrepancy analysis
- [x] Scientific synthesis
- [ ] Full 1M-sentence training
- [ ] Full paper-scale benchmark
- [ ] Complete pooling ablation
