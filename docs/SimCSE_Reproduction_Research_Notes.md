# SimCSE Reproduction — Complete Research Notes & Experimental Log

**Project:** `simcse-reproduction`  
**Task:** Reproduce unsupervised SimCSE with BERT-base-uncased, validate the implementation against the official Princeton-NLP reference, and document practical CPU-constrained results.

---

## 1. Project Goal

The goal is to reproduce the **unsupervised SimCSE** training pipeline represented by the official Princeton-NLP implementation.

Intended official configuration:

- Model: `bert-base-uncased`
- Dataset: Wikipedia 1M sentences
- Maximum sequence length: 32
- Batch size: 64
- Learning rate: `3e-5`
- Epochs: 1
- Temperature: `0.05`
- Training pooler: CLS + MLP
- `mlp_only_train: true`
- Random seed: 42
- Linear learning-rate scheduler
- Independent dropout masks for the two unsupervised views
- Evaluation representation: raw CLS when the MLP is train-only

Reference recorded in the project configuration:

```yaml
target_reference:
  official_model: princeton-nlp/unsup-simcse-bert-base-uncased
  reported_average_sts: 76.25
```

The reproduction is being conducted on a CPU-only laptop, which made full 1M-sentence training computationally impractical.

---

# 2. Repository and Environment

Project:

```text
D:\CODING\Paper Reproduction\simcse-reproduction
```

Virtual environment:

```text
simcse-reproduction
```

Normal shell:

```text
(simcse-reproduction) D:\CODING\Paper Reproduction\simcse-reproduction>
```

A Windows/IDE interruption temporarily left the shell outside the virtual environment. The environment was restored using the project's existing `.venv`; `uv init` was unnecessary because the project was already initialized.

Important: packages such as `transformers` were installed in the virtual environment.

---

# 3. Baseline Configuration

```yaml
experiment_name: unsupervised_bert_base

model_name: bert-base-uncased
mlp_only_train: true

dataset_path: data/raw/wiki1m_for_simcse.txt
max_seq_length: 32

batch_size: 64
learning_rate: 3.0e-5
epochs: 1
temperature: 0.05
dropout: 0.1
max_grad_norm: 1.0

seed: 42

shuffle: true
drop_last: true
num_workers: 0

target_reference:
  official_model: princeton-nlp/unsup-simcse-bert-base-uncased
  reported_average_sts: 76.25
```

Tests:

```text
python src/test_config.py
```

Result:

```text
Configuration test: PASSED
```

and:

```text
python src/test_experiment.py
```

Result:

```text
Experiment directory test: PASSED
```

---

# 4. Data Pipeline

Dataset:

```text
data/raw/wiki1m_for_simcse.txt
```

Full size:

```text
1,000,000 sentences
```

Observed first sentence:

```text
YMCA in South Australia
```

Observed second sentence:

```text
South Australia (SA)  has a unique position in Australia's history as, unlike the other states which were founded as colonies, South Australia began as a self governing province Many were attracted to this and Adelaide and SA developed as an independent and free thinking state.
```

Observed last sentence:

```text
He also reported a new English translation to appear in 2019, with a different introduction and appendices.
```

`WikipediaSentenceDataset`:

- one sentence per line
- strips whitespace
- skips empty lines
- supports `max_sentences`
- returns one sentence at a time

`SimCSECollator`:

- tokenizes
- truncates
- dynamically pads
- returns PyTorch tensors

The dataset does **not** create positive pairs. The same sentence is represented twice, and independent dropout later creates the stochastic views.

---

# 5. Data Pipeline Tests

Command:

```text
python src/test_data.py
```

All tests passed.

### Tiny dataset

```text
Number of sentences: 10
input_ids shape: torch.Size([4, 2, 10])
attention_mask shape: torch.Size([4, 2, 10])
```

Example:

```text
A dog is running.
```

Both token views were identical.

Results:

```text
Identical token views: PASSED
Tiny dataset test: PASSED
```

### Full Wikipedia dataset

```text
Number of sentences: 1000000
input_ids shape: torch.Size([64, 2, 32])
attention_mask shape: torch.Size([64, 2, 32])
```

Result:

```text
Full dataset batch test: PASSED
```

### Truncation

```text
input_ids shape: torch.Size([1, 2, 8])
attention_mask shape: torch.Size([1, 2, 8])
```

Result:

```text
Truncation test: PASSED
```

Final:

```text
ALL DATA PIPELINE TESTS PASSED
```

---

# 6. Model Architecture

The model is:

```text
BERT-base encoder
      |
      v
raw [CLS]
      |
      v
Linear(768 -> 768)
      |
      v
Tanh
      |
      v
training embedding
```

With:

```text
mlp_only_train=True
```

training uses CLS + MLP, while evaluation returns raw CLS.

BERT is loaded without its standard pooling layer:

```python
BertModel.from_pretrained(
    model_name,
    add_pooling_layer=False,
)
```

CLS:

```python
outputs.last_hidden_state[:, 0]
```

MLP initialization follows BERT's initializer:

```text
initializer_range = 0.02
weights ~ Normal(0, 0.02)
bias = 0
```

---

# 7. Model Tests

Command:

```text
python src/test_model.py
```

All passed.

### Two-view shapes

```text
Input: torch.Size([2, 2, 7])
z1: torch.Size([2, 768])
z2: torch.Size([2, 768])
```

### Training dropout

Representative result:

```text
z1 cosine similarity: 0.8686599731445312
z2 cosine similarity: 0.7726988792419434
Maximum z1 difference: 0.5619804859161377
Maximum z2 difference: 0.83474200963974
```

### Evaluation determinism

```text
Cosine similarity: 0.9999999403953552
Maximum difference: 0.0
```

### MLP-only-during-training

```text
Evaluation vs raw CLS maximum difference: 0.0
Training vs raw CLS maximum difference: 6.5632147789001465
```

### MLP initialization

```text
Expected initializer range: 0.02
MLP weight std: 0.020020615309476852
Maximum absolute bias: 0.0
```

Final:

```text
ALL MODEL TESTS PASSED
```

---

# 8. State Dict Cleanup

An intermediate implementation accidentally exposed:

```text
mlp.0.weight
mlp.0.bias
train_pooler.mlp.0.weight
train_pooler.mlp.0.bias
```

The duplicate `train_pooler` MLP namespace was removed.

Final verification:

```text
mlp.0.weight: True
train_pooler.mlp.0.weight: False
MLP state entries: ['mlp.0.weight', 'mlp.0.bias']
```

This was necessary for clean checkpoint compatibility.

---

# 9. Pooler Tests

Command:

```text
python src/test_pooler.py
```

All passed:

- `cls_before_pooler`
- `cls + mlp`
- `avg`
- `avg_top2`
- `avg_first_last`
- invalid pooler rejection

Final:

```text
ALL POOLER TESTS PASSED
```

---

# 10. Evaluation Embedding Tests

Command:

```text
python src/test_evaluation_embedding.py
```

Results:

```text
Explicit use_mlp=False shape: torch.Size([2, 768])
Automatic evaluation shape: torch.Size([2, 768])
Maximum difference: 0.0
```

Evaluation determinism:

```text
Maximum difference: 0.0
```

Training/evaluation distinction:

```text
Training embedding shape: torch.Size([2, 768])
Evaluation embedding shape: torch.Size([2, 768])
Maximum difference: 6.770946979522705
```

Sentence cosine:

```text
Evaluation cosine similarity: 0.9602562785148621
```

Final:

```text
ALL EVALUATION EMBEDDING TESTS PASSED
```

---

# 11. SimCSE Loss

The loss creates an `N x N` similarity matrix from normalized `z1` and `z2`.

The diagonal contains positive pairs.

Labels:

```text
[0, 1, ..., N-1]
```

Temperature:

```text
0.05
```

Negative pairs are off-diagonal entries.

---

# 12. Loss Tests

Command:

```text
python src/test_loss.py
```

All passed.

Similarity matrix:

```text
z1: torch.Size([4, 768])
z2: torch.Size([4, 768])
similarity matrix: torch.Size([4, 4])
```

Labels:

```text
tensor([0, 1, 2, 3])
```

Representative loss:

```text
1.4740763902664185
```

Gradient test:

```text
z1 gradient exists: True
z2 gradient exists: True
```

Positive-pair structure:

```text
Loss with strongly aligned positive pairs: 0.0
Loss with random pairs: 19.816303253173828
```

Final:

```text
ALL LOSS TESTS PASSED
```

---

# 13. Single Training Step

Command:

```text
python src/test_training_step.py
```

Configuration:

```text
Model: bert-base-uncased
Batch size: 4
Max length: 32
Learning rate: 3e-05
Temperature: 0.05
Device: cpu
```

Results:

```text
z1 shape: torch.Size([4, 768])
z2 shape: torch.Size([4, 768])

Loss: 0.27037313580513
Positive cosine: 0.8399868011474609
Negative cosine: 0.66851407289505

Parameters with gradients: 199
Total gradient norm: 25.830873292778218
Maximum parameter change: 3.0279159545898438e-05
```

Final:

```text
SINGLE TRAINING STEP TEST: PASSED
```

---

# 14. Training Interface

Command:

```text
python src/test_training_interface.py
```

Results:

```text
Number of sentences: 8
Batch input_ids shape: torch.Size([4, 2, 32])
Batch attention_mask shape: torch.Size([4, 2, 32])

Two-view batch structure: PASSED
Identical input views: True
Identical attention-mask views: True
Identical token views: PASSED

z1 shape: torch.Size([4, 768])
z2 shape: torch.Size([4, 768])

Single-call model interface: PASSED

Mean positive cosine: 0.816817045211792
Maximum absolute difference: 0.776429295539856

Dropout stochasticity: PASSED
```

---

# 15. Tiny Training Run

Command:

```text
python src/train.py --max-sentences 256 --experiment-name tiny_runner_test
```

Configuration:

```text
256 sentences
4 batches
1 epoch
CPU
```

Steps:

```text
1: Loss 1.330755 | PosSim 0.848911 | NegSim 0.615742
2: Loss 0.310414 | PosSim 0.812974 | NegSim 0.428712
3: Loss 0.113424 | PosSim 0.798553 | NegSim 0.331461
4: Loss 0.089390 | PosSim 0.804764 | NegSim 0.323873
```

Summary:

```text
Average loss: 0.460996
Average positive cosine: 0.816301
Average negative cosine: 0.424947
```

Checkpoint:

```text
experiments\tiny_runner_test\20260816_231924\checkpoint\epoch_1
```

---

# 16. Diagnostic Metric Correction

An intermediate training-interface change caused diagnostic cosine values to be printed at approximately 16 rather than in `[-1, 1]`.

Example of the bad output:

```text
PosSim 17.078285
NegSim 12.464424
```

This was identified as a diagnostic calculation issue.

After correction:

```text
PosSim 0.853914
NegSim 0.623221
```

The loss itself was not being interpreted from the incorrect diagnostic values; the metric reporting was fixed.

---

# 17. Linear Learning-Rate Scheduler

The trainer now uses a linear scheduler.

For 4 steps:

```text
Total training steps: 4
Warmup steps: 0
Initial learning rate: 3e-05
```

Observed:

```text
Step 1: 0.00002250
Step 2: 0.00001500
Step 3: 0.00000750
Step 4: 0.00000000
```

Example training:

```text
Step 1 | Loss 1.461460 | PosSim 0.853914 | NegSim 0.623221
Step 2 | Loss 0.330283 | PosSim 0.817796 | NegSim 0.430658
Step 3 | Loss 0.175542 | PosSim 0.801721 | NegSim 0.345528
Step 4 | Loss 0.109332 | PosSim 0.811395 | NegSim 0.349864
```

Summary:

```text
Average loss: 0.519154
Average positive cosine: 0.821206
Average negative cosine: 0.437318
```

---

# 18. Checkpoint Format

Current checkpoint keys:

```text
epoch
global_step
model_state_dict
optimizer_state_dict
scheduler_state_dict
rng_state
config
```

RNG state:

```text
python
numpy
torch
loader_generator
```

Scheduler state includes:

```text
base_lrs
last_epoch
_step_count
_is_initial
_get_lr_called_within_step
_last_lr
lr_lambdas
```

Example:

```text
Epoch: 1
Global step: 4
Scheduler last_epoch: 4
Scheduler base_lrs: [3e-05]
Scheduler last LR: [1.5e-05]
```

---

# 19. Checkpoint Tests

Command:

```text
python src/test_checkpoint.py
```

Results:

```text
Model state: PASSED
Optimizer state: PASSED
Scheduler state: PASSED
Metadata: PASSED
```

---

# 20. Checkpoint Compatibility

Command:

```text
python src/test_checkpoint_compatibility.py
```

Results:

```text
Current model parameter count: 199
Checkpoint parameter count: 199

Missing from checkpoint: 0
Unexpected checkpoint keys: 0
Shape mismatches: 0
```

Final:

```text
CHECKPOINT IS FULLY COMPATIBLE.
Strict model loading: PASSED
```

---

# 21. Training Resume

Command:

```text
python src/test_training_resume.py
```

Continuous run:

```text
Step 1/4 | Loss 1.61028230 | LR 0.0000225000
Step 2/4 | Loss 1.61016369 | LR 0.0000150000
Step 3/4 | Loss 1.61007464 | LR 0.0000075000
Step 4/4 | Loss 1.61001503 | LR 0.0000000000
```

Interrupted after step 2:

```text
Before checkpoint step 1/4 | Loss 1.61028230 | LR 0.0000225000
Before checkpoint step 2/4 | Loss 1.61016369 | LR 0.0000150000
```

Restored:

```text
global step: 2
scheduler step: 2
LR: 1.5e-05
```

Resume:

```text
After resume step 3/4 | Loss 1.61007464 | LR 0.0000075000
After resume step 4/4 | Loss 1.61001503 | LR 0.0000000000
```

Final:

```text
Model final state: PASSED
Optimizer final state: PASSED
Scheduler final state: PASSED
TRAINING RESUME TEST: PASSED
```

An earlier version of this test had:

```text
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

The comparison code was fixed.

---

# 22. RNG-State Reproducibility

A major improvement was adding:

```text
python RNG
numpy RNG
torch RNG
DataLoader generator RNG
```

to checkpoints.

A two-epoch continuous run was compared with an interrupted/resumed run.

Continuous epoch 2:

```text
Step 1 | Loss 0.125772
Step 2 | Loss 0.068128
Step 3 | Loss 0.042708
Step 4 | Loss 0.085753
```

Summary:

```text
Average loss: 0.080590
Average positive cosine: 0.809520
Average negative cosine: 0.307572
```

After restoring the epoch-1 checkpoint, the resumed run produced exactly the same values.

This established deterministic resume.

---

# 23. Final Checkpoint Equivalence

Command:

```text
python src/test_checkpoint_equivalence.py
```

Compared:

```text
Continuous checkpoint:
experiments\rng_continuous\20260817_013758\checkpoint\epoch_2\training_state.pt

Resumed checkpoint:
experiments\rng_resumed\20260817_014600\checkpoint\epoch_2\training_state.pt
```

Results:

```text
Epoch: 2
Global step: 8

Model state: IDENTICAL
Optimizer state: IDENTICAL
Scheduler state: IDENTICAL
RNG state: IDENTICAL
Configuration: IDENTICAL
```

Final:

```text
ALL FINAL CHECKPOINT STATES ARE IDENTICAL.
Deterministic resume test: PASSED
```

This is one of the strongest validation results in the project.

---

# 24. STS-B Evaluation

Full validation set:

```text
1500 examples
```

First example:

```text
A man with a hard hat is dancing.
A man wearing a hard hat is dancing.
Gold score: 5.0
```

---

# 25. Raw BERT Baseline

Command:

```text
python src/evaluate_sts.py
```

Using raw:

```text
bert-base-uncased
```

Full validation result:

```text
STS-B Spearman: 0.301829
STS-B Spearman (%): 30.18
```

A 100-example smoke test gave:

```text
STS-B Spearman: -0.010160
STS-B Spearman (%): -1.02
```

The 100-example result is only a smoke test and should not be used as the final benchmark.

---

# 26. Simple Cosine Sanity Checks

Raw BERT:

```text
A: A woman is reading.
B: A woman is reading a book.
Cosine similarity: 0.9633

A: A man is playing a guitar.
B: A person is playing music.
Cosine similarity: 0.8640
```

Tiny trained checkpoint:

```text
A: A woman is reading.
B: A woman is reading a book.
Cosine similarity: 0.9638

A: A man is playing a guitar.
B: A person is playing music.
Cosine similarity: 0.7017
```

These are sanity checks, not benchmark scores.

---

# 27. Official Pretrained SimCSE Evaluation

Command:

```text
python src/evaluate_sts.py --model princeton-nlp/unsup-simcse-bert-base-uncased --max-length 32
```

Full validation:

```text
1500 examples
```

Result:

```text
STS-B Spearman: 0.821220
STS-B Spearman (%): 82.12
```

Thus our evaluation pipeline produces:

```text
Official pretrained SimCSE: 82.12%
```

The project configuration separately records the paper/reference target:

```text
76.25
```

These should not be conflated.

The official checkpoint download produced an expected warning that pooler weights were unused when loading the checkpoint into the configured BERT architecture. This is consistent with using the hidden-state CLS representation rather than the standard BERT pooler.

---

# 28. Official Training Reference

The official unsupervised example uses approximately:

```bash
python train.py     --model_name_or_path bert-base-uncased     --train_file data/wiki1m_for_simcse.txt     --output_dir result/my-unsup-simcse-bert-base-uncased     --num_train_epochs 1     --per_device_train_batch_size 64     --learning_rate 3e-5     --max_seq_length 32     --evaluation_strategy steps     --metric_for_best_model stsb_spearman     --load_best_model_at_end     --eval_steps 125     --pooler_type cls     --mlp_only_train     --overwrite_output_dir     --temp 0.05     --do_train     --do_eval     --fp16
```

Important reference characteristics:

- BERT-base uncased
- Wikipedia 1M
- batch size 64
- LR 3e-5
- max length 32
- temperature 0.05
- CLS pooler
- MLP only during training
- one epoch
- evaluation during training
- FP16 in the original setup

Our local implementation preserves the core algorithm while operating on CPU.

---

# 29. Full 1M-Sentence Attempt

Command:

```text
python src/train.py --experiment-name unsup_bert_base_full
```

The run reported:

```text
Device: cpu
Number of sentences: 1000000
Number of batches per epoch: 15625
Total training steps: 15625
Warmup steps: 0
Initial learning rate: 3e-05
```

The run progressed through at least step 185 in the captured log.

Examples:

```text
Step 1   Loss 0.867656
Step 2   Loss 0.138821
Step 3   Loss 0.041478
...
Step 100 Loss 0.001621
...
Step 182 Loss 0.000970
Step 183 Loss 0.001050
Step 184 Loss 0.000235
Step 185 Loss 0.000391
```

The full-scale experiment was judged computationally impractical on the CPU laptop.

This is a hardware constraint, not an implementation failure.

---

# 30. Throughput Benchmark and Corrected Expectations

A 1,024-sentence run:

```text
python src/train.py --max-sentences 1024 --experiment-name throughput_1024
```

completed successfully:

```text
1024 sentences
16 batches
1 epoch
CPU
```

A short benchmark initially suggested a very optimistic full-run duration. That estimate was incorrect because the short benchmark did not predict sustained long-run performance.

The subsequent 1M run demonstrated that the CPU workload was far slower in practice.

Therefore:

> The earlier estimate that the 1M experiment would finish in roughly 1–1.5 hours is retracted.

The project should use measured sustained throughput from longer runs rather than extrapolating from a tiny benchmark.

---

# 31. Resource-Constrained Reproduction Decision

The full official setup requires:

```text
1,000,000 sentences
15,625 steps
```

On the available CPU laptop, this is not practical.

Therefore the project will use:

> **Resource-constrained / reduced-data reproduction**

The algorithm remains unchanged, but the number of training sentences is reduced.

This is a scientifically honest deviation and must be documented explicitly.

---

# 32. Recommended Reduced-Scale Experiments

Primary next experiment:

```text
10,000 sentences
1 epoch
batch size 64
max length 32
learning rate 3e-5
temperature 0.05
seed 42
```

Approximate steps:

```text
10000 / 64 ≈ 157
```

Optional:

```text
25,000 sentences
50,000 sentences
```

The purpose is to determine how much of the SimCSE learning effect can be reproduced on a CPU-feasible subset.

---

# 33. Correct Scientific Interpretation

The reduced-data run should **not** be claimed to reproduce the official 1M-sentence benchmark exactly.

Recommended wording:

> Resource-constrained reproduction of the SimCSE training procedure.

or:

> Reduced-data reproduction of unsupervised SimCSE with BERT-base.

What remains unchanged:

- BERT architecture
- CLS extraction
- MLP
- MLP initialization
- dropout
- two stochastic views
- contrastive objective
- temperature
- AdamW
- linear scheduler
- gradient clipping
- tokenization
- max sequence length
- seed
- checkpointing
- RNG restoration
- STS-B evaluation

Primary deviation:

```text
Training dataset size
```

---

# 34. Consolidated Experimental State

This section supersedes the earlier planning/status sections below Section 33. It is the
current source of truth for the empirical study.

## Final reference results

| System | Training data | Effective examples | Steps | STS-B Spearman | Runtime | Status |
|---|---:|---:|---:|---:|---:|---|
| Raw `bert-base-uncased` CLS | none | 0 | 0 | **30.18%** | — | Complete |
| Local 10K SimCSE | 10,000 | 9,984 | 156 | **53.55%** | **~55–70 min** | Complete |
| Local 25K SimCSE | 25,000 | 24,960 | 390 | **62.59%** | **2 h 50 min** | Complete |
| Local 50K SimCSE | 50,000 | 49,984 | 781 | **69.07%** | **not yet recorded** | Complete |
| Official pretrained SimCSE | official training | — | — | **82.12%** in local evaluation | — | Complete |
| Paper/project reference | ~1M | — | 15,625 | **76.25 reported** | — | Reference |

### 50K result

```text
Training sentences:       50,000
Effective examples:       49,984
Steps:                        781
Epochs:                         1
Final average loss:         0.003028
Average positive cosine:    0.791234
Average negative cosine:    0.051675
STS-B Spearman:             0.690704
STS-B Spearman (%):         69.07
Device:                     CPU
Evaluation split:           validation
Evaluation examples:       1,500
Evaluation representation:  raw CLS
MLP used during evaluation: False
```

Checkpoint:

```text
experiments/reduced_50k/20260818_121443/checkpoint/epoch_1/training_state.pt
```

The 50K run completed all 781 steps and the resulting checkpoint was evaluated on the
full 1,500-example STS-B validation split. The evaluation used raw CLS with the MLP
disabled, consistent with `mlp_only_train=true`.

## Training-data scaling

The local empirical curve is now:

```text
Raw BERT CLS:   30.18%
10K SimCSE:     53.55%
25K SimCSE:     62.59%
50K SimCSE:     69.07%
```

Absolute improvements:

```text
Raw BERT -> 10K:  +23.37 percentage points
10K -> 25K:        +9.04 percentage points
25K -> 50K:        +6.48 percentage points
Raw BERT -> 50K:  +38.89 percentage points
```

The successive marginal gains are therefore:

```text
+23.37 -> +9.04 -> +6.48 percentage points
```

This is consistent with a **diminishing marginal return** over the tested reduced-data
range. The final Phase A.1 analysis should quantify this carefully rather than claiming
a universal scaling law from only three trained data points.

The 50K model remains below both comparison references:

```text
50K local:            69.07%
Paper reference:      76.25%
Official pretrained:  82.12%
```

The gap from 50K to the paper reference is:

```text
76.25 - 69.07 = 7.18 percentage points
```

The gap from 50K to the locally evaluated official checkpoint is:

```text
82.12 - 69.07 = 13.05 percentage points
```

These gaps should be interpreted in the context of the reduced-data CPU experiment and
should not be presented as evidence that the implementation itself is deficient.

---

# 35. Final Reduced-Scale Experimental Protocol

The 10K, 25K, and 50K experiments use the same algorithmic configuration.

```text
Model:                       bert-base-uncased
Training objective:          unsupervised SimCSE
MLP only during training:    true
Max sequence length:         32
Batch size:                  64
Learning rate:               3e-5
Temperature:                 0.05
Dropout:                     0.1
Gradient clipping:           1.0
Epochs:                      1
Seed:                        42
Shuffle:                     true
Drop last:                   true
Workers:                     0
Scheduler:                   linear
Evaluation representation:  raw CLS
```

Only the number of training sentences changes.

| Training sentences | Effective examples | Steps |
|---:|---:|---:|
| 10,000 | 9,984 | 156 |
| 25,000 | 24,960 | 390 |
| 50,000 | 49,984 | 781 |

Because `drop_last=true`, the effective number of examples is the number of complete
batches:

```text
10,000 -> 156 * 64 = 9,984
25,000 -> 390 * 64 = 24,960
50,000 -> 781 * 64 = 49,984
```

This preserves the interpretation of the experiment as a controlled training-data-size
comparison.

---

# 36. Final 1M Computational Finding

The full 1M-sentence experiment was genuinely launched with the intended baseline setup:

```text
Training sentences: 1,000,000
Batches / optimizer steps: 15,625
Batch size: 64
Epochs: 1
Device: CPU
```

The run reached only a small fraction of the required steps before being stopped as
computationally impractical.

Therefore:

> **Full 1M-sentence SimCSE training is computationally impractical on the available CPU laptop.**

This is a hardware/resource limitation, not an implementation failure.

No final STS-B score is reported for the incomplete 1M run.

The 1M experiment is therefore classified as:

```text
Launch:              COMPLETE
Partial execution:   COMPLETE
Full training:       NOT COMPLETE
Final STS-B score:   NOT AVAILABLE
```

The reduced-data study is the intended empirical alternative.

---

# 37. Reproducibility Principles

1. Do not silently change the SimCSE algorithm to make CPU training faster.
2. Do not describe reduced-data training as paper-scale reproduction.
3. Record deviations from the official setup.
4. Keep checkpoint state complete enough for deterministic resume.
5. Validate individual components before expensive training.
6. Use the official pretrained checkpoint as an external reference.
7. Use the full STS-B validation split for final benchmark claims.
8. Record hardware and resource limitations.
9. Prefer reproducible feasible experiments over impractically long runs.
10. Separate implementation correctness from benchmark performance.
11. Keep the training-data scaling experiments algorithmically identical.
12. Treat the 10K/25K/50K curve as an empirical reduced-data study, not as a universal
   scaling law.

---

# 38. What Has Been Established

## Implementation

- [x] Configuration system
- [x] Wikipedia dataset
- [x] Dynamic tokenization/collation
- [x] Two-view input construction
- [x] BERT-base encoder
- [x] CLS extraction
- [x] Training MLP
- [x] MLP initialization
- [x] Train-only MLP behavior
- [x] Poolers
- [x] Similarity matrix
- [x] Contrastive loss
- [x] Diagnostics
- [x] AdamW
- [x] Gradient clipping
- [x] Linear scheduler
- [x] Checkpoint saving/loading
- [x] RNG state saving/loading
- [x] Deterministic resume

## Validation

- [x] Configuration tests
- [x] Experiment-directory tests
- [x] Data-pipeline tests
- [x] Model tests
- [x] Pooler tests
- [x] Loss tests
- [x] Single-training-step test
- [x] Training-interface test
- [x] Evaluation-embedding tests
- [x] Checkpoint tests
- [x] Checkpoint compatibility
- [x] Resume test
- [x] Final checkpoint equivalence

## Evaluation

- [x] Full 1,500-example STS-B validation loading
- [x] Raw BERT baseline
- [x] Official pretrained SimCSE evaluation
- [x] 10K evaluation
- [x] 25K evaluation
- [x] 50K evaluation
- [x] Simple cosine sanity checks

## Reduced-data experiments

- [x] 10K training
- [x] 10K STS-B evaluation
- [x] 25K training
- [x] 25K STS-B evaluation
- [x] 25K result analysis
- [x] 50K training
- [x] 50K STS-B evaluation

## Not completed

- [ ] Full 1M-sentence training
- [ ] Full paper-scale benchmark reproduction

---

# 39. Current Scientific Position

The project can now be described as:

> **A validated implementation-level reproduction of unsupervised SimCSE with
> BERT-base-uncased, including deterministic checkpoint/resume behavior and evaluation
> against the official pretrained SimCSE model, followed by a resource-constrained
> reduced-data training study on CPU hardware.**

The central empirical result is:

```text
Training data increases
        ↓
STS-B performance increases
        ↓
but the marginal gain decreases
over the tested 10K–50K range.
```

The strongest evidence is:

```text
10K -> 53.55%
25K -> 62.59%
50K -> 69.07%
```

The project should **not** claim:

> "We reproduced the original SimCSE 1M-sentence training result."

The appropriate claim is:

> **We reproduced and validated the core unsupervised SimCSE training implementation and
> demonstrated its learning behavior under a resource-constrained reduced-data training
> regime.**

---

# 40. Phase A — Analysis and Scientific Interpretation

**Current status:** Phase A.1 (training-data scaling), Phase A.2 (controlled
ablations), and Phase A.3 (alignment/uniformity geometry) are **COMPLETE**.
No additional training is required for these phases.

The following sections supersede all earlier planning/status statements below
Section 39.

## Phase A.1 — Training-data scaling: COMPLETE

The controlled reduced-data curve is:

```text
Raw BERT CLS:   30.18%
10K SimCSE:     53.55%
25K SimCSE:     62.59%
50K SimCSE:     69.07%
```

Absolute changes:

```text
Raw BERT -> 10K:  +23.37 percentage points
10K -> 25K:        +9.04 percentage points
25K -> 50K:        +6.48 percentage points
Raw BERT -> 50K:  +38.89 percentage points
```

The observed 10K–25K–50K sequence shows diminishing marginal improvement over
the tested range. This is an empirical observation, not a universal scaling law.

Reference gaps at 50K:

```text
50K local:            69.07%
Paper/project value:  76.25%
Official pretrained:  82.12%

Gap to paper value:    7.18 percentage points
Gap to official model: 13.05 percentage points
```

The 50K result is therefore best interpreted as a successful
resource-constrained reproduction and learning-behavior study, not as a
paper-scale 1M-sentence reproduction.

### Primary scaling conclusion

> Increasing the amount of unsupervised training data from 10K to 50K
> consistently improved STS-B performance, while the incremental gain became
> smaller across the tested intervals.

---

# 41. Phase A.2 — Controlled Ablations: COMPLETE

All planned controlled ablations under the current compute budget are complete.

## Final ablation table

| Experiment | Training size | Batch | Temperature | Dropout | Fixed mask | STS-B Spearman | Runtime | Status |
|---|---:|---:|---:|---:|---|---:|---:|---|
| No dropout | 50K | 64 | 0.05 | 0.00 | No | **60.21%** | **4 h 10 min** | Complete |
| Fixed dropout mask | 50K | 64 | 0.05 | 0.10 | Yes | **41.07%** | **5 h 15 min** | Complete |
| Standard dropout | 50K | 64 | 0.05 | 0.20 | No | **76.38%** | **5 h 52 min** | Complete |
| Temperature | 50K | 64 | 0.01 | 0.10 | No | **68.88%** | **5 h 50 min** | Complete |
| Temperature | 50K | 64 | 1.00 | 0.10 | No | **58.33%** | **5 h 48 min** | Complete |

The ablations are separate from the 10K/25K/50K scaling curve. The scaling
experiments keep the baseline configuration fixed; the ablations vary one
mechanism at a time.

### Ablation interpretation

**Dropout mechanism**

- Removing dropout substantially reduces STS-B performance.
- Sharing the same 0.10 dropout mask between the two views performs even worse.
- Standard independent dropout at 0.20 reaches **76.38%**, the strongest
  result among the tested 50K ablations.
- The fixed-mask condition is a mechanism ablation: it removes the
  independent stochastic perturbation that creates distinct positive views.

**Temperature**

- 0.01 gives **68.88%**.
- 1.00 gives **58.33%**.
- The baseline 0.05 remains the reference configuration for the main
  reproduction curve.
- The results demonstrate that contrastive temperature materially affects
  downstream STS-B performance.

---

# 42. Phase A.3 — Alignment and Uniformity: COMPLETE

Representation geometry has now been measured for the complete final 50K
experiment set.

## Methodology

The implementation follows the Wang–Isola metrics used in the SimCSE analysis:

```text
Alignment:
    mean || normalize(f(x)) - normalize(f(x+)) ||²

Uniformity:
    log E[ exp(-2 ||normalize(f(x)) - normalize(f(y))||²) ]
```

Both metrics are **lower-is-better**.

For this analysis:

- STS-B validation split: **1,500 pairs**
- Positive alignment pairs: **208 pairs with gold score > 4**
- Uniformity population: **2,910 unique sentences**
- Maximum sequence length: **32**
- Batch size: **64**
- Evaluation representation: **raw CLS**
- SimCSE MLP: disabled during evaluation
- Uniformity excludes self-pairs and duplicate pair counting

The analysis was implemented in `src/alignment_uniformity.py` and
`src/analyze_all_geometry.py`, with dedicated metric tests.

## Final geometry results

| Experiment | STS-B | Alignment ↓ | Uniformity ↓ |
|---|---:|---:|---:|
| Raw BERT | **30.18%** | **0.1771** | **-1.0560** |
| SimCSE 50K | **69.07%** | **0.2924** | **-2.5705** |
| No dropout | **60.21%** | **0.4486** | **-2.7144** |
| Fixed mask 0.10 | **41.07%** | **0.4057** | **-2.3718** |
| Standard dropout 0.20 | **76.38%** | **0.1946** | **-2.3814** |
| Temperature 0.01 | **68.88%** | **0.1642** | **-1.5738** |
| Temperature 1.00 | **58.33%** | **0.1030** | **-1.9329** |

Raw values are preserved in `geometry_results.csv`.

## Geometry findings

### 1. SimCSE strongly improves uniformity

Raw BERT uniformity is:

```text
-1.0560
```

while the main 50K SimCSE model reaches:

```text
-2.5705
```

Because lower is better, this is a substantial improvement in representation
spread/isotropy.

### 2. Uniformity improvement alone does not explain downstream quality

The no-dropout model achieves the strongest uniformity among the tested local
models:

```text
Uniformity = -2.7144
```

yet its STS-B score is only:

```text
60.21%
```

This is important: better uniformity by itself does not guarantee better
sentence representations.

### 3. Alignment is essential to the interpretation

The no-dropout model has poor alignment:

```text
Alignment = 0.4486
```

and the fixed-mask model is also poor:

```text
Alignment = 0.4057
```

Both conditions substantially underperform standard stochastic dropout.

The best STS-B ablation, standard dropout 0.20, has:

```text
STS-B      = 76.38%
Alignment  = 0.1946
Uniformity = -2.3814
```

This is a much better balance between keeping positive pairs close and
spreading representations.

### 4. Alignment alone is also insufficient

Temperature 1.00 produces the numerically best alignment:

```text
Alignment = 0.1030
```

but only:

```text
STS-B = 58.33%
```

Its uniformity is also much worse than the main SimCSE model:

```text
Uniformity = -1.9329
```

Thus the results support the central representation-geometry interpretation:
**good sentence embeddings require a useful balance of alignment and
uniformity rather than optimizing either metric independently.**

### 5. The main 50K SimCSE model improves uniformity while retaining usable alignment

Compared with raw BERT:

```text
Raw BERT:
    Alignment  = 0.1771
    Uniformity = -1.0560
    STS-B      = 30.18%

SimCSE 50K:
    Alignment  = 0.2924
    Uniformity = -2.5705
    STS-B      = 69.07%
```

The main model therefore trades some alignment for a very large uniformity
improvement while producing a much stronger STS-B result.

This should be described as an observed geometry/result relationship, not as
proof that one metric alone causes the downstream improvement.

---

# 43. Phase A.3 — Geometry/STS-B Synthesis

The completed geometry analysis gives a stronger interpretation of the
ablation results than training loss alone.

The key pattern is:

```text
                     Uniformity
                         ↑
                         |
          no-dropout     |       SimCSE / dropout
                         |
                         |
Raw BERT ----------------+------------------------→ Alignment quality
                         |
                  fixed-mask
```

The exact plot should be generated from the saved geometry results rather than
reconstructed from this schematic.

The scientifically defensible conclusion is:

> The experiments support the SimCSE representation-geometry hypothesis:
> contrastive training can substantially improve uniformity, but downstream
> sentence-similarity quality depends on maintaining an appropriate alignment–
> uniformity balance.

The ablations further show why independent stochastic dropout matters:
conditions that remove or share the dropout perturbation can improve
uniformity while simultaneously damaging positive-pair alignment and STS-B
performance.

---

# 44. Phase A.4 — Discrepancy Analysis: NEXT PHASE

The next phase is **not additional training**. It is discrepancy analysis and
scientific interpretation.

The goal is to explain the remaining differences between:

```text
Raw BERT
    ↓
Local reduced-data SimCSE
    ↓
Best local ablation
    ↓
Official pretrained SimCSE
    ↓
Paper/reference value
```

The analysis should use the already completed experiments and should not add
new uncontrolled runs unless a specific unresolved discrepancy requires one.

## Questions for the next phase

1. Why does the 50K baseline reach 69.07% while the official pretrained model
   reaches 82.12%?
2. Why does the 50K standard-dropout 0.20 ablation reach 76.38%, exceeding the
   paper/reference value of 76.25% despite using only 50K sentences?
3. Why does the main 50K baseline at dropout 0.10 produce substantially lower
   STS-B than the 0.20 ablation?
4. How do the alignment/uniformity values explain the differences among the
   baseline and ablations?
5. Which differences are attributable to training-data size, and which are
   attributable to stochastic/hyperparameter choices?
6. Are there any implementation/configuration differences remaining between the
   local reproduction and the official Princeton-NLP setup?
7. Which conclusions are robust across the completed experiments?

## Controlled discrepancy checklist

```text
[ ] Model/checkpoint
[ ] Tokenizer
[ ] Dataset identity and preprocessing
[ ] Training-data size
[ ] Effective examples after drop_last
[ ] Number of optimizer steps
[ ] Batch size
[ ] Learning rate
[ ] Scheduler
[ ] Temperature
[ ] Dropout probability
[ ] Independent vs shared dropout masks
[ ] MLP-only-training behavior
[ ] Evaluation pooling
[ ] Random seed
[ ] STS-B split
[ ] STS-B metric implementation
[ ] Alignment positive-pair definition
[ ] Uniformity population and pair handling
[ ] CPU/resource constraints
```

The objective is explanation, not forced numerical agreement.

---

# 45. Final Scientific Status

The project has now established:

## Implementation

- [x] Configuration system
- [x] Wikipedia dataset
- [x] Dynamic tokenization/collation
- [x] Two-view input construction
- [x] BERT-base encoder
- [x] CLS extraction
- [x] Training MLP
- [x] MLP initialization
- [x] Train-only MLP behavior
- [x] Poolers
- [x] Similarity matrix
- [x] Contrastive loss
- [x] Diagnostics
- [x] AdamW
- [x] Gradient clipping
- [x] Linear scheduler
- [x] Checkpoint saving/loading
- [x] RNG state saving/loading
- [x] Deterministic resume

## Validation

- [x] Configuration tests
- [x] Experiment-directory tests
- [x] Data-pipeline tests
- [x] Model tests
- [x] Pooler tests
- [x] Loss tests
- [x] Single-training-step test
- [x] Training-interface test
- [x] Evaluation-embedding tests
- [x] Checkpoint tests
- [x] Checkpoint compatibility
- [x] Resume test
- [x] Final checkpoint equivalence
- [x] Alignment/uniformity metric tests

## Evaluation

- [x] Full 1,500-example STS-B validation loading
- [x] Raw BERT baseline
- [x] Official pretrained SimCSE evaluation
- [x] 10K evaluation
- [x] 25K evaluation
- [x] 50K evaluation
- [x] All completed ablation evaluations
- [x] Alignment analysis
- [x] Uniformity analysis
- [x] Geometry/STS-B synthesis

## Reduced-data experiments

- [x] 10K training
- [x] 10K STS-B evaluation
- [x] 25K training
- [x] 25K STS-B evaluation
- [x] 25K result analysis
- [x] 50K training
- [x] 50K STS-B evaluation

## Ablations

- [x] No dropout
- [x] Fixed dropout mask 0.10
- [x] Standard dropout 0.20
- [x] Temperature 0.01
- [x] Temperature 1.00

## Representation geometry

- [x] Raw BERT alignment/uniformity
- [x] Main 50K SimCSE alignment/uniformity
- [x] Dropout ablation geometry
- [x] Temperature ablation geometry
- [x] Geometry/STS-B comparison

## Not completed

- [ ] Full 1M-sentence training
- [ ] Full paper-scale benchmark reproduction
- [ ] Final discrepancy analysis
- [ ] Final research report

---

# 46. Final Experimental Table

| Experiment | Sentences | Effective examples | Steps | STS-B Spearman | Alignment | Uniformity | Runtime | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Raw BERT baseline | 0 | 0 | 0 | **30.18%** | **0.1771** | **-1.0560** | — | Complete |
| Local 10K SimCSE | 10,000 | 9,984 | 156 | **53.55%** | — | — | **~55–70 min** | Complete |
| Local 25K SimCSE | 25,000 | 24,960 | 390 | **62.59%** | — | — | **2 h 50 min** | Complete |
| Local 50K SimCSE | 50,000 | 49,984 | 781 | **69.07%** | **0.2924** | **-2.5705** | **5 h 45 min** | Complete |
| No dropout | 50,000 | 49,984 | 781 | **60.21%** | **0.4486** | **-2.7144** | **4 h 10 min** | Complete |
| Fixed mask 0.10 | 50,000 | 49,984 | 781 | **41.07%** | **0.4057** | **-2.3718** | **5 h 15 min** | Complete |
| Standard dropout 0.20 | 50,000 | 49,984 | 781 | **76.38%** | **0.1946** | **-2.3814** | **5 h 52 min** | Complete |
| Temperature 0.01 | 50,000 | 49,984 | 781 | **68.88%** | **0.1642** | **-1.5738** | **5 h 50 min** | Complete |
| Temperature 1.00 | 50,000 | 49,984 | 781 | **58.33%** | **0.1030** | **-1.9329** | **5 h 48 min** | Complete |
| Official pretrained SimCSE | — | — | — | **82.12%** | — | — | — | Complete |
| Paper/project reference | ~1M | — | 15,625 | **76.25 reported** | — | — | — | Reference |

---

# 47. Current Research Position

The project can now be described as:

> **A validated implementation-level reproduction of unsupervised SimCSE with
> BERT-base-uncased, including deterministic checkpoint/resume behavior,
> evaluation against the official pretrained SimCSE model, a controlled
> resource-constrained training-data scaling study, completed mechanism
> ablations, and representation-geometry analysis through alignment and
> uniformity.**

The strongest overall empirical findings are:

```text
1. More training data improved STS-B from 53.55% → 62.59% → 69.07%
   across 10K → 25K → 50K.

2. The marginal gain decreased across the tested data increments.

3. Independent stochastic dropout is critical:
   no-dropout and fixed-mask variants performed substantially worse.

4. Standard dropout 0.20 produced the strongest tested 50K result:
   76.38%.

5. Alignment and uniformity move differently across conditions.

6. Uniformity improvement alone is insufficient:
   no-dropout achieved excellent uniformity but poor STS-B.

7. Alignment alone is insufficient:
   temperature 1.00 achieved the best alignment but only 58.33% STS-B.

8. The best results are associated with a useful alignment–uniformity balance.

9. The 1M-sentence paper-scale run remains computationally impractical on the
   available CPU hardware.

10. Therefore, the project is a validated reduced-data reproduction and
    mechanism study, not a full 1M-sentence benchmark reproduction.
```

The appropriate overall claim remains:

> **We reproduced and validated the core unsupervised SimCSE training
> implementation and characterized its learning behavior, mechanism-level
> ablations, and representation geometry under a resource-constrained
> reduced-data training regime.**

---

# 48. Next Phase — Discrepancy Analysis and Final Scientific Synthesis

**Status: READY TO BEGIN**

The expensive experimental work is complete. The next task is to turn the
completed measurements into a coherent scientific explanation.

The workflow is now:

```text
Completed experiments
        ↓
Discrepancy analysis
        ↓
Geometry + performance synthesis
        ↓
Robust conclusions
        ↓
Limitations
        ↓
Final figures/tables
        ↓
Final research report
```

No additional large training run should be started unless the discrepancy
analysis identifies a specific, testable unresolved issue that cannot be
answered from the completed evidence.
# 50. Phase A.4 — Discrepancy Analysis: COMPLETE

The completed experiments and the official Princeton-NLP materials were
cross-checked. Several important comparison issues are now resolved.

## 50.1 The 76.25 reference was being compared to the wrong metric

This is the most important correction.

The official SimCSE repository reports **76.25 as Avg. STS**, not as STS-B alone.
The published seven-task table gives:

```text
STS12   68.40
STS13   82.41
STS14   74.38
STS15   80.91
STS16   78.56
STS-B   76.85
SICK-R  72.23
Avg.    76.25
```

Therefore:

> **76.25 must not be treated as an STS-B reference score.**

The local project currently evaluates the GLUE STS-B **validation** split only.
Consequently, the following earlier statements are not directly comparable:

```text
Local 50K STS-B validation: 69.07%
Paper/reference:            76.25%
```

and:

```text
Local dropout 0.20 validation: 76.38%
Paper/reference:               76.25%
```

The second comparison must especially **not** be described as the ablation
"beating the paper result."

The correct external STS-B value from the published seven-task table is:

```text
Official SimCSE-BERTbase STS-B: 76.85
Official SimCSE-BERTbase Avg STS: 76.25
```

These are different evaluation quantities.

## 50.2 The locally measured 82.12% official-checkpoint score is also not a direct
paper comparison

The project obtains:

```text
Official pretrained checkpoint on local GLUE STS-B validation: 82.12%
```

This is a useful **pipeline reference**, because the same local evaluator can
measure the local checkpoints against it.

It should not, however, be presented as equivalent to the official 76.25
Avg. STS number.

The project therefore has three distinct reference quantities:

| Reference | Metric/protocol | Score | Use |
|---|---|---:|---|
| Published SimCSE result | Avg. across 7 STS tasks | **76.25** | External paper reference |
| Published SimCSE result | STS-B | **76.85** | External STS-B reference |
| Local official checkpoint | GLUE STS-B validation | **82.12%** | Local pipeline/reference model |

This distinction is now mandatory in the final report.

## 50.3 Training-time model selection differs from the local reduced-data runs

The official example enables:

```text
evaluation_strategy = steps
eval_steps = 125
metric_for_best_model = stsb_spearman
load_best_model_at_end = true
```

and the official trainer explicitly saves the best-validation checkpoint and
reloads the best checkpoint for final evaluation.

The local reduced-data trainer instead runs the epoch and evaluates the
resulting final epoch checkpoint. It does not perform periodic STS-B
validation/model selection during these reduced-data experiments.

Therefore the local 10K/25K/50K results are not exact reproductions of the
official checkpoint-selection procedure.

This is a real methodological difference and is more important than a mere
logging difference.

## 50.4 Training-data scale remains the largest intentional deviation

The official unsupervised setup uses approximately:

```text
1,000,000 Wikipedia sentences
15,625 optimizer steps
batch size 64
one epoch
```

The local primary experiment uses:

```text
50,000 Wikipedia sentences
781 optimizer steps
batch size 64
one epoch
```

Thus the main baseline sees only about **5% of the intended training
sentences**.

This remains the principal deliberate resource constraint.

The completed scaling curve is therefore scientifically useful precisely
because it measures what happens as the available training data increases
within the CPU-feasible regime.

## 50.5 Hardware and numerical environment are additional differences

The official repository states that its reported results were obtained with
NVIDIA 3090 GPUs and CUDA 11, and notes that device/software differences can
produce some performance variation.

The local project trains on CPU and uses the local PyTorch/Transformers
environment.

This can contribute to numerical differences, but it should **not** be used as
the primary explanation for the large gap between the 50K baseline and the
published result. The dominant known differences are training-data scale and
checkpoint-selection/evaluation protocol.

## 50.6 The implementation is algorithmically close to the official method

The source comparison supports the following equivalences:

```text
BERT-base-uncased
batch size 64
learning rate 3e-5
temperature 0.05
max length 32
CLS pooler
MLP during training only
raw CLS at evaluation
same-sentence two-view construction
independent dropout in the normal SimCSE path
in-batch contrastive negatives
```

The official implementation likewise flattens the two views, encodes them
together, applies the MLP to the CLS representation, and uses cosine
similarity divided by temperature. citeturn3view0turn2view0

The local source implements the same conceptual path. The fixed-mask ablation
is deliberately different: it restores RNG state between the two forwards so
the two identical views receive the same stochastic masks. That is a mechanism
ablation, not the standard SimCSE training path.

## 50.7 The ablation results are stronger evidence than the raw loss values

The completed geometry results explain why training loss alone is not an
adequate interpretation.

```text
Condition             STS-B     Alignment     Uniformity
---------------------------------------------------------
Raw BERT              30.18%     0.1771       -1.0560
Main SimCSE 50K       69.07%     0.2924       -2.5705
No dropout            60.21%     0.4486       -2.7144
Fixed mask 0.10       41.07%     0.4057       -2.3718
Dropout 0.20          76.38%     0.1946       -2.3814
Temp 0.01             68.88%     0.1642       -1.5738
Temp 1.00             58.33%     0.1030       -1.9329
```

The pattern is not monotonic in either geometry metric:

- **No dropout** has the best uniformity but poor alignment and only 60.21%
  STS-B.
- **Temperature 1.00** has the best alignment but only 58.33% STS-B.
- **Dropout 0.20** combines relatively strong alignment with substantially
  improved uniformity and obtains the strongest local STS-B result.

Therefore the most defensible interpretation is a **balance hypothesis**:
downstream sentence-similarity performance is associated with achieving both
useful positive-pair alignment and useful global representation uniformity.

This is an interpretation supported by the completed experiments, not a causal
proof.

## 50.8 A subtle geometry-methodology limitation is now documented

The local uniformity implementation:

```text
- normalizes embeddings
- evaluates unordered pairs
- excludes self-pairs
- uses all unique sentences from the local STS-B validation split
```

This is a sensible and numerically stable implementation of the uniformity
quantity, but it is a specific empirical estimator.

The final report should therefore call these:

> **local alignment/uniformity measurements under the project's stated
> estimator and validation-set protocol**

rather than implying they are numerically identical to every implementation
used in the original SimCSE analysis.

The alignment definition is directly based on normalized positive-pair
distances, while the local uniformity computation uses the corresponding
hyperspherical exponential-distance formulation.

## 50.9 What the discrepancy analysis now supports

The evidence supports the following hierarchy:

### Established

1. The core implementation is validated.
2. The reduced-data SimCSE procedure learns substantially better STS-B
   representations than raw BERT.
3. More training data improved performance across 10K → 25K → 50K.
4. Independent dropout is a critical mechanism.
5. The tested dropout and temperature settings materially affect performance.
6. The representation geometry changes substantially under SimCSE training.
7. Alignment and uniformity jointly provide a more informative interpretation
   than either metric alone.

### Strong but scoped interpretation

1. The observed 10K–50K curve exhibits diminishing marginal gains.
2. The best local ablation suggests that stronger independent dropout can be
   beneficial in the reduced-data regime.
3. The geometry results support an alignment–uniformity balance interpretation.

### Not established

1. That 50K training reproduces the original 1M-sentence benchmark.
2. That 76.38% local validation performance exceeds the published 76.25
   SimCSE result.
3. That 82.12% local validation performance is directly comparable to 76.25.
4. A universal data-scaling law.
5. A causal relationship between a single geometry metric and STS-B.

---

# 51. Corrected Reference Framework for the Final Report

The final report should use the following comparison structure.

## A. Internal reduced-data comparison

```text
Raw BERT validation        30.18%
10K SimCSE validation      53.55%
25K SimCSE validation      62.59%
50K SimCSE validation      69.07%
```

This is the primary controlled scaling experiment.

## B. Internal mechanism comparison

```text
No dropout                60.21%
Fixed mask 0.10           41.07%
Standard dropout 0.20     76.38%
Temperature 0.01         68.88%
Temperature 1.00         58.33%
```

This is the primary ablation experiment.

## C. Local pipeline reference

```text
Official pretrained SimCSE checkpoint
GLUE STS-B validation: 82.12%
```

This establishes that the local evaluation pipeline can produce a strong
result from the released checkpoint.

## D. External paper/reference comparison

```text
Published SimCSE-BERTbase STS-B:  76.85
Published SimCSE-BERTbase Avg:    76.25
```

These should remain explicitly labeled as external reference values.

---

# 52. Phase A.4 Conclusion

The discrepancy analysis is now complete.

The apparent contradictions in the earlier notes are largely explained by
**evaluation-metric/protocol mismatch**, not by an unexplained failure of the
implementation.

The most important correction is:

> **76.25 is the published seven-task Avg. STS score, not the STS-B score.**

The second important correction is:

> **82.12% is a local GLUE STS-B validation result for the released official
> checkpoint and is therefore a local pipeline reference, not a reproduction
> of the published Avg. STS metric.**

The third is:

> **The local reduced-data runs did not use the official periodic
> validation/best-checkpoint selection procedure.**

With these distinctions enforced, the experimental story is internally
consistent and scientifically defensible.

---

# 53. Phase A.5 — Final Scientific Synthesis: NEXT

The next and final analytical phase is to construct the report-level evidence:

```text
Research question
      ↓
Implementation validation
      ↓
Reduced-data scaling
      ↓
Controlled ablations
      ↓
Alignment/uniformity geometry
      ↓
Discrepancy analysis
      ↓
Unified scientific interpretation
      ↓
Figures + final tables
      ↓
Limitations
      ↓
Final report
```

No additional large training run is currently justified.

The remaining work is **scientific synthesis and presentation**, not further
experimental expansion.
