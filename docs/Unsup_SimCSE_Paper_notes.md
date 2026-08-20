# UnSup SimCSE Paper Notes

> Based on **SimCSE: Simple Contrastive Learning of Sentence Embeddings** by Tianyu Gao, Xingcheng Yao, and Danqi Chen.

## Problem

### What is a sentence embedding?
**Ans.** A sentence embedding is a fixed-size vector representation of a whole sentence. In SimCSE, the encoder maps a sentence $x$ to a representation:

$$
h = f_\theta(x)
$$

The vector is intended to capture the sentence's semantic meaning so that semantically similar sentences have similar embeddings.

### Why isn't vanilla pretrained BERT automatically a good sentence encoder?
**Ans.** Vanilla BERT is pretrained mainly with token-level language-modeling objectives, not directly with an objective that makes whole-sentence embeddings useful for semantic similarity. Its sentence representations can also be **anisotropic**: many embeddings occupy a relatively narrow region/cone of the vector space instead of being well spread out. This makes cosine similarities less informative for distinguishing different semantic relationships.

SimCSE fine-tunes the pretrained encoder with a contrastive objective to improve both:
- **alignment**: semantically related sentences become close;
- **uniformity**: embeddings are better spread across the representation space.

### What problem is SimCSE trying to solve?
**Ans.** SimCSE tries to learn better **sentence embeddings**, especially embeddings that work well for semantic textual similarity (STS). It does this by training a pretrained Transformer with a contrastive objective that pulls positive sentence pairs together and pushes negative examples apart.

---

## Previous approaches

### What kinds of data augmentation were previously used?
**Ans.** Previous NLP contrastive approaches used discrete transformations such as:
- word deletion,
- word reordering,
- word substitution/replacement,
- synonym replacement,
- cropping or selecting part of the text,
- MLM-based token replacement.

SimCSE shows that simply using standard dropout noise on the hidden representations can work better than these discrete augmentations.

### Why are discrete NLP augmentations problematic?
**Ans.** Text is **discrete**, so changing the words can easily change the sentence's meaning. For example, deleting or replacing a word may turn a sentence into something that is no longer semantically equivalent to the original. This makes it difficult to construct reliable positive pairs using discrete text transformations.

---

## Main idea

### What is the positive pair?
**Ans.** In **unsupervised SimCSE**, the positive pair is the **same sentence twice**:

$$
x_i^+ = x_i
$$

The sentence is passed through the **same encoder twice**, but with **independently sampled dropout masks**. Therefore, the two hidden representations are slightly different even though the input text is identical.

So:

$$
h_i^{(1)} = f_\theta(x_i, z_i)
$$

$$
h_i^{(2)} = f_\theta(x_i, z'_i)
$$

where $z_i$ and $z'_i$ are different random dropout masks.

### What is the negative pair?
**Ans.** In unsupervised SimCSE, the negatives are the **other sentences in the same mini-batch**.

For a batch of $N$ sentences, the representation of sentence $i$'s first view is trained to identify its matching second view $i$ among the $N$ candidate second views.

So there are $N-1$ off-diagonal in-batch negatives for each anchor.

**Important:** these other sentences are treated as negatives by the training objective; they are not guaranteed to be semantically unrelated in reality.

In **supervised SimCSE**, NLI contradiction pairs are additionally used as **hard negatives**.

### Where does randomness come from?
**Ans.** The randomness comes from **standard dropout inside the Transformer**. During the two forward passes, independently sampled dropout masks are applied to intermediate representations, including dropout in fully connected layers and attention probabilities.

The paper does **not** add a special SimCSE-specific noise mechanism; it uses the Transformer's standard dropout.

### What does dropout do?
**Ans.** Dropout randomly masks some intermediate activations according to a Bernoulli mask, creating slightly different hidden representations on different forward passes.

It is better to say:

> Dropout creates two slightly different **hidden representations** of the same sentence.

It does **not** directly modify the input sentence $x$, and it is not best described as simply "masking points in the input sentence vector."

During backpropagation, the contrastive loss updates the trainable parameters $\theta$ of the pretrained Transformer (and the relevant pooling/MLP components). The dropout mask is sampled again on later forward passes.

### Why is only one encoder required?
**Ans.** Because the two positive views are the **same sentence**, SimCSE does not need two independently parameterized encoders.

The same encoder $f_\theta$ processes both copies:

$$
x_i \rightarrow f_\theta(x_i,z_i)
$$

$$
x_i \rightarrow f_\theta(x_i,z'_i)
$$

Only the dropout masks differ. The parameters are shared.

This is different from earlier contrastive setups where $x_i$ and $x_i^+$ were genuinely different texts and therefore often used dual encoders.

---

## Loss

### What is InfoNCE?
**Ans.** InfoNCE is a contrastive-learning objective that makes the model identify the correct positive example among a set of candidates.

For SimCSE, for an anchor $h_i$, the loss is:

$$
\ell_i
=
-\log
\frac{
\exp\!\left(\operatorname{sim}(h_i,h_i^+)/\tau\right)
}{
\displaystyle\sum_{j=1}^{N}
\exp\!\left(\operatorname{sim}(h_i,h_j^+)/\tau\right)
}.
$$

The numerator is the similarity to the correct positive. The denominator contains the positive plus the other in-batch candidates.

Intuitively:

> **Make the correct positive score high relative to all the other candidates.**

This is equivalent to a cross-entropy classification problem where the correct matching sentence has the target label.

### What is temperature?
**Ans.** Temperature $\tau$ controls how sharply similarity differences are converted into the softmax distribution.

The logits are:

$$
\frac{\operatorname{sim}(h_i,h_j)}{\tau}
$$

- **Small $\tau$** → larger differences between similarities → sharper softmax.
- **Large $\tau$** → smaller differences → softer/flatter softmax.

The paper reports that a tuned temperature of **0.05** worked well for cosine similarity in its BERT-base experiments.

### Why does cosine similarity appear?
**Ans.** SimCSE uses cosine similarity:

$$
\operatorname{sim}(h_1,h_2)
=
\frac{h_1^\top h_2}{\lVert h_1\rVert\,\lVert h_2\rVert}
$$

It measures the angle/direction between vectors:
- $1$: same direction;
- $0$: orthogonal;
- $-1$: opposite directions.

Cosine similarity is useful because sentence embedding quality is commonly evaluated by semantic similarity using cosine similarity, and normalization removes the effect of vector magnitude.

More precisely, the paper's alignment/uniformity analysis is **not the same thing as cosine similarity**:
- the contrastive loss uses cosine similarity;
- **alignment** measures how close positive pairs are;
- **uniformity** measures how well the overall embeddings are spread out.

The paper also experimentally compares cosine similarity with dot product and finds cosine similarity better with a suitable temperature.

---

## Expected result

### What does the paper claim?
**Ans.** The paper claims that SimCSE substantially improves sentence embeddings on standard semantic textual similarity (STS) tasks.

Using **BERT-base**:
- **Unsupervised SimCSE:** about **76.3%** average Spearman correlation.
- **Supervised SimCSE:** about **81.6%** average Spearman correlation.

The paper describes these as improvements of about **4.2** and **2.2 percentage points** over the previous best results, respectively.

In the detailed Table 5, the corresponding averages are **76.25** for unsupervised SimCSE-BERT-base and **81.57** for supervised SimCSE-BERT-base.

The main conclusion is that a surprisingly simple recipe—**same sentence + two independent dropout masks + contrastive learning**—can produce very strong sentence embeddings.

### What is the key reason SimCSE works?
**Ans.** The paper argues that pretrained representations already have reasonably good alignment, but their embedding space is often highly anisotropic. Contrastive learning improves the **uniformity** of the space by pushing negative examples apart, while dropout provides enough variation between the two positive views to avoid representation collapse.

Supervised NLI pairs then provide stronger semantic supervision and further improve **alignment** between positive pairs.

---

## Dropout and backpropagation

> Dropout applies random 0/1 masks to **intermediate activations inside the Transformer** during training. For the same sentence, two forward passes use independently sampled dropout masks, so they produce two slightly different sentence representations. The contrastive loss compares these representations and backpropagates through the shared encoder, fine-tuning its trainable parameters. A new dropout mask is sampled on subsequent forward passes.

So the important pipeline is:

$$
x
\rightarrow
\text{Transformer + dropout}
\rightarrow
h^{(1)}
$$

and independently:

$$
x
\rightarrow
\text{same Transformer + different dropout}
\rightarrow
h^{(2)}
$$

then:

$$
(h^{(1)},h^{(2)})
\rightarrow
\text{cosine similarities}
\rightarrow
\text{InfoNCE / cross-entropy loss}
\rightarrow
\text{backpropagation}
\rightarrow
\theta \text{ updated}
$$

**Do not say:** "dropout masks points in $x_i$."  
**Better:** "dropout masks intermediate hidden activations."

---

## Lessons

1. A paper, released code, released checkpoint, and current repository are related artifacts, but they are not automatically identical artifacts.

2. **Conceptually:** SimCSE performs two stochastic forward passes of the same sentence.

3. **Implementation:** duplicated sentences can be represented as a larger flattened batch and passed through the shared encoder; each copy receives its own dropout randomness.

4. There is **one shared encoder**, not two independently parameterized encoders.

5. The positive pair in **unsupervised SimCSE is identical at the text level**. The variation comes from different dropout masks inside the network.

6. The negative examples come from the other examples in the mini-batch. For batch size $N$, each anchor has $N-1$ off-diagonal in-batch negatives.

7. In-batch negatives are computationally attractive because the model does not need to separately construct or encode a huge set of negative sentences.

8. In supervised SimCSE, NLI **entailment pairs** are positives and **contradiction pairs** are hard negatives.

9. Dropout is not merely a way to randomly delete words. It is noise applied to **hidden representations** inside the Transformer.

10. Removing dropout, or using the same dropout mask for both views, makes the two positive embeddings identical and causes severe performance degradation / representation collapse.

11. Contrastive learning improves the **uniformity** of the embedding space by pushing negative examples apart.

12. Supervised NLI training provides additional semantic information and improves **alignment** of positive pairs.

13. The paper connects this to BERT's **anisotropy problem**: pretrained sentence embeddings tend to occupy a narrow region of the vector space.

14. SimCSE fine-tunes the pretrained encoder parameters; it is not just a post-processing method applied after BERT.

---

# The original `train.py` in 15 lines

> These implementation notes are kept at the conceptual level. The paper itself does not contain the full repository source code, so exact code-level details should be verified against the corresponding source version.

1. Parse model, data, and training arguments.
2. Load the training text dataset.
3. Load the pretrained BERT/RoBERTa configuration.
4. Load the corresponding tokenizer.
5. Load SimCSE's contrastive-learning model (`BertForCL` or `RobertaForCL`).
6. For the unsupervised one-column setup, use the same sentence as both sentence views.
7. Therefore each example becomes effectively $[x,x]$.
8. Tokenize both copies.
9. Represent the examples as a batch with two sentence views.
10. Pad the batch as needed before it is passed to the model.
11. Construct the contrastive-learning trainer.
12. The trainer feeds batches to the SimCSE model.
13. The model computes the contrastive loss using the two views.
14. The trainer performs optimization, evaluation, and checkpointing.
15. Training therefore fine-tunes the pretrained Transformer using the SimCSE objective.

---

# The original `trainers.py` in 15 lines

1. Extend Hugging Face's `Trainer` with SimCSE-specific evaluation, checkpointing, and training behavior through `CLTrainer`.
2. Replace normal evaluation with SentEval-based evaluation using the model's sentence embeddings.
3. SentEval's `batcher` converts tokenized sentences back into strings, tokenizes them with the SimCSE tokenizer, and moves the inputs to the training device.
4. Run the model with `sent_emb=True` and `torch.no_grad()` to obtain sentence embeddings without computing gradients.
5. Evaluate STSBenchmark and SICKRelatedness using Spearman correlation and average the two scores into `eval_avg_sts`.
6. Optionally evaluate the seven transfer-classification tasks and compute `eval_avg_transfer`.
7. Log evaluation metrics so the training control logic can use them for model selection.
8. Override checkpoint saving so that, when a best-model metric is configured, the best-validation model can be preserved according to the SimCSE training logic.
9. Save the best model together with optimizer, scheduler, and trainer state so training can be resumed correctly.
10. Otherwise fall back to Hugging Face's normal checkpoint naming, saving, state management, and checkpoint rotation behavior.
11. Override `train()` to construct the dataloader, determine optimization steps, and create the optimizer and learning-rate scheduler.
12. Restore optimizer, scheduler, and trainer state when training resumes from a checkpoint, including skipping already-processed batches where required.
13. Run the epoch/batch loop with gradient accumulation, optional distributed synchronization control, gradient clipping, optimizer updates, and learning-rate scheduling.
14. Periodically call Hugging Face callbacks and `_maybe_log_save_evaluate()` so training can log metrics, evaluate with SentEval, and save checkpoints.
15. At the end, reload the best checkpoint when requested, log final training statistics, trigger training-end callbacks, and return a `TrainOutput`.

---

# The original `models.py` in 15 lines

1. Define the MLP head that transforms BERT/RoBERTa's `[CLS]` representation with a linear layer and `tanh`.
2. Define cosine similarity divided by temperature to produce contrastive-learning logits.
3. Define sentence pooling strategies such as `cls`, `cls_before_pooler`, `avg`, `avg_top2`, and `avg_first_last`.
4. Initialize the SimCSE components: selected pooler, optional MLP, and temperature-scaled similarity function.
5. In `cl_forward`, receive a batch conceptually shaped as `[batch, num_sent, sequence_length]`.
6. Flatten the sentence-view dimension so the Transformer receives `[batch × num_sent, sequence_length]`.
7. Encode all views with the **same shared pretrained Transformer**, with dropout providing different stochastic views during training.
8. Pool the Transformer outputs into one sentence representation per input sentence.
9. Reshape the representations back to `[batch, num_sent, hidden_size]` and separate them into `z1`, `z2`, and optionally `z3`.
10. During distributed training, gather sentence representations across GPUs so examples from other GPUs can also act as in-batch negatives.
11. Compute pairwise `z1`–`z2` cosine similarities, producing a similarity matrix.
12. Divide similarities by temperature and use diagonal labels so each `z1` must identify its matching `z2`.
13. For supervised SimCSE, compare the anchor with hard-negative contradiction representations and apply the configured hard-negative weight.
14. If MLM is enabled, compute the masked-language-model loss and combine it with the contrastive loss using `mlm_weight`.
15. `BertForCL` and `RobertaForCL` expose the common logic through `forward()`; `sent_emb=True` switches to sentence-embedding inference behavior rather than contrastive-loss training.

---

# The original `evaluation.py` in 15 lines

1. Parse the model checkpoint, pooling strategy, evaluation mode, and task-selection arguments.
2. Load the Hugging Face model and tokenizer from `model_name_or_path`, then move the model to GPU if available or CPU otherwise.
3. Expand `task_set` into the corresponding STS tasks, transfer-classification tasks, or the full combination.
4. Configure SentEval's fast/dev or full/test evaluation settings, including its classifier, cross-validation folds, optimizer, and batch size.
5. Define an empty `prepare()` hook because this evaluation does not require additional task-specific preprocessing.
6. Define `batcher()` as the bridge between SentEval's tokenized sentences and the Hugging Face Transformer model.
7. Convert rare byte-encoded tokens to UTF-8 strings, join each tokenized sentence back into text, and tokenize the sentences again with the Hugging Face tokenizer.
8. Pad the batch, optionally truncate to `max_length`, move tokenizer outputs to the model's device, and run the model under `torch.no_grad()`.
9. Extract the hidden representations needed by the requested pooling strategy.
10. For `cls`, return the BERT pooled `[CLS]` representation; for `cls_before_pooler`, return the raw final-layer `[CLS]` representation used by unsupervised SimCSE at test time.
11. For `avg`, average final-layer token representations using the attention mask; for `avg_first_last` and `avg_top2`, average the relevant Transformer layers before masked token averaging.
12. Create a SentEval engine for each requested task, let SentEval call `batcher()` to obtain sentence embeddings, and store the task results.
13. In dev mode, report STS-B and SICK-R development Spearman scores and optionally report the seven transfer-task development accuracies and their average.
14. In test/fasttest mode, report STS12–16 using SentEval's `"all"` Spearman aggregation, STS-B and SICK-R using test Spearman correlation, and average the seven STS scores.
15. Also report the seven transfer-task test accuracies and their average, while `print_table()` formats the collected scores into readable tables.

---

## One mental model to remember

**Unsupervised SimCSE:**

```text
same sentence x
       │
       ├───────────────┐
       │               │
 dropout mask z₁   dropout mask z₂
       │               │
       ▼               ▼
      z₁_repr         z₂_repr
       │               │
       └──── positive ─┘
                │
        compare against
       other batch views
                │
                ▼
          InfoNCE loss
                │
                ▼
       backpropagate into
        shared BERT θ
```

### In one sentence

> **SimCSE turns the same sentence into two slightly different hidden representations using independent dropout masks, then trains the shared Transformer to recognize those two representations as a positive pair while treating other sentences in the batch as negatives.**
