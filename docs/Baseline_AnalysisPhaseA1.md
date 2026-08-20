# Training-Data Scaling Analysis

## 1. Final experimental points

| Condition                  | Training sentences | Effective examples |  Steps |      STS-B Spearman |
| -------------------------- | -----------------: | -----------------: | -----: | ------------------: |
| Raw BERT CLS               |                  0 |                  0 |      0 |          **30.18%** |
| SimCSE 10K                 |             10,000 |              9,984 |    156 |          **53.55%** |
| SimCSE 25K                 |             25,000 |             24,960 |    390 |          **62.59%** |
| SimCSE 50K                 |             50,000 |             49,984 |    781 |          **69.07%** |
| Official pretrained SimCSE |                  — |                  — |      — |          **82.12%** |
| Paper/reference            |                ~1M |                  — | 15,625 | **76.25% reported** |

The 50K result is directly confirmed by the completed training/evaluation log: 781 steps, full 1,500-example STS-B validation, raw CLS evaluation, and Spearman 0.690704.

## 2. The central result: performance keeps increasing

Raw BERT
30.18%
   │
   │ +23.37 pp
   ▼
10K
53.55%
   │
   │ +9.04 pp
   ▼
25K
62.59%
   │
   │ +6.48 pp
   ▼
50K
69.07%

So there is no evidence of saturation by 50K.
Every increase in training data produced a higher STS-B score.

But the important second observation is that the gains are getting smaller.

Marginal improvement

10K → 25K

62.59−53.55=+9.04 pp

25K → 50K

69.07−62.59=+6.48 pp

Therefore the marginal improvement decreased by:

9.04−6.48=2.56 pp

The second increment is only about:

6.48/9.04 ≈ 71.7%
of the previous increment.

That is a fairly clear diminishing-return pattern.
The earlier project notes specifically identified this as the question the 50K experiment was supposed to answer.

## 3. But there is an even stronger way to see it

The amount of additional training data is not the same in the two comparisons.

10K → 25K

Additional data:
25,000−10,000=15,000

Performance gain:
+9.04 pp

Gain per 1K additional sentences:
9.04/15=0.603 pp

25K → 50K

Additional data:
50,000−25,000=25,000

Performance gain:
+6.48 pp

Gain per 1K:
6.48/25=0.259 pp

So the marginal efficiency falls from approximately:
0.603 → 0.259 STS points per 1K sentences

That's roughly a 57% reduction in marginal gain per additional 1K sentences.
This is stronger evidence for diminishing returns than simply looking at the two vertical differences.

## 4. Improvement over vanilla BERT

This is another important result.

10K
53.55−30.18=+23.37 pp
25K
62.59−30.18=+32.41 pp
50K
69.07−30.18=+38.89 pp

So by 50K, the reduced-data SimCSE model has gained 38.89 percentage points over the raw BERT CLS baseline.

The project therefore demonstrates something stronger than merely "SimCSE works":
Increasing the amount of unsupervised training data continues to improve the learned sentence representation, even within the severely reduced CPU-feasible regime.

## 5. How close are we to the references?

There are two different reference points, and we should not conflate them.
Against the paper/reference value

50K:
76.25−69.07=7.18 pp
So we're 7.18 percentage points below the reported 76.25 reference.

Against the locally evaluated official pretrained checkpoint
82.12−69.07=13.05 pp

So we're 13.05 percentage points below the official pretrained model evaluated through our pipeline.
The project notes explicitly distinguish the reported paper value from the locally evaluated official pretrained reference.
That distinction should remain in the final report.

## 6. The most interesting interpretation

STS-B
70% |                              ● 50K 69.07
    |                         /
65% |                    ● 25K 62.59
    |                  /
60% |                /
    |              /
55% |        ● 10K 53.55
    |       /
50% |      /
    |     /
45% |    /
    |   /
40% |  /
    | /
35% |
    |
30% | ● Raw BERT 30.18
    +--------------------------------
       0     10K      25K       50K

The curve is monotonically increasing but concave downward over the observed range.

That is exactly the pattern we'd expect if additional data continues to help while each additional block of data contributes progressively less.

However, I would not write:
"We have proven that SimCSE follows a diminishing-returns law."

That would be too strong.

We have only three trained data points. The scientifically defensible statement is:
The observed 10K–25K–50K results exhibit diminishing marginal improvement in STS-B performance, with the gain falling from +9.04 percentage points to +6.48 percentage points as training data increases.

That's supported directly by the experiment.

## 7. Training dynamics also support the story

Interestingly, the training metrics themselves become more stable as the dataset gets larger.

| Dataset | Avg loss | Avg positive cosine | Avg negative cosine |
| ------- | -------: | ------------------: | ------------------: |
| 10K     | 0.015423 |            0.797257 |            0.096789 |
| 25K     | 0.004820 |            0.791352 |            0.068614 |
| 50K     | 0.003028 |            0.791234 |            0.051675 |

The 50K run ended with average positive cosine essentially unchanged from 25K:
0.791234≈0.791352

while average negative cosine decreased:
0.068614→0.051675

and the loss decreased:
0.004820→0.003028

The 50K training log confirms these final averages.
This is interesting because it suggests that the improvement in downstream STS-B isn't coming from simply pushing positive-pair cosine higher and higher.
Instead, the representation geometry appears to be becoming more discriminative: positive similarity remains high while negative similarity becomes lower.
We should be cautious here: these are batch-level training diagnostics, not the paper's formal alignment/uniformity metrics. So this is a hypothesis/interpretation, not yet the final geometric explanation.
And that naturally leads into the later alignment/uniformity phase.

## 8. Resource efficiency

This is where the 1M experiment becomes scientifically useful rather than merely a failed attempt.

The original 1M run was attempted at:
1,000,000 sentences
batch size 64
15,625 steps
1 epoch
CPU

It reached approximately step 196 after ~1.5 hours before being stopped. The project notes conclude that a full 1M CPU run would take multiple days and is computationally impractical on the available hardware.
Meanwhile, the reduced experiments produced:
10K → 53.55%
25K → 62.59%
50K → 69.07%

Thus, rather than pretending to reproduce the 1M benchmark, we have a legitimate resource-constrained scaling experiment.
That is actually a much cleaner scientific framing.

## Finding

Increasing unsupervised SimCSE training data from 9,984 to 49,984 effective examples produced a consistent improvement in STS-B performance, from 53.55% to 69.07%. However, the marginal gain diminished from +9.04 percentage points between 10K and 25K to +6.48 percentage points between 25K and 50K, indicating diminishing returns within the observed reduced-data regime.

Evidence
Raw BERT       30.18%
10K            53.55%   +23.37
25K            62.59%   +9.04
50K            69.07%   +6.48
Scientific interpretation

SimCSE is still learning at 50K.
It has not saturated.
But data efficiency is declining.

And importantly:
50K is still 7.18 points below the reported paper reference and 13.05 points below the locally evaluated official pretrained model.
So the reduced-data curve has not reached the reference performance yet.
