(simcse-reproduction) D:\CODING\Paper Reproduction\simcse-reproduction>python src/analyze_all_geometry.py

Device: cpu
Experiments root: experiments
STS-B split: validation
Alignment positives: STS-B score > 4.0

========================================================================
[1/7] raw_BERT
STS-B=0.301829 | alignment=0.177113 | uniformity=-1.056028 | positive_pairs=208

========================================================================
[2/7] SimCSE_50K
Checkpoint: experiments\reduced_50k\20260818_121443\checkpoint\epoch_1\training_state.pt
STS-B=0.690704 | alignment=0.292355 | uniformity=-2.570531 | positive_pairs=208

========================================================================
[3/7] no_dropout
Checkpoint: experiments\dropout_no_dropout_50k\20260818_235440\checkpoint\epoch_1\training_state.pt
STS-B=0.602082 | alignment=0.448588 | uniformity=-2.714421 | positive_pairs=208

========================================================================
[4/7] fixed_mask_0.10
Checkpoint: experiments\dropout_fixed_mask_50k\20260819_055050\checkpoint\epoch_1\training_state.pt
STS-B=0.410671 | alignment=0.405743 | uniformity=-2.371783 | positive_pairs=208

========================================================================
[5/7] dropout_0.20
Checkpoint: experiments\dropout_020_50k\20260819_153016\checkpoint\epoch_1\training_state.pt
STS-B=0.763789 | alignment=0.194571 | uniformity=-2.381358 | positive_pairs=208

========================================================================
[6/7] temperature_0.01
Checkpoint: experiments\temperature_001_50k\20260819_214835\checkpoint\epoch_1\training_state.pt
STS-B=0.688782 | alignment=0.164158 | uniformity=-1.573842 | positive_pairs=208

========================================================================
[7/7] temperature_1.00
Checkpoint: experiments\temperature_100_50k\20260820_042041\checkpoint\epoch_1\training_state.pt
STS-B=0.583291 | alignment=0.103016 | uniformity=-1.932856 | positive_pairs=208

========================================================================
Saved combined results: analysis\phase_a3\geometry_results.csv
