(simcse-reproduction) D:\CODING\Paper Reproduction\simcse-reproduction>python src/train.py --max-sentences 25000 --experiment-name reduced_25k

======================================================================
SIMCSE TRAINING
======================================================================
Experiment directory: experiments\reduced_25k\20260817_170650
Device: cpu

Configuration:
experiment_name: unsupervised_bert_base
model_name: bert-base-uncased
mlp_only_train: True
dataset_path: data/raw/wiki1m_for_simcse.txt
max_seq_length: 32
batch_size: 64
learning_rate: 3e-05
epochs: 1
temperature: 0.05
dropout: 0.1
max_grad_norm: 1.0
seed: 42
shuffle: True
drop_last: True
num_workers: 0
target_reference: {'official_model': 'princeton-nlp/unsup-simcse-bert-base-uncased', 'reported_average_sts': 76.25}

Command-line sentence limit:
25000

Loading tokenizer...

Loading dataset...
Number of sentences: 25000
Number of batches per epoch: 390

Loading model...

Total training steps: 390
Warmup steps: 0
Initial learning rate: 3e-05

======================================================================
STARTING TRAINING
======================================================================
Epoch 1/1 | Step 0001/0390 | Loss 0.881923 | PosSim 0.860412 | NegSim 0.592189 | LR 0.00002992
Epoch 1/1 | Step 0002/0390 | Loss 0.147804 | PosSim 0.830781 | NegSim 0.401631 | LR 0.00002985
Epoch 1/1 | Step 0003/0390 | Loss 0.042351 | PosSim 0.828154 | NegSim 0.354582 | LR 0.00002977
Epoch 1/1 | Step 0004/0390 | Loss 0.036213 | PosSim 0.815334 | NegSim 0.296887 | LR 0.00002969
Epoch 1/1 | Step 0005/0390 | Loss 0.036718 | PosSim 0.812876 | NegSim 0.274610 | LR 0.00002962
Epoch 1/1 | Step 0006/0390 | Loss 0.008063 | PosSim 0.815901 | NegSim 0.261748 | LR 0.00002954
Epoch 1/1 | Step 0007/0390 | Loss 0.012189 | PosSim 0.820328 | NegSim 0.246848 | LR 0.00002946
Epoch 1/1 | Step 0008/0390 | Loss 0.018368 | PosSim 0.821953 | NegSim 0.212620 | LR 0.00002938
Epoch 1/1 | Step 0009/0390 | Loss 0.004932 | PosSim 0.804563 | NegSim 0.189528 | LR 0.00002931
Epoch 1/1 | Step 0010/0390 | Loss 0.003439 | PosSim 0.818002 | NegSim 0.186379 | LR 0.00002923
Epoch 1/1 | Step 0011/0390 | Loss 0.003221 | PosSim 0.813567 | NegSim 0.171963 | LR 0.00002915
Epoch 1/1 | Step 0012/0390 | Loss 0.003210 | PosSim 0.814204 | NegSim 0.160926 | LR 0.00002908
Epoch 1/1 | Step 0013/0390 | Loss 0.009342 | PosSim 0.799677 | NegSim 0.142976 | LR 0.00002900
Epoch 1/1 | Step 0014/0390 | Loss 0.001973 | PosSim 0.813840 | NegSim 0.141167 | LR 0.00002892
Epoch 1/1 | Step 0015/0390 | Loss 0.009200 | PosSim 0.803506 | NegSim 0.139554 | LR 0.00002885
Epoch 1/1 | Step 0016/0390 | Loss 0.005473 | PosSim 0.801498 | NegSim 0.127127 | LR 0.00002877
Epoch 1/1 | Step 0017/0390 | Loss 0.002095 | PosSim 0.813567 | NegSim 0.120362 | LR 0.00002869
Epoch 1/1 | Step 0018/0390 | Loss 0.008461 | PosSim 0.794924 | NegSim 0.125844 | LR 0.00002862
Epoch 1/1 | Step 0019/0390 | Loss 0.001744 | PosSim 0.809933 | NegSim 0.123117 | LR 0.00002854
Epoch 1/1 | Step 0020/0390 | Loss 0.006684 | PosSim 0.808363 | NegSim 0.106212 | LR 0.00002846
Epoch 1/1 | Step 0021/0390 | Loss 0.005471 | PosSim 0.790974 | NegSim 0.109978 | LR 0.00002838
Epoch 1/1 | Step 0022/0390 | Loss 0.007321 | PosSim 0.815860 | NegSim 0.113034 | LR 0.00002831
Epoch 1/1 | Step 0023/0390 | Loss 0.007608 | PosSim 0.795433 | NegSim 0.106088 | LR 0.00002823
Epoch 1/1 | Step 0024/0390 | Loss 0.000691 | PosSim 0.816781 | NegSim 0.112122 | LR 0.00002815
Epoch 1/1 | Step 0025/0390 | Loss 0.001089 | PosSim 0.808450 | NegSim 0.107955 | LR 0.00002808
Epoch 1/1 | Step 0026/0390 | Loss 0.005122 | PosSim 0.818209 | NegSim 0.101783 | LR 0.00002800
Epoch 1/1 | Step 0027/0390 | Loss 0.002890 | PosSim 0.813189 | NegSim 0.103878 | LR 0.00002792
Epoch 1/1 | Step 0028/0390 | Loss 0.000566 | PosSim 0.816813 | NegSim 0.116572 | LR 0.00002785
Epoch 1/1 | Step 0029/0390 | Loss 0.000784 | PosSim 0.813372 | NegSim 0.097638 | LR 0.00002777
Epoch 1/1 | Step 0030/0390 | Loss 0.006090 | PosSim 0.815363 | NegSim 0.102314 | LR 0.00002769
Epoch 1/1 | Step 0031/0390 | Loss 0.000621 | PosSim 0.814250 | NegSim 0.091968 | LR 0.00002762
Epoch 1/1 | Step 0032/0390 | Loss 0.000635 | PosSim 0.814336 | NegSim 0.098518 | LR 0.00002754
Epoch 1/1 | Step 0033/0390 | Loss 0.001001 | PosSim 0.805739 | NegSim 0.086463 | LR 0.00002746
Epoch 1/1 | Step 0034/0390 | Loss 0.007670 | PosSim 0.811144 | NegSim 0.084808 | LR 0.00002738
Epoch 1/1 | Step 0035/0390 | Loss 0.000633 | PosSim 0.824340 | NegSim 0.098189 | LR 0.00002731
Epoch 1/1 | Step 0036/0390 | Loss 0.008456 | PosSim 0.791419 | NegSim 0.086685 | LR 0.00002723
Epoch 1/1 | Step 0037/0390 | Loss 0.000636 | PosSim 0.815537 | NegSim 0.096127 | LR 0.00002715
Epoch 1/1 | Step 0038/0390 | Loss 0.001610 | PosSim 0.812721 | NegSim 0.090425 | LR 0.00002708
Epoch 1/1 | Step 0039/0390 | Loss 0.000874 | PosSim 0.809425 | NegSim 0.094123 | LR 0.00002700
Epoch 1/1 | Step 0040/0390 | Loss 0.000740 | PosSim 0.816491 | NegSim 0.091529 | LR 0.00002692
Epoch 1/1 | Step 0041/0390 | Loss 0.004688 | PosSim 0.805426 | NegSim 0.091835 | LR 0.00002685
Epoch 1/1 | Step 0042/0390 | Loss 0.000773 | PosSim 0.808229 | NegSim 0.097596 | LR 0.00002677
Epoch 1/1 | Step 0043/0390 | Loss 0.003923 | PosSim 0.818772 | NegSim 0.096887 | LR 0.00002669
Epoch 1/1 | Step 0044/0390 | Loss 0.000308 | PosSim 0.808573 | NegSim 0.093615 | LR 0.00002662
Epoch 1/1 | Step 0045/0390 | Loss 0.002316 | PosSim 0.810855 | NegSim 0.091046 | LR 0.00002654
Epoch 1/1 | Step 0046/0390 | Loss 0.000456 | PosSim 0.805851 | NegSim 0.093004 | LR 0.00002646
Epoch 1/1 | Step 0047/0390 | Loss 0.002548 | PosSim 0.811546 | NegSim 0.094508 | LR 0.00002638
Epoch 1/1 | Step 0048/0390 | Loss 0.000491 | PosSim 0.805593 | NegSim 0.090814 | LR 0.00002631
Epoch 1/1 | Step 0049/0390 | Loss 0.010524 | PosSim 0.791357 | NegSim 0.088698 | LR 0.00002623
Epoch 1/1 | Step 0050/0390 | Loss 0.001826 | PosSim 0.815929 | NegSim 0.089793 | LR 0.00002615
Epoch 1/1 | Step 0051/0390 | Loss 0.001055 | PosSim 0.804420 | NegSim 0.087595 | LR 0.00002608
Epoch 1/1 | Step 0052/0390 | Loss 0.015010 | PosSim 0.813155 | NegSim 0.086052 | LR 0.00002600
Epoch 1/1 | Step 0053/0390 | Loss 0.026426 | PosSim 0.802947 | NegSim 0.083905 | LR 0.00002592
Epoch 1/1 | Step 0054/0390 | Loss 0.001554 | PosSim 0.809096 | NegSim 0.075964 | LR 0.00002585
Epoch 1/1 | Step 0055/0390 | Loss 0.000571 | PosSim 0.800975 | NegSim 0.086209 | LR 0.00002577
Epoch 1/1 | Step 0056/0390 | Loss 0.000459 | PosSim 0.814495 | NegSim 0.095863 | LR 0.00002569
Epoch 1/1 | Step 0057/0390 | Loss 0.000683 | PosSim 0.813261 | NegSim 0.087105 | LR 0.00002562
Epoch 1/1 | Step 0058/0390 | Loss 0.000264 | PosSim 0.800555 | NegSim 0.081054 | LR 0.00002554
Epoch 1/1 | Step 0059/0390 | Loss 0.001095 | PosSim 0.795856 | NegSim 0.081546 | LR 0.00002546
Epoch 1/1 | Step 0060/0390 | Loss 0.000303 | PosSim 0.803758 | NegSim 0.077836 | LR 0.00002538
Epoch 1/1 | Step 0061/0390 | Loss 0.000725 | PosSim 0.797694 | NegSim 0.084205 | LR 0.00002531
Epoch 1/1 | Step 0062/0390 | Loss 0.000795 | PosSim 0.797342 | NegSim 0.085227 | LR 0.00002523
Epoch 1/1 | Step 0063/0390 | Loss 0.000363 | PosSim 0.803687 | NegSim 0.091265 | LR 0.00002515
Epoch 1/1 | Step 0064/0390 | Loss 0.001912 | PosSim 0.806973 | NegSim 0.080663 | LR 0.00002508
Epoch 1/1 | Step 0065/0390 | Loss 0.000904 | PosSim 0.800395 | NegSim 0.093468 | LR 0.00002500
Epoch 1/1 | Step 0066/0390 | Loss 0.001065 | PosSim 0.809518 | NegSim 0.092983 | LR 0.00002492
Epoch 1/1 | Step 0067/0390 | Loss 0.010640 | PosSim 0.802937 | NegSim 0.080451 | LR 0.00002485
Epoch 1/1 | Step 0068/0390 | Loss 0.000692 | PosSim 0.807082 | NegSim 0.101024 | LR 0.00002477
Epoch 1/1 | Step 0069/0390 | Loss 0.000378 | PosSim 0.806731 | NegSim 0.083524 | LR 0.00002469
Epoch 1/1 | Step 0070/0390 | Loss 0.002856 | PosSim 0.799803 | NegSim 0.086913 | LR 0.00002462
Epoch 1/1 | Step 0071/0390 | Loss 0.000532 | PosSim 0.809579 | NegSim 0.082766 | LR 0.00002454
Epoch 1/1 | Step 0072/0390 | Loss 0.001528 | PosSim 0.804899 | NegSim 0.083926 | LR 0.00002446
Epoch 1/1 | Step 0073/0390 | Loss 0.000969 | PosSim 0.815570 | NegSim 0.086612 | LR 0.00002438
Epoch 1/1 | Step 0074/0390 | Loss 0.000404 | PosSim 0.799528 | NegSim 0.092760 | LR 0.00002431
Epoch 1/1 | Step 0075/0390 | Loss 0.002814 | PosSim 0.814049 | NegSim 0.092832 | LR 0.00002423
Epoch 1/1 | Step 0076/0390 | Loss 0.000298 | PosSim 0.804080 | NegSim 0.081078 | LR 0.00002415
Epoch 1/1 | Step 0077/0390 | Loss 0.000828 | PosSim 0.804251 | NegSim 0.081858 | LR 0.00002408
Epoch 1/1 | Step 0078/0390 | Loss 0.001780 | PosSim 0.804693 | NegSim 0.087954 | LR 0.00002400
Epoch 1/1 | Step 0079/0390 | Loss 0.000327 | PosSim 0.801108 | NegSim 0.076164 | LR 0.00002392
Epoch 1/1 | Step 0080/0390 | Loss 0.000463 | PosSim 0.805611 | NegSim 0.086555 | LR 0.00002385
Epoch 1/1 | Step 0081/0390 | Loss 0.005395 | PosSim 0.802639 | NegSim 0.077024 | LR 0.00002377
Epoch 1/1 | Step 0082/0390 | Loss 0.000265 | PosSim 0.798714 | NegSim 0.075187 | LR 0.00002369
Epoch 1/1 | Step 0083/0390 | Loss 0.002475 | PosSim 0.801804 | NegSim 0.071938 | LR 0.00002362
Epoch 1/1 | Step 0084/0390 | Loss 0.000487 | PosSim 0.805114 | NegSim 0.085364 | LR 0.00002354
Epoch 1/1 | Step 0085/0390 | Loss 0.000716 | PosSim 0.793112 | NegSim 0.070818 | LR 0.00002346
Epoch 1/1 | Step 0086/0390 | Loss 0.000663 | PosSim 0.795790 | NegSim 0.085833 | LR 0.00002338
Epoch 1/1 | Step 0087/0390 | Loss 0.000310 | PosSim 0.794182 | NegSim 0.071167 | LR 0.00002331
Epoch 1/1 | Step 0088/0390 | Loss 0.000280 | PosSim 0.802717 | NegSim 0.069693 | LR 0.00002323
Epoch 1/1 | Step 0089/0390 | Loss 0.000336 | PosSim 0.799841 | NegSim 0.071849 | LR 0.00002315
Epoch 1/1 | Step 0090/0390 | Loss 0.000260 | PosSim 0.805383 | NegSim 0.074697 | LR 0.00002308
Epoch 1/1 | Step 0091/0390 | Loss 0.000697 | PosSim 0.802653 | NegSim 0.078580 | LR 0.00002300
Epoch 1/1 | Step 0092/0390 | Loss 0.005007 | PosSim 0.796382 | NegSim 0.076938 | LR 0.00002292
Epoch 1/1 | Step 0093/0390 | Loss 0.000608 | PosSim 0.797721 | NegSim 0.069079 | LR 0.00002285
Epoch 1/1 | Step 0094/0390 | Loss 0.000838 | PosSim 0.794829 | NegSim 0.072470 | LR 0.00002277
Epoch 1/1 | Step 0095/0390 | Loss 0.000815 | PosSim 0.788918 | NegSim 0.071998 | LR 0.00002269
Epoch 1/1 | Step 0096/0390 | Loss 0.002014 | PosSim 0.786441 | NegSim 0.069672 | LR 0.00002262
Epoch 1/1 | Step 0097/0390 | Loss 0.001027 | PosSim 0.796943 | NegSim 0.073324 | LR 0.00002254
Epoch 1/1 | Step 0098/0390 | Loss 0.000425 | PosSim 0.790611 | NegSim 0.064417 | LR 0.00002246
Epoch 1/1 | Step 0099/0390 | Loss 0.000719 | PosSim 0.791309 | NegSim 0.066014 | LR 0.00002238
Epoch 1/1 | Step 0100/0390 | Loss 0.000329 | PosSim 0.786127 | NegSim 0.063617 | LR 0.00002231
Epoch 1/1 | Step 0101/0390 | Loss 0.000255 | PosSim 0.791133 | NegSim 0.066944 | LR 0.00002223
Epoch 1/1 | Step 0102/0390 | Loss 0.000343 | PosSim 0.792814 | NegSim 0.070009 | LR 0.00002215
Epoch 1/1 | Step 0103/0390 | Loss 0.000869 | PosSim 0.797528 | NegSim 0.072301 | LR 0.00002208
Epoch 1/1 | Step 0104/0390 | Loss 0.000865 | PosSim 0.802487 | NegSim 0.061485 | LR 0.00002200
Epoch 1/1 | Step 0105/0390 | Loss 0.000322 | PosSim 0.783977 | NegSim 0.062332 | LR 0.00002192
Epoch 1/1 | Step 0106/0390 | Loss 0.000371 | PosSim 0.790972 | NegSim 0.061079 | LR 0.00002185
Epoch 1/1 | Step 0107/0390 | Loss 0.000846 | PosSim 0.794818 | NegSim 0.059351 | LR 0.00002177
Epoch 1/1 | Step 0108/0390 | Loss 0.000554 | PosSim 0.787529 | NegSim 0.074536 | LR 0.00002169
Epoch 1/1 | Step 0109/0390 | Loss 0.000880 | PosSim 0.789841 | NegSim 0.061898 | LR 0.00002162
Epoch 1/1 | Step 0110/0390 | Loss 0.000197 | PosSim 0.797832 | NegSim 0.061207 | LR 0.00002154
Epoch 1/1 | Step 0111/0390 | Loss 0.000474 | PosSim 0.792691 | NegSim 0.070431 | LR 0.00002146
Epoch 1/1 | Step 0112/0390 | Loss 0.000214 | PosSim 0.799615 | NegSim 0.059835 | LR 0.00002138
Epoch 1/1 | Step 0113/0390 | Loss 0.000332 | PosSim 0.788899 | NegSim 0.064535 | LR 0.00002131
Epoch 1/1 | Step 0114/0390 | Loss 0.002065 | PosSim 0.783489 | NegSim 0.061342 | LR 0.00002123
Epoch 1/1 | Step 0115/0390 | Loss 0.001229 | PosSim 0.791472 | NegSim 0.066019 | LR 0.00002115
Epoch 1/1 | Step 0116/0390 | Loss 0.000568 | PosSim 0.792844 | NegSim 0.068689 | LR 0.00002108
Epoch 1/1 | Step 0117/0390 | Loss 0.000411 | PosSim 0.792028 | NegSim 0.065604 | LR 0.00002100
Epoch 1/1 | Step 0118/0390 | Loss 0.001020 | PosSim 0.788823 | NegSim 0.056191 | LR 0.00002092
Epoch 1/1 | Step 0119/0390 | Loss 0.000434 | PosSim 0.798791 | NegSim 0.069663 | LR 0.00002085
Epoch 1/1 | Step 0120/0390 | Loss 0.001609 | PosSim 0.788483 | NegSim 0.064034 | LR 0.00002077
Epoch 1/1 | Step 0121/0390 | Loss 0.000528 | PosSim 0.785806 | NegSim 0.068235 | LR 0.00002069
Epoch 1/1 | Step 0122/0390 | Loss 0.000442 | PosSim 0.782677 | NegSim 0.063340 | LR 0.00002062
Epoch 1/1 | Step 0123/0390 | Loss 0.000423 | PosSim 0.785202 | NegSim 0.068153 | LR 0.00002054
Epoch 1/1 | Step 0124/0390 | Loss 0.000319 | PosSim 0.790840 | NegSim 0.059286 | LR 0.00002046
Epoch 1/1 | Step 0125/0390 | Loss 0.011090 | PosSim 0.780434 | NegSim 0.052442 | LR 0.00002038
Epoch 1/1 | Step 0126/0390 | Loss 0.000687 | PosSim 0.782504 | NegSim 0.055454 | LR 0.00002031
Epoch 1/1 | Step 0127/0390 | Loss 0.000323 | PosSim 0.791044 | NegSim 0.057367 | LR 0.00002023
Epoch 1/1 | Step 0128/0390 | Loss 0.000655 | PosSim 0.794725 | NegSim 0.063488 | LR 0.00002015
Epoch 1/1 | Step 0129/0390 | Loss 0.000641 | PosSim 0.791639 | NegSim 0.055467 | LR 0.00002008
Epoch 1/1 | Step 0130/0390 | Loss 0.011546 | PosSim 0.792772 | NegSim 0.054276 | LR 0.00002000
Epoch 1/1 | Step 0131/0390 | Loss 0.000269 | PosSim 0.783039 | NegSim 0.052655 | LR 0.00001992
Epoch 1/1 | Step 0132/0390 | Loss 0.000203 | PosSim 0.800615 | NegSim 0.065077 | LR 0.00001985
Epoch 1/1 | Step 0133/0390 | Loss 0.000706 | PosSim 0.790437 | NegSim 0.059589 | LR 0.00001977
Epoch 1/1 | Step 0134/0390 | Loss 0.001142 | PosSim 0.792290 | NegSim 0.056205 | LR 0.00001969
Epoch 1/1 | Step 0135/0390 | Loss 0.000323 | PosSim 0.795729 | NegSim 0.057627 | LR 0.00001962
Epoch 1/1 | Step 0136/0390 | Loss 0.021794 | PosSim 0.780614 | NegSim 0.056147 | LR 0.00001954
Epoch 1/1 | Step 0137/0390 | Loss 0.000272 | PosSim 0.793835 | NegSim 0.061538 | LR 0.00001946
Epoch 1/1 | Step 0138/0390 | Loss 0.000513 | PosSim 0.792358 | NegSim 0.066891 | LR 0.00001938
Epoch 1/1 | Step 0139/0390 | Loss 0.000610 | PosSim 0.795129 | NegSim 0.060515 | LR 0.00001931
Epoch 1/1 | Step 0140/0390 | Loss 0.000747 | PosSim 0.793183 | NegSim 0.056572 | LR 0.00001923
Epoch 1/1 | Step 0141/0390 | Loss 0.001494 | PosSim 0.783691 | NegSim 0.055699 | LR 0.00001915
Epoch 1/1 | Step 0142/0390 | Loss 0.000617 | PosSim 0.788059 | NegSim 0.057269 | LR 0.00001908
Epoch 1/1 | Step 0143/0390 | Loss 0.000322 | PosSim 0.779166 | NegSim 0.050619 | LR 0.00001900
Epoch 1/1 | Step 0144/0390 | Loss 0.002861 | PosSim 0.792204 | NegSim 0.060542 | LR 0.00001892
Epoch 1/1 | Step 0145/0390 | Loss 0.025423 | PosSim 0.789424 | NegSim 0.056731 | LR 0.00001885
Epoch 1/1 | Step 0146/0390 | Loss 0.000258 | PosSim 0.790266 | NegSim 0.060702 | LR 0.00001877
Epoch 1/1 | Step 0147/0390 | Loss 0.002003 | PosSim 0.799568 | NegSim 0.061450 | LR 0.00001869
Epoch 1/1 | Step 0148/0390 | Loss 0.000303 | PosSim 0.789909 | NegSim 0.065904 | LR 0.00001862
Epoch 1/1 | Step 0149/0390 | Loss 0.004115 | PosSim 0.787444 | NegSim 0.057357 | LR 0.00001854
Epoch 1/1 | Step 0150/0390 | Loss 0.000478 | PosSim 0.790172 | NegSim 0.063029 | LR 0.00001846
Epoch 1/1 | Step 0151/0390 | Loss 0.000199 | PosSim 0.792918 | NegSim 0.057537 | LR 0.00001838
Epoch 1/1 | Step 0152/0390 | Loss 0.001008 | PosSim 0.795777 | NegSim 0.066847 | LR 0.00001831
Epoch 1/1 | Step 0153/0390 | Loss 0.000566 | PosSim 0.800149 | NegSim 0.063509 | LR 0.00001823
Epoch 1/1 | Step 0154/0390 | Loss 0.000649 | PosSim 0.781354 | NegSim 0.055876 | LR 0.00001815
Epoch 1/1 | Step 0155/0390 | Loss 0.000710 | PosSim 0.793733 | NegSim 0.055452 | LR 0.00001808
Epoch 1/1 | Step 0156/0390 | Loss 0.000410 | PosSim 0.794721 | NegSim 0.056580 | LR 0.00001800
Epoch 1/1 | Step 0157/0390 | Loss 0.002148 | PosSim 0.793248 | NegSim 0.061684 | LR 0.00001792
Epoch 1/1 | Step 0158/0390 | Loss 0.000406 | PosSim 0.790145 | NegSim 0.056728 | LR 0.00001785
Epoch 1/1 | Step 0159/0390 | Loss 0.000220 | PosSim 0.797232 | NegSim 0.059939 | LR 0.00001777
Epoch 1/1 | Step 0160/0390 | Loss 0.000507 | PosSim 0.784875 | NegSim 0.060263 | LR 0.00001769
Epoch 1/1 | Step 0161/0390 | Loss 0.000937 | PosSim 0.778815 | NegSim 0.066275 | LR 0.00001762
Epoch 1/1 | Step 0162/0390 | Loss 0.013215 | PosSim 0.785424 | NegSim 0.059310 | LR 0.00001754
Epoch 1/1 | Step 0163/0390 | Loss 0.000162 | PosSim 0.791600 | NegSim 0.058713 | LR 0.00001746
Epoch 1/1 | Step 0164/0390 | Loss 0.000263 | PosSim 0.784731 | NegSim 0.068098 | LR 0.00001738
Epoch 1/1 | Step 0165/0390 | Loss 0.000618 | PosSim 0.790023 | NegSim 0.058658 | LR 0.00001731
Epoch 1/1 | Step 0166/0390 | Loss 0.009856 | PosSim 0.786693 | NegSim 0.065846 | LR 0.00001723
Epoch 1/1 | Step 0167/0390 | Loss 0.000187 | PosSim 0.788388 | NegSim 0.057324 | LR 0.00001715
Epoch 1/1 | Step 0168/0390 | Loss 0.000234 | PosSim 0.795548 | NegSim 0.051827 | LR 0.00001708
Epoch 1/1 | Step 0169/0390 | Loss 0.000176 | PosSim 0.789885 | NegSim 0.058038 | LR 0.00001700
Epoch 1/1 | Step 0170/0390 | Loss 0.000254 | PosSim 0.790553 | NegSim 0.056103 | LR 0.00001692
Epoch 1/1 | Step 0171/0390 | Loss 0.000355 | PosSim 0.786822 | NegSim 0.060821 | LR 0.00001685
Epoch 1/1 | Step 0172/0390 | Loss 0.000926 | PosSim 0.790156 | NegSim 0.057659 | LR 0.00001677
Epoch 1/1 | Step 0173/0390 | Loss 0.002055 | PosSim 0.789962 | NegSim 0.050054 | LR 0.00001669
Epoch 1/1 | Step 0174/0390 | Loss 0.001895 | PosSim 0.791812 | NegSim 0.060636 | LR 0.00001662
Epoch 1/1 | Step 0175/0390 | Loss 0.000220 | PosSim 0.794989 | NegSim 0.059804 | LR 0.00001654
Epoch 1/1 | Step 0176/0390 | Loss 0.000822 | PosSim 0.796613 | NegSim 0.068517 | LR 0.00001646
Epoch 1/1 | Step 0177/0390 | Loss 0.000199 | PosSim 0.789031 | NegSim 0.053307 | LR 0.00001638
Epoch 1/1 | Step 0178/0390 | Loss 0.000250 | PosSim 0.789691 | NegSim 0.056369 | LR 0.00001631
Epoch 1/1 | Step 0179/0390 | Loss 0.001373 | PosSim 0.782905 | NegSim 0.058420 | LR 0.00001623
Epoch 1/1 | Step 0180/0390 | Loss 0.000361 | PosSim 0.785063 | NegSim 0.053195 | LR 0.00001615
Epoch 1/1 | Step 0181/0390 | Loss 0.001521 | PosSim 0.777828 | NegSim 0.053189 | LR 0.00001608
Epoch 1/1 | Step 0182/0390 | Loss 0.000195 | PosSim 0.786936 | NegSim 0.056925 | LR 0.00001600
Epoch 1/1 | Step 0183/0390 | Loss 0.000269 | PosSim 0.780587 | NegSim 0.058621 | LR 0.00001592
Epoch 1/1 | Step 0184/0390 | Loss 0.000491 | PosSim 0.774716 | NegSim 0.060584 | LR 0.00001585
Epoch 1/1 | Step 0185/0390 | Loss 0.000290 | PosSim 0.785900 | NegSim 0.055659 | LR 0.00001577
Epoch 1/1 | Step 0186/0390 | Loss 0.000246 | PosSim 0.778366 | NegSim 0.052767 | LR 0.00001569
Epoch 1/1 | Step 0187/0390 | Loss 0.000405 | PosSim 0.786845 | NegSim 0.055921 | LR 0.00001562
Epoch 1/1 | Step 0188/0390 | Loss 0.027501 | PosSim 0.790802 | NegSim 0.055384 | LR 0.00001554
Epoch 1/1 | Step 0189/0390 | Loss 0.000347 | PosSim 0.788665 | NegSim 0.061441 | LR 0.00001546
Epoch 1/1 | Step 0190/0390 | Loss 0.000307 | PosSim 0.788415 | NegSim 0.053272 | LR 0.00001538
Epoch 1/1 | Step 0191/0390 | Loss 0.000254 | PosSim 0.787009 | NegSim 0.050472 | LR 0.00001531
Epoch 1/1 | Step 0192/0390 | Loss 0.000713 | PosSim 0.783237 | NegSim 0.054137 | LR 0.00001523
Epoch 1/1 | Step 0193/0390 | Loss 0.000238 | PosSim 0.789221 | NegSim 0.054427 | LR 0.00001515
Epoch 1/1 | Step 0194/0390 | Loss 0.012052 | PosSim 0.789771 | NegSim 0.055855 | LR 0.00001508
Epoch 1/1 | Step 0195/0390 | Loss 0.000359 | PosSim 0.787568 | NegSim 0.058117 | LR 0.00001500
Epoch 1/1 | Step 0196/0390 | Loss 0.004583 | PosSim 0.791001 | NegSim 0.053546 | LR 0.00001492
Epoch 1/1 | Step 0197/0390 | Loss 0.001283 | PosSim 0.785249 | NegSim 0.054117 | LR 0.00001485
Epoch 1/1 | Step 0198/0390 | Loss 0.000139 | PosSim 0.789066 | NegSim 0.053056 | LR 0.00001477
Epoch 1/1 | Step 0199/0390 | Loss 0.000330 | PosSim 0.780815 | NegSim 0.052756 | LR 0.00001469
Epoch 1/1 | Step 0200/0390 | Loss 0.004006 | PosSim 0.783332 | NegSim 0.052815 | LR 0.00001462
Epoch 1/1 | Step 0201/0390 | Loss 0.003075 | PosSim 0.792528 | NegSim 0.060480 | LR 0.00001454
Epoch 1/1 | Step 0202/0390 | Loss 0.000373 | PosSim 0.790949 | NegSim 0.055480 | LR 0.00001446
Epoch 1/1 | Step 0203/0390 | Loss 0.000335 | PosSim 0.786927 | NegSim 0.060554 | LR 0.00001438
Epoch 1/1 | Step 0204/0390 | Loss 0.000247 | PosSim 0.789253 | NegSim 0.053690 | LR 0.00001431
Epoch 1/1 | Step 0205/0390 | Loss 0.000693 | PosSim 0.785415 | NegSim 0.056129 | LR 0.00001423
Epoch 1/1 | Step 0206/0390 | Loss 0.000656 | PosSim 0.774479 | NegSim 0.053636 | LR 0.00001415
Epoch 1/1 | Step 0207/0390 | Loss 0.000430 | PosSim 0.777461 | NegSim 0.055103 | LR 0.00001408
Epoch 1/1 | Step 0208/0390 | Loss 0.000754 | PosSim 0.786957 | NegSim 0.052156 | LR 0.00001400
Epoch 1/1 | Step 0209/0390 | Loss 0.000285 | PosSim 0.773306 | NegSim 0.050853 | LR 0.00001392
Epoch 1/1 | Step 0210/0390 | Loss 0.000420 | PosSim 0.785307 | NegSim 0.050140 | LR 0.00001385
Epoch 1/1 | Step 0211/0390 | Loss 0.000112 | PosSim 0.782926 | NegSim 0.048885 | LR 0.00001377
Epoch 1/1 | Step 0212/0390 | Loss 0.000544 | PosSim 0.778924 | NegSim 0.057869 | LR 0.00001369
Epoch 1/1 | Step 0213/0390 | Loss 0.000272 | PosSim 0.778865 | NegSim 0.048223 | LR 0.00001362
Epoch 1/1 | Step 0214/0390 | Loss 0.000271 | PosSim 0.781794 | NegSim 0.055323 | LR 0.00001354
Epoch 1/1 | Step 0215/0390 | Loss 0.000341 | PosSim 0.779527 | NegSim 0.054137 | LR 0.00001346
Epoch 1/1 | Step 0216/0390 | Loss 0.001861 | PosSim 0.781540 | NegSim 0.050928 | LR 0.00001338
Epoch 1/1 | Step 0217/0390 | Loss 0.000493 | PosSim 0.777886 | NegSim 0.050410 | LR 0.00001331
Epoch 1/1 | Step 0218/0390 | Loss 0.000298 | PosSim 0.779627 | NegSim 0.048908 | LR 0.00001323
Epoch 1/1 | Step 0219/0390 | Loss 0.000611 | PosSim 0.775524 | NegSim 0.051624 | LR 0.00001315
Epoch 1/1 | Step 0220/0390 | Loss 0.000351 | PosSim 0.774960 | NegSim 0.048346 | LR 0.00001308
Epoch 1/1 | Step 0221/0390 | Loss 0.001001 | PosSim 0.767797 | NegSim 0.047287 | LR 0.00001300
Epoch 1/1 | Step 0222/0390 | Loss 0.000278 | PosSim 0.785314 | NegSim 0.052063 | LR 0.00001292
Epoch 1/1 | Step 0223/0390 | Loss 0.000304 | PosSim 0.773481 | NegSim 0.052400 | LR 0.00001285
Epoch 1/1 | Step 0224/0390 | Loss 0.000363 | PosSim 0.777799 | NegSim 0.049944 | LR 0.00001277
Epoch 1/1 | Step 0225/0390 | Loss 0.000197 | PosSim 0.778412 | NegSim 0.041286 | LR 0.00001269
Epoch 1/1 | Step 0226/0390 | Loss 0.031766 | PosSim 0.773373 | NegSim 0.051138 | LR 0.00001262
Epoch 1/1 | Step 0227/0390 | Loss 0.000359 | PosSim 0.775038 | NegSim 0.056617 | LR 0.00001254
Epoch 1/1 | Step 0228/0390 | Loss 0.000215 | PosSim 0.780536 | NegSim 0.056972 | LR 0.00001246
Epoch 1/1 | Step 0229/0390 | Loss 0.000324 | PosSim 0.777432 | NegSim 0.046390 | LR 0.00001238
Epoch 1/1 | Step 0230/0390 | Loss 0.000476 | PosSim 0.774858 | NegSim 0.050804 | LR 0.00001231
Epoch 1/1 | Step 0231/0390 | Loss 0.000926 | PosSim 0.780568 | NegSim 0.046707 | LR 0.00001223
Epoch 1/1 | Step 0232/0390 | Loss 0.000157 | PosSim 0.787452 | NegSim 0.052875 | LR 0.00001215
Epoch 1/1 | Step 0233/0390 | Loss 0.000193 | PosSim 0.783341 | NegSim 0.051037 | LR 0.00001208
Epoch 1/1 | Step 0234/0390 | Loss 0.000321 | PosSim 0.776990 | NegSim 0.054312 | LR 0.00001200
Epoch 1/1 | Step 0235/0390 | Loss 0.000238 | PosSim 0.782103 | NegSim 0.055121 | LR 0.00001192
Epoch 1/1 | Step 0236/0390 | Loss 0.027827 | PosSim 0.785661 | NegSim 0.057145 | LR 0.00001185
Epoch 1/1 | Step 0237/0390 | Loss 0.000308 | PosSim 0.775163 | NegSim 0.056448 | LR 0.00001177
Epoch 1/1 | Step 0238/0390 | Loss 0.000161 | PosSim 0.777797 | NegSim 0.056358 | LR 0.00001169
Epoch 1/1 | Step 0239/0390 | Loss 0.000376 | PosSim 0.778213 | NegSim 0.048798 | LR 0.00001162
Epoch 1/1 | Step 0240/0390 | Loss 0.000323 | PosSim 0.782033 | NegSim 0.053610 | LR 0.00001154
Epoch 1/1 | Step 0241/0390 | Loss 0.000609 | PosSim 0.782458 | NegSim 0.052182 | LR 0.00001146
Epoch 1/1 | Step 0242/0390 | Loss 0.000168 | PosSim 0.794394 | NegSim 0.058402 | LR 0.00001138
Epoch 1/1 | Step 0243/0390 | Loss 0.000294 | PosSim 0.785506 | NegSim 0.050755 | LR 0.00001131
Epoch 1/1 | Step 0244/0390 | Loss 0.000790 | PosSim 0.785091 | NegSim 0.050819 | LR 0.00001123
Epoch 1/1 | Step 0245/0390 | Loss 0.000382 | PosSim 0.777264 | NegSim 0.052295 | LR 0.00001115
Epoch 1/1 | Step 0246/0390 | Loss 0.008862 | PosSim 0.794806 | NegSim 0.048228 | LR 0.00001108
Epoch 1/1 | Step 0247/0390 | Loss 0.000247 | PosSim 0.792928 | NegSim 0.048204 | LR 0.00001100
Epoch 1/1 | Step 0248/0390 | Loss 0.000266 | PosSim 0.785637 | NegSim 0.050284 | LR 0.00001092
Epoch 1/1 | Step 0249/0390 | Loss 0.000314 | PosSim 0.793237 | NegSim 0.047857 | LR 0.00001085
Epoch 1/1 | Step 0250/0390 | Loss 0.000285 | PosSim 0.793265 | NegSim 0.052545 | LR 0.00001077
Epoch 1/1 | Step 0251/0390 | Loss 0.000180 | PosSim 0.783862 | NegSim 0.055291 | LR 0.00001069
Epoch 1/1 | Step 0252/0390 | Loss 0.000616 | PosSim 0.791687 | NegSim 0.056280 | LR 0.00001062
Epoch 1/1 | Step 0253/0390 | Loss 0.000984 | PosSim 0.786626 | NegSim 0.060115 | LR 0.00001054
Epoch 1/1 | Step 0254/0390 | Loss 0.000251 | PosSim 0.789074 | NegSim 0.058140 | LR 0.00001046
Epoch 1/1 | Step 0255/0390 | Loss 0.000161 | PosSim 0.790852 | NegSim 0.054739 | LR 0.00001038
Epoch 1/1 | Step 0256/0390 | Loss 0.000208 | PosSim 0.783402 | NegSim 0.059065 | LR 0.00001031
Epoch 1/1 | Step 0257/0390 | Loss 0.000211 | PosSim 0.788717 | NegSim 0.053050 | LR 0.00001023
Epoch 1/1 | Step 0258/0390 | Loss 0.000244 | PosSim 0.792368 | NegSim 0.051454 | LR 0.00001015
Epoch 1/1 | Step 0259/0390 | Loss 0.000842 | PosSim 0.781972 | NegSim 0.047826 | LR 0.00001008
Epoch 1/1 | Step 0260/0390 | Loss 0.000758 | PosSim 0.776997 | NegSim 0.049506 | LR 0.00001000
Epoch 1/1 | Step 0261/0390 | Loss 0.000519 | PosSim 0.790703 | NegSim 0.060080 | LR 0.00000992
Epoch 1/1 | Step 0262/0390 | Loss 0.000378 | PosSim 0.787162 | NegSim 0.055630 | LR 0.00000985
Epoch 1/1 | Step 0263/0390 | Loss 0.000228 | PosSim 0.784339 | NegSim 0.050533 | LR 0.00000977
Epoch 1/1 | Step 0264/0390 | Loss 0.000312 | PosSim 0.776256 | NegSim 0.062430 | LR 0.00000969
Epoch 1/1 | Step 0265/0390 | Loss 0.000231 | PosSim 0.792491 | NegSim 0.057602 | LR 0.00000962
Epoch 1/1 | Step 0266/0390 | Loss 0.000292 | PosSim 0.778304 | NegSim 0.056299 | LR 0.00000954
Epoch 1/1 | Step 0267/0390 | Loss 0.000163 | PosSim 0.792125 | NegSim 0.053483 | LR 0.00000946
Epoch 1/1 | Step 0268/0390 | Loss 0.000227 | PosSim 0.784810 | NegSim 0.051118 | LR 0.00000938
Epoch 1/1 | Step 0269/0390 | Loss 0.006946 | PosSim 0.790518 | NegSim 0.054545 | LR 0.00000931
Epoch 1/1 | Step 0270/0390 | Loss 0.000597 | PosSim 0.781182 | NegSim 0.048179 | LR 0.00000923
Epoch 1/1 | Step 0271/0390 | Loss 0.000369 | PosSim 0.786697 | NegSim 0.054755 | LR 0.00000915
Epoch 1/1 | Step 0272/0390 | Loss 0.000138 | PosSim 0.786006 | NegSim 0.053055 | LR 0.00000908
Epoch 1/1 | Step 0273/0390 | Loss 0.000261 | PosSim 0.786307 | NegSim 0.060788 | LR 0.00000900
Epoch 1/1 | Step 0274/0390 | Loss 0.001013 | PosSim 0.774185 | NegSim 0.050746 | LR 0.00000892
Epoch 1/1 | Step 0275/0390 | Loss 0.000286 | PosSim 0.791232 | NegSim 0.052711 | LR 0.00000885
Epoch 1/1 | Step 0276/0390 | Loss 0.000262 | PosSim 0.786339 | NegSim 0.053007 | LR 0.00000877
Epoch 1/1 | Step 0277/0390 | Loss 0.000592 | PosSim 0.777642 | NegSim 0.053439 | LR 0.00000869
Epoch 1/1 | Step 0278/0390 | Loss 0.000227 | PosSim 0.778403 | NegSim 0.056951 | LR 0.00000862
Epoch 1/1 | Step 0279/0390 | Loss 0.001109 | PosSim 0.788221 | NegSim 0.051767 | LR 0.00000854
Epoch 1/1 | Step 0280/0390 | Loss 0.000185 | PosSim 0.786738 | NegSim 0.053232 | LR 0.00000846
Epoch 1/1 | Step 0281/0390 | Loss 0.000668 | PosSim 0.785455 | NegSim 0.051910 | LR 0.00000838
Epoch 1/1 | Step 0282/0390 | Loss 0.000501 | PosSim 0.778385 | NegSim 0.051685 | LR 0.00000831
Epoch 1/1 | Step 0283/0390 | Loss 0.004802 | PosSim 0.785878 | NegSim 0.056344 | LR 0.00000823
Epoch 1/1 | Step 0284/0390 | Loss 0.000313 | PosSim 0.783823 | NegSim 0.052390 | LR 0.00000815
Epoch 1/1 | Step 0285/0390 | Loss 0.000236 | PosSim 0.786767 | NegSim 0.051972 | LR 0.00000808
Epoch 1/1 | Step 0286/0390 | Loss 0.000162 | PosSim 0.782521 | NegSim 0.048329 | LR 0.00000800
Epoch 1/1 | Step 0287/0390 | Loss 0.000213 | PosSim 0.785197 | NegSim 0.055538 | LR 0.00000792
Epoch 1/1 | Step 0288/0390 | Loss 0.000464 | PosSim 0.793219 | NegSim 0.052993 | LR 0.00000785
Epoch 1/1 | Step 0289/0390 | Loss 0.000248 | PosSim 0.795784 | NegSim 0.053702 | LR 0.00000777
Epoch 1/1 | Step 0290/0390 | Loss 0.000265 | PosSim 0.775068 | NegSim 0.042994 | LR 0.00000769
Epoch 1/1 | Step 0291/0390 | Loss 0.000135 | PosSim 0.783868 | NegSim 0.045471 | LR 0.00000762
Epoch 1/1 | Step 0292/0390 | Loss 0.000336 | PosSim 0.784885 | NegSim 0.047348 | LR 0.00000754
Epoch 1/1 | Step 0293/0390 | Loss 0.000829 | PosSim 0.778768 | NegSim 0.047818 | LR 0.00000746
Epoch 1/1 | Step 0294/0390 | Loss 0.000270 | PosSim 0.781309 | NegSim 0.045522 | LR 0.00000738
Epoch 1/1 | Step 0295/0390 | Loss 0.000252 | PosSim 0.785244 | NegSim 0.050380 | LR 0.00000731
Epoch 1/1 | Step 0296/0390 | Loss 0.000380 | PosSim 0.775718 | NegSim 0.051798 | LR 0.00000723
Epoch 1/1 | Step 0297/0390 | Loss 0.017102 | PosSim 0.784111 | NegSim 0.048622 | LR 0.00000715
Epoch 1/1 | Step 0298/0390 | Loss 0.000138 | PosSim 0.785523 | NegSim 0.042172 | LR 0.00000708
Epoch 1/1 | Step 0299/0390 | Loss 0.000278 | PosSim 0.780926 | NegSim 0.046902 | LR 0.00000700
Epoch 1/1 | Step 0300/0390 | Loss 0.000114 | PosSim 0.790928 | NegSim 0.048631 | LR 0.00000692
Epoch 1/1 | Step 0301/0390 | Loss 0.000235 | PosSim 0.791356 | NegSim 0.048525 | LR 0.00000685
Epoch 1/1 | Step 0302/0390 | Loss 0.000324 | PosSim 0.782941 | NegSim 0.046114 | LR 0.00000677
Epoch 1/1 | Step 0303/0390 | Loss 0.000211 | PosSim 0.776878 | NegSim 0.049338 | LR 0.00000669
Epoch 1/1 | Step 0304/0390 | Loss 0.000287 | PosSim 0.792335 | NegSim 0.048613 | LR 0.00000662
Epoch 1/1 | Step 0305/0390 | Loss 0.000212 | PosSim 0.782365 | NegSim 0.048789 | LR 0.00000654
Epoch 1/1 | Step 0306/0390 | Loss 0.000229 | PosSim 0.782386 | NegSim 0.048451 | LR 0.00000646
Epoch 1/1 | Step 0307/0390 | Loss 0.001042 | PosSim 0.788674 | NegSim 0.044262 | LR 0.00000638
Epoch 1/1 | Step 0308/0390 | Loss 0.000219 | PosSim 0.793843 | NegSim 0.049031 | LR 0.00000631
Epoch 1/1 | Step 0309/0390 | Loss 0.000161 | PosSim 0.787789 | NegSim 0.053476 | LR 0.00000623
Epoch 1/1 | Step 0310/0390 | Loss 0.000378 | PosSim 0.783158 | NegSim 0.049110 | LR 0.00000615
Epoch 1/1 | Step 0311/0390 | Loss 0.000125 | PosSim 0.795710 | NegSim 0.053341 | LR 0.00000608
Epoch 1/1 | Step 0312/0390 | Loss 0.000640 | PosSim 0.785665 | NegSim 0.052162 | LR 0.00000600
Epoch 1/1 | Step 0313/0390 | Loss 0.000154 | PosSim 0.783581 | NegSim 0.051374 | LR 0.00000592
Epoch 1/1 | Step 0314/0390 | Loss 0.003604 | PosSim 0.780772 | NegSim 0.050420 | LR 0.00000585
Epoch 1/1 | Step 0315/0390 | Loss 0.000283 | PosSim 0.785380 | NegSim 0.050815 | LR 0.00000577
Epoch 1/1 | Step 0316/0390 | Loss 0.003724 | PosSim 0.791958 | NegSim 0.053527 | LR 0.00000569
Epoch 1/1 | Step 0317/0390 | Loss 0.000260 | PosSim 0.775729 | NegSim 0.045446 | LR 0.00000562
Epoch 1/1 | Step 0318/0390 | Loss 0.000267 | PosSim 0.782290 | NegSim 0.041321 | LR 0.00000554
Epoch 1/1 | Step 0319/0390 | Loss 0.000260 | PosSim 0.785986 | NegSim 0.055052 | LR 0.00000546
Epoch 1/1 | Step 0320/0390 | Loss 0.000198 | PosSim 0.786677 | NegSim 0.051394 | LR 0.00000538
Epoch 1/1 | Step 0321/0390 | Loss 0.000369 | PosSim 0.784431 | NegSim 0.051859 | LR 0.00000531
Epoch 1/1 | Step 0322/0390 | Loss 0.000206 | PosSim 0.789293 | NegSim 0.042345 | LR 0.00000523
Epoch 1/1 | Step 0323/0390 | Loss 0.000152 | PosSim 0.790583 | NegSim 0.043597 | LR 0.00000515
Epoch 1/1 | Step 0324/0390 | Loss 0.000317 | PosSim 0.779745 | NegSim 0.046336 | LR 0.00000508
Epoch 1/1 | Step 0325/0390 | Loss 0.000262 | PosSim 0.783000 | NegSim 0.047595 | LR 0.00000500
Epoch 1/1 | Step 0326/0390 | Loss 0.000119 | PosSim 0.797463 | NegSim 0.045539 | LR 0.00000492
Epoch 1/1 | Step 0327/0390 | Loss 0.000177 | PosSim 0.783039 | NegSim 0.049700 | LR 0.00000485
Epoch 1/1 | Step 0328/0390 | Loss 0.000524 | PosSim 0.778392 | NegSim 0.049703 | LR 0.00000477
Epoch 1/1 | Step 0329/0390 | Loss 0.000223 | PosSim 0.779061 | NegSim 0.044331 | LR 0.00000469
Epoch 1/1 | Step 0330/0390 | Loss 0.000223 | PosSim 0.785948 | NegSim 0.049090 | LR 0.00000462
Epoch 1/1 | Step 0331/0390 | Loss 0.016615 | PosSim 0.788916 | NegSim 0.053016 | LR 0.00000454
Epoch 1/1 | Step 0332/0390 | Loss 0.003535 | PosSim 0.780729 | NegSim 0.043883 | LR 0.00000446
Epoch 1/1 | Step 0333/0390 | Loss 0.000183 | PosSim 0.791334 | NegSim 0.048335 | LR 0.00000438
Epoch 1/1 | Step 0334/0390 | Loss 0.000820 | PosSim 0.784642 | NegSim 0.054045 | LR 0.00000431
Epoch 1/1 | Step 0335/0390 | Loss 0.000564 | PosSim 0.782931 | NegSim 0.046453 | LR 0.00000423
Epoch 1/1 | Step 0336/0390 | Loss 0.000291 | PosSim 0.785582 | NegSim 0.047610 | LR 0.00000415
Epoch 1/1 | Step 0337/0390 | Loss 0.000206 | PosSim 0.789935 | NegSim 0.048297 | LR 0.00000408
Epoch 1/1 | Step 0338/0390 | Loss 0.000289 | PosSim 0.781107 | NegSim 0.053542 | LR 0.00000400
Epoch 1/1 | Step 0339/0390 | Loss 0.000224 | PosSim 0.786857 | NegSim 0.048562 | LR 0.00000392
Epoch 1/1 | Step 0340/0390 | Loss 0.000690 | PosSim 0.789269 | NegSim 0.047495 | LR 0.00000385
Epoch 1/1 | Step 0341/0390 | Loss 0.000379 | PosSim 0.777904 | NegSim 0.048020 | LR 0.00000377
Epoch 1/1 | Step 0342/0390 | Loss 0.000191 | PosSim 0.789896 | NegSim 0.045161 | LR 0.00000369
Epoch 1/1 | Step 0343/0390 | Loss 0.001134 | PosSim 0.773377 | NegSim 0.044220 | LR 0.00000362
Epoch 1/1 | Step 0344/0390 | Loss 0.000222 | PosSim 0.783032 | NegSim 0.041594 | LR 0.00000354
Epoch 1/1 | Step 0345/0390 | Loss 0.000234 | PosSim 0.791425 | NegSim 0.050594 | LR 0.00000346
Epoch 1/1 | Step 0346/0390 | Loss 0.000133 | PosSim 0.789818 | NegSim 0.039415 | LR 0.00000338
Epoch 1/1 | Step 0347/0390 | Loss 0.022730 | PosSim 0.791952 | NegSim 0.043098 | LR 0.00000331
Epoch 1/1 | Step 0348/0390 | Loss 0.000175 | PosSim 0.789520 | NegSim 0.044977 | LR 0.00000323
Epoch 1/1 | Step 0349/0390 | Loss 0.000333 | PosSim 0.791662 | NegSim 0.051795 | LR 0.00000315
Epoch 1/1 | Step 0350/0390 | Loss 0.018867 | PosSim 0.779613 | NegSim 0.044586 | LR 0.00000308
Epoch 1/1 | Step 0351/0390 | Loss 0.000125 | PosSim 0.792832 | NegSim 0.046573 | LR 0.00000300
Epoch 1/1 | Step 0352/0390 | Loss 0.000435 | PosSim 0.788556 | NegSim 0.048454 | LR 0.00000292
Epoch 1/1 | Step 0353/0390 | Loss 0.000258 | PosSim 0.775919 | NegSim 0.045446 | LR 0.00000285
Epoch 1/1 | Step 0354/0390 | Loss 0.000326 | PosSim 0.781229 | NegSim 0.044608 | LR 0.00000277
Epoch 1/1 | Step 0355/0390 | Loss 0.001297 | PosSim 0.786511 | NegSim 0.044172 | LR 0.00000269
Epoch 1/1 | Step 0356/0390 | Loss 0.000142 | PosSim 0.794472 | NegSim 0.054158 | LR 0.00000262
Epoch 1/1 | Step 0357/0390 | Loss 0.000344 | PosSim 0.787077 | NegSim 0.042753 | LR 0.00000254
Epoch 1/1 | Step 0358/0390 | Loss 0.000140 | PosSim 0.782260 | NegSim 0.040627 | LR 0.00000246
Epoch 1/1 | Step 0359/0390 | Loss 0.000334 | PosSim 0.773264 | NegSim 0.045787 | LR 0.00000238
Epoch 1/1 | Step 0360/0390 | Loss 0.000206 | PosSim 0.775407 | NegSim 0.046576 | LR 0.00000231
Epoch 1/1 | Step 0361/0390 | Loss 0.000221 | PosSim 0.783317 | NegSim 0.046247 | LR 0.00000223
Epoch 1/1 | Step 0362/0390 | Loss 0.000191 | PosSim 0.784174 | NegSim 0.056801 | LR 0.00000215
Epoch 1/1 | Step 0363/0390 | Loss 0.000275 | PosSim 0.786090 | NegSim 0.047303 | LR 0.00000208
Epoch 1/1 | Step 0364/0390 | Loss 0.000557 | PosSim 0.786837 | NegSim 0.045047 | LR 0.00000200
Epoch 1/1 | Step 0365/0390 | Loss 0.000782 | PosSim 0.777863 | NegSim 0.046991 | LR 0.00000192
Epoch 1/1 | Step 0366/0390 | Loss 0.000435 | PosSim 0.784681 | NegSim 0.042097 | LR 0.00000185
Epoch 1/1 | Step 0367/0390 | Loss 0.000289 | PosSim 0.780097 | NegSim 0.044817 | LR 0.00000177
Epoch 1/1 | Step 0368/0390 | Loss 0.000267 | PosSim 0.782965 | NegSim 0.041328 | LR 0.00000169
Epoch 1/1 | Step 0369/0390 | Loss 0.000368 | PosSim 0.781839 | NegSim 0.044895 | LR 0.00000162
Epoch 1/1 | Step 0370/0390 | Loss 0.000532 | PosSim 0.771159 | NegSim 0.044073 | LR 0.00000154
Epoch 1/1 | Step 0371/0390 | Loss 0.000279 | PosSim 0.785465 | NegSim 0.050622 | LR 0.00000146
Epoch 1/1 | Step 0372/0390 | Loss 0.000525 | PosSim 0.795453 | NegSim 0.043386 | LR 0.00000138
Epoch 1/1 | Step 0373/0390 | Loss 0.000491 | PosSim 0.773590 | NegSim 0.045278 | LR 0.00000131
Epoch 1/1 | Step 0374/0390 | Loss 0.001412 | PosSim 0.789310 | NegSim 0.043066 | LR 0.00000123
Epoch 1/1 | Step 0375/0390 | Loss 0.000153 | PosSim 0.780165 | NegSim 0.041579 | LR 0.00000115
Epoch 1/1 | Step 0376/0390 | Loss 0.000368 | PosSim 0.790726 | NegSim 0.049584 | LR 0.00000108
Epoch 1/1 | Step 0377/0390 | Loss 0.000819 | PosSim 0.780169 | NegSim 0.042607 | LR 0.00000100
Epoch 1/1 | Step 0378/0390 | Loss 0.000166 | PosSim 0.789874 | NegSim 0.047037 | LR 0.00000092
Epoch 1/1 | Step 0379/0390 | Loss 0.000374 | PosSim 0.786097 | NegSim 0.047901 | LR 0.00000085
Epoch 1/1 | Step 0380/0390 | Loss 0.000244 | PosSim 0.777209 | NegSim 0.040373 | LR 0.00000077
Epoch 1/1 | Step 0381/0390 | Loss 0.000150 | PosSim 0.786374 | NegSim 0.042578 | LR 0.00000069
Epoch 1/1 | Step 0382/0390 | Loss 0.000233 | PosSim 0.781399 | NegSim 0.047069 | LR 0.00000062
Epoch 1/1 | Step 0383/0390 | Loss 0.000141 | PosSim 0.792163 | NegSim 0.044074 | LR 0.00000054
Epoch 1/1 | Step 0384/0390 | Loss 0.000162 | PosSim 0.786798 | NegSim 0.051807 | LR 0.00000046
Epoch 1/1 | Step 0385/0390 | Loss 0.000160 | PosSim 0.793996 | NegSim 0.046912 | LR 0.00000038
Epoch 1/1 | Step 0386/0390 | Loss 0.000964 | PosSim 0.785168 | NegSim 0.045562 | LR 0.00000031
Epoch 1/1 | Step 0387/0390 | Loss 0.000194 | PosSim 0.789884 | NegSim 0.050427 | LR 0.00000023
Epoch 1/1 | Step 0388/0390 | Loss 0.009028 | PosSim 0.783107 | NegSim 0.043872 | LR 0.00000015
Epoch 1/1 | Step 0389/0390 | Loss 0.000211 | PosSim 0.789679 | NegSim 0.044206 | LR 0.00000008
Epoch 1/1 | Step 0390/0390 | Loss 0.001139 | PosSim 0.779892 | NegSim 0.044533 | LR 0.00000000

----------------------------------------------------------------------
Epoch 1 summary
----------------------------------------------------------------------
Average loss: 0.004820
Average positive cosine: 0.791352
Average negative cosine: 0.068614

Checkpoint saved: experiments\reduced_25k\20260817_170650\checkpoint\epoch_1

======================================================================
TRAINING COMPLETE
======================================================================

Experiment saved to:
experiments\reduced_25k\20260817_170650

(simcse-reproduction) D:\CODING\Paper Reproduction\simcse-reproduction>python src/evaluate_sts.py --checkpoint experiments/reduced_25k/20260817_170650/checkpoint/epoch_1/training_state.pt --max-length 32

======================================================================
SIMCSE STS-B EVALUATION
======================================================================

Evaluation mode: LOCAL CHECKPOINT
Checkpoint: experiments/reduced_25k/20260817_170650/checkpoint/epoch_1/training_state.pt
Split: validation
Max length: 32
Batch size: 64
Maximum samples: None (full split)

Device: cpu

Loading checkpoint...
Checkpoint: experiments\reduced_25k\20260817_170650\checkpoint\epoch_1\training_state.pt
Checkpoint epoch: 1
Checkpoint global step: 390
Model: bert-base-uncased
MLP only during training: True

Loading tokenizer...

Loading SimCSE model...
Loading checkpoint weights...
Checkpoint weights loaded successfully.

Loading STS-B dataset...
Number of examples: 1500

First example:
Sentence 1: A man with a hard hat is dancing.
Sentence 2: A man wearing a hard hat is dancing.
Gold score: 5.0

Evaluation representation:
BERT -> raw CLS
MLP used during evaluation: False

Running evaluation...

----------------------------------------------------------------------
STS-B Spearman: 0.625932
STS-B Spearman (%): 62.59
----------------------------------------------------------------------

Evaluation complete.
