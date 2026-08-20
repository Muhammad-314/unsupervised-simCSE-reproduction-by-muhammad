# 10k Attempt UnSup SimCSE

(simcse-reproduction) D:\CODING\Paper Reproduction\simcse-reproduction>python src/train.py --max-sentences 10000 --experiment-name reduced_10k

======================================================================
SIMCSE TRAINING
======================================================================
Experiment directory: experiments\reduced_10k\20260817_151456
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
10000

Loading tokenizer...

Loading dataset...
Number of sentences: 10000
Number of batches per epoch: 156

Loading model...

Total training steps: 156
Warmup steps: 0
Initial learning rate: 3e-05

======================================================================
STARTING TRAINING
======================================================================
Epoch 1/1 | Step 0001/0156 | Loss 1.019431 | PosSim 0.852614 | NegSim 0.590278 | LR 0.00002981
Epoch 1/1 | Step 0002/0156 | Loss 0.293714 | PosSim 0.822240 | NegSim 0.411255 | LR 0.00002962
Epoch 1/1 | Step 0003/0156 | Loss 0.072543 | PosSim 0.821393 | NegSim 0.330483 | LR 0.00002942
Epoch 1/1 | Step 0004/0156 | Loss 0.090949 | PosSim 0.807575 | NegSim 0.308706 | LR 0.00002923
Epoch 1/1 | Step 0005/0156 | Loss 0.063549 | PosSim 0.798425 | NegSim 0.301457 | LR 0.00002904
Epoch 1/1 | Step 0006/0156 | Loss 0.009494 | PosSim 0.827499 | NegSim 0.293622 | LR 0.00002885
Epoch 1/1 | Step 0007/0156 | Loss 0.015888 | PosSim 0.821809 | NegSim 0.269712 | LR 0.00002865
Epoch 1/1 | Step 0008/0156 | Loss 0.047409 | PosSim 0.804568 | NegSim 0.228112 | LR 0.00002846
Epoch 1/1 | Step 0009/0156 | Loss 0.059755 | PosSim 0.818728 | NegSim 0.232684 | LR 0.00002827
Epoch 1/1 | Step 0010/0156 | Loss 0.025260 | PosSim 0.814882 | NegSim 0.191389 | LR 0.00002808
Epoch 1/1 | Step 0011/0156 | Loss 0.017172 | PosSim 0.818196 | NegSim 0.189441 | LR 0.00002788
Epoch 1/1 | Step 0012/0156 | Loss 0.028099 | PosSim 0.815582 | NegSim 0.188009 | LR 0.00002769
Epoch 1/1 | Step 0013/0156 | Loss 0.018066 | PosSim 0.812512 | NegSim 0.168828 | LR 0.00002750
Epoch 1/1 | Step 0014/0156 | Loss 0.002497 | PosSim 0.818876 | NegSim 0.169124 | LR 0.00002731
Epoch 1/1 | Step 0015/0156 | Loss 0.005167 | PosSim 0.819115 | NegSim 0.160757 | LR 0.00002712
Epoch 1/1 | Step 0016/0156 | Loss 0.011193 | PosSim 0.814434 | NegSim 0.152089 | LR 0.00002692
Epoch 1/1 | Step 0017/0156 | Loss 0.030316 | PosSim 0.827846 | NegSim 0.149534 | LR 0.00002673
Epoch 1/1 | Step 0018/0156 | Loss 0.002239 | PosSim 0.816819 | NegSim 0.135343 | LR 0.00002654
Epoch 1/1 | Step 0019/0156 | Loss 0.005971 | PosSim 0.804938 | NegSim 0.139537 | LR 0.00002635
Epoch 1/1 | Step 0020/0156 | Loss 0.023351 | PosSim 0.810734 | NegSim 0.129104 | LR 0.00002615
Epoch 1/1 | Step 0021/0156 | Loss 0.030099 | PosSim 0.814756 | NegSim 0.125943 | LR 0.00002596
Epoch 1/1 | Step 0022/0156 | Loss 0.002671 | PosSim 0.816473 | NegSim 0.124654 | LR 0.00002577
Epoch 1/1 | Step 0023/0156 | Loss 0.002441 | PosSim 0.807512 | NegSim 0.116317 | LR 0.00002558
Epoch 1/1 | Step 0024/0156 | Loss 0.001791 | PosSim 0.809214 | NegSim 0.111842 | LR 0.00002538
Epoch 1/1 | Step 0025/0156 | Loss 0.002571 | PosSim 0.805987 | NegSim 0.123869 | LR 0.00002519
Epoch 1/1 | Step 0026/0156 | Loss 0.002633 | PosSim 0.809188 | NegSim 0.109626 | LR 0.00002500
Epoch 1/1 | Step 0027/0156 | Loss 0.001034 | PosSim 0.814512 | NegSim 0.122792 | LR 0.00002481
Epoch 1/1 | Step 0028/0156 | Loss 0.002752 | PosSim 0.812688 | NegSim 0.099463 | LR 0.00002462
Epoch 1/1 | Step 0029/0156 | Loss 0.002354 | PosSim 0.815808 | NegSim 0.108401 | LR 0.00002442
Epoch 1/1 | Step 0030/0156 | Loss 0.001497 | PosSim 0.819571 | NegSim 0.118222 | LR 0.00002423
Epoch 1/1 | Step 0031/0156 | Loss 0.033808 | PosSim 0.811963 | NegSim 0.097558 | LR 0.00002404
Epoch 1/1 | Step 0032/0156 | Loss 0.001203 | PosSim 0.822668 | NegSim 0.111157 | LR 0.00002385
Epoch 1/1 | Step 0033/0156 | Loss 0.017450 | PosSim 0.800309 | NegSim 0.101830 | LR 0.00002365
Epoch 1/1 | Step 0034/0156 | Loss 0.001067 | PosSim 0.807634 | NegSim 0.090353 | LR 0.00002346
Epoch 1/1 | Step 0035/0156 | Loss 0.001555 | PosSim 0.816344 | NegSim 0.097386 | LR 0.00002327
Epoch 1/1 | Step 0036/0156 | Loss 0.001310 | PosSim 0.807517 | NegSim 0.102968 | LR 0.00002308
Epoch 1/1 | Step 0037/0156 | Loss 0.002765 | PosSim 0.805563 | NegSim 0.095862 | LR 0.00002288
Epoch 1/1 | Step 0038/0156 | Loss 0.006939 | PosSim 0.811753 | NegSim 0.095385 | LR 0.00002269
Epoch 1/1 | Step 0039/0156 | Loss 0.009978 | PosSim 0.805899 | NegSim 0.092531 | LR 0.00002250
Epoch 1/1 | Step 0040/0156 | Loss 0.003137 | PosSim 0.797097 | NegSim 0.088561 | LR 0.00002231
Epoch 1/1 | Step 0041/0156 | Loss 0.001679 | PosSim 0.803655 | NegSim 0.093798 | LR 0.00002212
Epoch 1/1 | Step 0042/0156 | Loss 0.000855 | PosSim 0.808639 | NegSim 0.087102 | LR 0.00002192
Epoch 1/1 | Step 0043/0156 | Loss 0.002207 | PosSim 0.805821 | NegSim 0.092159 | LR 0.00002173
Epoch 1/1 | Step 0044/0156 | Loss 0.000356 | PosSim 0.804904 | NegSim 0.086171 | LR 0.00002154
Epoch 1/1 | Step 0045/0156 | Loss 0.003895 | PosSim 0.802897 | NegSim 0.087952 | LR 0.00002135
Epoch 1/1 | Step 0046/0156 | Loss 0.001056 | PosSim 0.808287 | NegSim 0.083657 | LR 0.00002115
Epoch 1/1 | Step 0047/0156 | Loss 0.001864 | PosSim 0.804468 | NegSim 0.084643 | LR 0.00002096
Epoch 1/1 | Step 0048/0156 | Loss 0.003808 | PosSim 0.793708 | NegSim 0.083778 | LR 0.00002077
Epoch 1/1 | Step 0049/0156 | Loss 0.050489 | PosSim 0.806185 | NegSim 0.084822 | LR 0.00002058
Epoch 1/1 | Step 0050/0156 | Loss 0.001053 | PosSim 0.806736 | NegSim 0.086339 | LR 0.00002038
Epoch 1/1 | Step 0051/0156 | Loss 0.012599 | PosSim 0.797860 | NegSim 0.084452 | LR 0.00002019
Epoch 1/1 | Step 0052/0156 | Loss 0.001872 | PosSim 0.789218 | NegSim 0.085384 | LR 0.00002000
Epoch 1/1 | Step 0053/0156 | Loss 0.000501 | PosSim 0.814246 | NegSim 0.082211 | LR 0.00001981
Epoch 1/1 | Step 0054/0156 | Loss 0.001992 | PosSim 0.793899 | NegSim 0.074020 | LR 0.00001962
Epoch 1/1 | Step 0055/0156 | Loss 0.006064 | PosSim 0.786976 | NegSim 0.077667 | LR 0.00001942
Epoch 1/1 | Step 0056/0156 | Loss 0.001729 | PosSim 0.801005 | NegSim 0.081790 | LR 0.00001923
Epoch 1/1 | Step 0057/0156 | Loss 0.000879 | PosSim 0.806753 | NegSim 0.081889 | LR 0.00001904
Epoch 1/1 | Step 0058/0156 | Loss 0.005388 | PosSim 0.801319 | NegSim 0.076079 | LR 0.00001885
Epoch 1/1 | Step 0059/0156 | Loss 0.001800 | PosSim 0.794090 | NegSim 0.086646 | LR 0.00001865
Epoch 1/1 | Step 0060/0156 | Loss 0.008402 | PosSim 0.796885 | NegSim 0.074598 | LR 0.00001846
Epoch 1/1 | Step 0061/0156 | Loss 0.001675 | PosSim 0.805804 | NegSim 0.083856 | LR 0.00001827
Epoch 1/1 | Step 0062/0156 | Loss 0.000726 | PosSim 0.806458 | NegSim 0.075954 | LR 0.00001808
Epoch 1/1 | Step 0063/0156 | Loss 0.000924 | PosSim 0.799452 | NegSim 0.080438 | LR 0.00001788
Epoch 1/1 | Step 0064/0156 | Loss 0.004565 | PosSim 0.806775 | NegSim 0.073945 | LR 0.00001769
Epoch 1/1 | Step 0065/0156 | Loss 0.010198 | PosSim 0.808682 | NegSim 0.080941 | LR 0.00001750
Epoch 1/1 | Step 0066/0156 | Loss 0.000300 | PosSim 0.802238 | NegSim 0.079583 | LR 0.00001731
Epoch 1/1 | Step 0067/0156 | Loss 0.000592 | PosSim 0.800249 | NegSim 0.076882 | LR 0.00001712
Epoch 1/1 | Step 0068/0156 | Loss 0.001096 | PosSim 0.799699 | NegSim 0.075574 | LR 0.00001692
Epoch 1/1 | Step 0069/0156 | Loss 0.008478 | PosSim 0.797984 | NegSim 0.078515 | LR 0.00001673
Epoch 1/1 | Step 0070/0156 | Loss 0.002994 | PosSim 0.804482 | NegSim 0.077808 | LR 0.00001654
Epoch 1/1 | Step 0071/0156 | Loss 0.000997 | PosSim 0.792090 | NegSim 0.079465 | LR 0.00001635
Epoch 1/1 | Step 0072/0156 | Loss 0.001991 | PosSim 0.800025 | NegSim 0.075613 | LR 0.00001615
Epoch 1/1 | Step 0073/0156 | Loss 0.001889 | PosSim 0.795262 | NegSim 0.076788 | LR 0.00001596
Epoch 1/1 | Step 0074/0156 | Loss 0.000702 | PosSim 0.803375 | NegSim 0.078092 | LR 0.00001577
Epoch 1/1 | Step 0075/0156 | Loss 0.000904 | PosSim 0.792014 | NegSim 0.068867 | LR 0.00001558
Epoch 1/1 | Step 0076/0156 | Loss 0.001601 | PosSim 0.790346 | NegSim 0.080784 | LR 0.00001538
Epoch 1/1 | Step 0077/0156 | Loss 0.000545 | PosSim 0.799600 | NegSim 0.071311 | LR 0.00001519
Epoch 1/1 | Step 0078/0156 | Loss 0.000899 | PosSim 0.793179 | NegSim 0.067115 | LR 0.00001500
Epoch 1/1 | Step 0079/0156 | Loss 0.001824 | PosSim 0.789727 | NegSim 0.072928 | LR 0.00001481
Epoch 1/1 | Step 0080/0156 | Loss 0.003364 | PosSim 0.792307 | NegSim 0.071723 | LR 0.00001462
Epoch 1/1 | Step 0081/0156 | Loss 0.002344 | PosSim 0.784061 | NegSim 0.069913 | LR 0.00001442
Epoch 1/1 | Step 0082/0156 | Loss 0.002295 | PosSim 0.794589 | NegSim 0.073528 | LR 0.00001423
Epoch 1/1 | Step 0083/0156 | Loss 0.000321 | PosSim 0.791076 | NegSim 0.072078 | LR 0.00001404
Epoch 1/1 | Step 0084/0156 | Loss 0.001417 | PosSim 0.789137 | NegSim 0.074828 | LR 0.00001385
Epoch 1/1 | Step 0085/0156 | Loss 0.000720 | PosSim 0.791663 | NegSim 0.075511 | LR 0.00001365
Epoch 1/1 | Step 0086/0156 | Loss 0.001101 | PosSim 0.795024 | NegSim 0.066939 | LR 0.00001346
Epoch 1/1 | Step 0087/0156 | Loss 0.000645 | PosSim 0.793726 | NegSim 0.074223 | LR 0.00001327
Epoch 1/1 | Step 0088/0156 | Loss 0.013231 | PosSim 0.786540 | NegSim 0.066982 | LR 0.00001308
Epoch 1/1 | Step 0089/0156 | Loss 0.001282 | PosSim 0.788239 | NegSim 0.068106 | LR 0.00001288
Epoch 1/1 | Step 0090/0156 | Loss 0.000545 | PosSim 0.791055 | NegSim 0.071593 | LR 0.00001269
Epoch 1/1 | Step 0091/0156 | Loss 0.000679 | PosSim 0.793456 | NegSim 0.070896 | LR 0.00001250
Epoch 1/1 | Step 0092/0156 | Loss 0.000937 | PosSim 0.793720 | NegSim 0.068441 | LR 0.00001231
Epoch 1/1 | Step 0093/0156 | Loss 0.026587 | PosSim 0.792481 | NegSim 0.073784 | LR 0.00001212
Epoch 1/1 | Step 0094/0156 | Loss 0.005353 | PosSim 0.779159 | NegSim 0.072074 | LR 0.00001192
Epoch 1/1 | Step 0095/0156 | Loss 0.000379 | PosSim 0.788852 | NegSim 0.063588 | LR 0.00001173
Epoch 1/1 | Step 0096/0156 | Loss 0.000950 | PosSim 0.783104 | NegSim 0.061471 | LR 0.00001154
Epoch 1/1 | Step 0097/0156 | Loss 0.002420 | PosSim 0.797383 | NegSim 0.076357 | LR 0.00001135
Epoch 1/1 | Step 0098/0156 | Loss 0.000239 | PosSim 0.789720 | NegSim 0.063116 | LR 0.00001115
Epoch 1/1 | Step 0099/0156 | Loss 0.000698 | PosSim 0.787418 | NegSim 0.061677 | LR 0.00001096
Epoch 1/1 | Step 0100/0156 | Loss 0.000971 | PosSim 0.790598 | NegSim 0.067721 | LR 0.00001077
Epoch 1/1 | Step 0101/0156 | Loss 0.000978 | PosSim 0.787489 | NegSim 0.071805 | LR 0.00001058
Epoch 1/1 | Step 0102/0156 | Loss 0.001537 | PosSim 0.781770 | NegSim 0.071074 | LR 0.00001038
Epoch 1/1 | Step 0103/0156 | Loss 0.001047 | PosSim 0.793203 | NegSim 0.067033 | LR 0.00001019
Epoch 1/1 | Step 0104/0156 | Loss 0.001374 | PosSim 0.795720 | NegSim 0.062628 | LR 0.00001000
Epoch 1/1 | Step 0105/0156 | Loss 0.024604 | PosSim 0.787448 | NegSim 0.055768 | LR 0.00000981
Epoch 1/1 | Step 0106/0156 | Loss 0.016589 | PosSim 0.786155 | NegSim 0.057556 | LR 0.00000962
Epoch 1/1 | Step 0107/0156 | Loss 0.000476 | PosSim 0.792868 | NegSim 0.066141 | LR 0.00000942
Epoch 1/1 | Step 0108/0156 | Loss 0.000419 | PosSim 0.786361 | NegSim 0.062149 | LR 0.00000923
Epoch 1/1 | Step 0109/0156 | Loss 0.002146 | PosSim 0.789419 | NegSim 0.062881 | LR 0.00000904
Epoch 1/1 | Step 0110/0156 | Loss 0.000610 | PosSim 0.791111 | NegSim 0.071534 | LR 0.00000885
Epoch 1/1 | Step 0111/0156 | Loss 0.000462 | PosSim 0.780595 | NegSim 0.062699 | LR 0.00000865
Epoch 1/1 | Step 0112/0156 | Loss 0.001077 | PosSim 0.792363 | NegSim 0.063380 | LR 0.00000846
Epoch 1/1 | Step 0113/0156 | Loss 0.000642 | PosSim 0.785571 | NegSim 0.063613 | LR 0.00000827
Epoch 1/1 | Step 0114/0156 | Loss 0.001308 | PosSim 0.787040 | NegSim 0.066825 | LR 0.00000808
Epoch 1/1 | Step 0115/0156 | Loss 0.000576 | PosSim 0.786729 | NegSim 0.069446 | LR 0.00000788
Epoch 1/1 | Step 0116/0156 | Loss 0.021732 | PosSim 0.776013 | NegSim 0.061774 | LR 0.00000769
Epoch 1/1 | Step 0117/0156 | Loss 0.000524 | PosSim 0.776640 | NegSim 0.064413 | LR 0.00000750
Epoch 1/1 | Step 0118/0156 | Loss 0.001078 | PosSim 0.793321 | NegSim 0.062050 | LR 0.00000731
Epoch 1/1 | Step 0119/0156 | Loss 0.003905 | PosSim 0.780149 | NegSim 0.068522 | LR 0.00000712
Epoch 1/1 | Step 0120/0156 | Loss 0.009280 | PosSim 0.783921 | NegSim 0.067160 | LR 0.00000692
Epoch 1/1 | Step 0121/0156 | Loss 0.000767 | PosSim 0.782432 | NegSim 0.062742 | LR 0.00000673
Epoch 1/1 | Step 0122/0156 | Loss 0.006717 | PosSim 0.778943 | NegSim 0.059311 | LR 0.00000654
Epoch 1/1 | Step 0123/0156 | Loss 0.001102 | PosSim 0.782022 | NegSim 0.060347 | LR 0.00000635
Epoch 1/1 | Step 0124/0156 | Loss 0.026224 | PosSim 0.774992 | NegSim 0.064856 | LR 0.00000615
Epoch 1/1 | Step 0125/0156 | Loss 0.001101 | PosSim 0.788350 | NegSim 0.065452 | LR 0.00000596
Epoch 1/1 | Step 0126/0156 | Loss 0.001361 | PosSim 0.783783 | NegSim 0.057956 | LR 0.00000577
Epoch 1/1 | Step 0127/0156 | Loss 0.001094 | PosSim 0.783219 | NegSim 0.061016 | LR 0.00000558
Epoch 1/1 | Step 0128/0156 | Loss 0.022612 | PosSim 0.782917 | NegSim 0.064375 | LR 0.00000538
Epoch 1/1 | Step 0129/0156 | Loss 0.000647 | PosSim 0.779743 | NegSim 0.068722 | LR 0.00000519
Epoch 1/1 | Step 0130/0156 | Loss 0.026211 | PosSim 0.778231 | NegSim 0.060993 | LR 0.00000500
Epoch 1/1 | Step 0131/0156 | Loss 0.008137 | PosSim 0.777858 | NegSim 0.063826 | LR 0.00000481
Epoch 1/1 | Step 0132/0156 | Loss 0.001113 | PosSim 0.784216 | NegSim 0.066587 | LR 0.00000462
Epoch 1/1 | Step 0133/0156 | Loss 0.000965 | PosSim 0.780999 | NegSim 0.061245 | LR 0.00000442
Epoch 1/1 | Step 0134/0156 | Loss 0.005704 | PosSim 0.784425 | NegSim 0.068061 | LR 0.00000423
Epoch 1/1 | Step 0135/0156 | Loss 0.002620 | PosSim 0.781775 | NegSim 0.066316 | LR 0.00000404
Epoch 1/1 | Step 0136/0156 | Loss 0.000330 | PosSim 0.795225 | NegSim 0.062253 | LR 0.00000385
Epoch 1/1 | Step 0137/0156 | Loss 0.000658 | PosSim 0.790660 | NegSim 0.066382 | LR 0.00000365
Epoch 1/1 | Step 0138/0156 | Loss 0.000642 | PosSim 0.801844 | NegSim 0.068679 | LR 0.00000346
Epoch 1/1 | Step 0139/0156 | Loss 0.000449 | PosSim 0.796135 | NegSim 0.068017 | LR 0.00000327
Epoch 1/1 | Step 0140/0156 | Loss 0.000397 | PosSim 0.786548 | NegSim 0.064890 | LR 0.00000308
Epoch 1/1 | Step 0141/0156 | Loss 0.000482 | PosSim 0.785170 | NegSim 0.061307 | LR 0.00000288
Epoch 1/1 | Step 0142/0156 | Loss 0.001180 | PosSim 0.779199 | NegSim 0.068238 | LR 0.00000269
Epoch 1/1 | Step 0143/0156 | Loss 0.005236 | PosSim 0.786772 | NegSim 0.064708 | LR 0.00000250
Epoch 1/1 | Step 0144/0156 | Loss 0.000577 | PosSim 0.787422 | NegSim 0.059428 | LR 0.00000231
Epoch 1/1 | Step 0145/0156 | Loss 0.000420 | PosSim 0.783095 | NegSim 0.066633 | LR 0.00000212
Epoch 1/1 | Step 0146/0156 | Loss 0.001263 | PosSim 0.777965 | NegSim 0.066845 | LR 0.00000192
Epoch 1/1 | Step 0147/0156 | Loss 0.000961 | PosSim 0.791853 | NegSim 0.067488 | LR 0.00000173
Epoch 1/1 | Step 0148/0156 | Loss 0.000936 | PosSim 0.795342 | NegSim 0.073983 | LR 0.00000154
Epoch 1/1 | Step 0149/0156 | Loss 0.002128 | PosSim 0.771768 | NegSim 0.062391 | LR 0.00000135
Epoch 1/1 | Step 0150/0156 | Loss 0.001162 | PosSim 0.784584 | NegSim 0.066048 | LR 0.00000115
Epoch 1/1 | Step 0151/0156 | Loss 0.000530 | PosSim 0.789072 | NegSim 0.065829 | LR 0.00000096
Epoch 1/1 | Step 0152/0156 | Loss 0.000856 | PosSim 0.784089 | NegSim 0.061999 | LR 0.00000077
Epoch 1/1 | Step 0153/0156 | Loss 0.000612 | PosSim 0.789476 | NegSim 0.068790 | LR 0.00000058
Epoch 1/1 | Step 0154/0156 | Loss 0.000566 | PosSim 0.778564 | NegSim 0.065231 | LR 0.00000038
Epoch 1/1 | Step 0155/0156 | Loss 0.000332 | PosSim 0.798553 | NegSim 0.062818 | LR 0.00000019
Epoch 1/1 | Step 0156/0156 | Loss 0.000496 | PosSim 0.791377 | NegSim 0.068521 | LR 0.00000000

----------------------------------------------------------------------
Epoch 1 summary
----------------------------------------------------------------------
Average loss: 0.015423
Average positive cosine: 0.797257
Average negative cosine: 0.096789

Checkpoint saved: experiments\reduced_10k\20260817_151456\checkpoint\epoch_1

======================================================================
TRAINING COMPLETE
======================================================================

Experiment saved to:
experiments\reduced_10k\20260817_151456

# Evaluation STS

(simcse-reproduction) D:\CODING\Paper Reproduction\simcse-reproduction>python src/evaluate_sts.py --checkpoint experiments/reduced_10k/20260817_151456/checkpoint/epoch_1/training_state.pt --max-length 32

======================================================================
SIMCSE STS-B EVALUATION
======================================================================

Evaluation mode: LOCAL CHECKPOINT
Checkpoint: experiments/reduced_10k/20260817_151456/checkpoint/epoch_1/training_state.pt
Split: validation
Max length: 32
Batch size: 64
Maximum samples: None (full split)

Device: cpu

Loading checkpoint...
Checkpoint: experiments\reduced_10k\20260817_151456\checkpoint\epoch_1\training_state.pt
Checkpoint epoch: 1
Checkpoint global step: 156
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
STS-B Spearman: 0.535533
STS-B Spearman (%): 53.55
----------------------------------------------------------------------

Evaluation complete.
