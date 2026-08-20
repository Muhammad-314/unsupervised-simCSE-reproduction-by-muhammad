# UnSup SimCSE full 1M training

(simcse-reproduction) D:\CODING\Paper Reproduction\simcse-reproduction>python src/train.py --experiment-name unsup_bert_base_full
======================================================================
SIMCSE TRAINING
======================================================================
Experiment directory: experiments\unsup_bert_base_full\20260817_023328
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

Loading tokenizer...

Loading dataset...
Number of sentences: 1000000
Number of batches per epoch: 15625

Loading model...

Total training steps: 15625
Warmup steps: 0
Initial learning rate: 3e-05

======================================================================
STARTING TRAINING
======================================================================
Epoch 1/1 | Step 0001/15625 | Loss 0.867656 | PosSim 0.838664 | NegSim 0.556529 | LR 0.00003000
Epoch 1/1 | Step 0002/15625 | Loss 0.138821 | PosSim 0.840593 | NegSim 0.416851 | LR 0.00003000
Epoch 1/1 | Step 0003/15625 | Loss 0.041478 | PosSim 0.827709 | NegSim 0.371523 | LR 0.00002999
Epoch 1/1 | Step 0004/15625 | Loss 0.025968 | PosSim 0.823105 | NegSim 0.327205 | LR 0.00002999
Epoch 1/1 | Step 0005/15625 | Loss 0.017177 | PosSim 0.829870 | NegSim 0.295896 | LR 0.00002999
Epoch 1/1 | Step 0006/15625 | Loss 0.006478 | PosSim 0.826065 | NegSim 0.284961 | LR 0.00002999
Epoch 1/1 | Step 0007/15625 | Loss 0.006337 | PosSim 0.820503 | NegSim 0.257141 | LR 0.00002999
Epoch 1/1 | Step 0008/15625 | Loss 0.010266 | PosSim 0.824123 | NegSim 0.232691 | LR 0.00002998
Epoch 1/1 | Step 0009/15625 | Loss 0.012383 | PosSim 0.821895 | NegSim 0.209344 | LR 0.00002998
Epoch 1/1 | Step 0010/15625 | Loss 0.013141 | PosSim 0.823605 | NegSim 0.190622 | LR 0.00002998
Epoch 1/1 | Step 0011/15625 | Loss 0.001482 | PosSim 0.817738 | NegSim 0.178380 | LR 0.00002998
Epoch 1/1 | Step 0012/15625 | Loss 0.002121 | PosSim 0.821136 | NegSim 0.175083 | LR 0.00002998
Epoch 1/1 | Step 0013/15625 | Loss 0.001469 | PosSim 0.824351 | NegSim 0.154523 | LR 0.00002998
Epoch 1/1 | Step 0014/15625 | Loss 0.001163 | PosSim 0.829849 | NegSim 0.153157 | LR 0.00002997
Epoch 1/1 | Step 0015/15625 | Loss 0.004682 | PosSim 0.829119 | NegSim 0.147484 | LR 0.00002997
Epoch 1/1 | Step 0016/15625 | Loss 0.002173 | PosSim 0.825898 | NegSim 0.137893 | LR 0.00002997
Epoch 1/1 | Step 0017/15625 | Loss 0.001505 | PosSim 0.823437 | NegSim 0.133036 | LR 0.00002997
Epoch 1/1 | Step 0018/15625 | Loss 0.002708 | PosSim 0.815928 | NegSim 0.131477 | LR 0.00002997
Epoch 1/1 | Step 0019/15625 | Loss 0.001072 | PosSim 0.821640 | NegSim 0.131170 | LR 0.00002996
Epoch 1/1 | Step 0020/15625 | Loss 0.000868 | PosSim 0.828614 | NegSim 0.118223 | LR 0.00002996
Epoch 1/1 | Step 0021/15625 | Loss 0.001129 | PosSim 0.825709 | NegSim 0.121250 | LR 0.00002996
Epoch 1/1 | Step 0022/15625 | Loss 0.009235 | PosSim 0.818354 | NegSim 0.120068 | LR 0.00002996
Epoch 1/1 | Step 0023/15625 | Loss 0.002346 | PosSim 0.826460 | NegSim 0.104892 | LR 0.00002996
Epoch 1/1 | Step 0024/15625 | Loss 0.001386 | PosSim 0.821644 | NegSim 0.107754 | LR 0.00002995
Epoch 1/1 | Step 0025/15625 | Loss 0.000638 | PosSim 0.820284 | NegSim 0.101721 | LR 0.00002995
Epoch 1/1 | Step 0026/15625 | Loss 0.002109 | PosSim 0.818382 | NegSim 0.106658 | LR 0.00002995
Epoch 1/1 | Step 0027/15625 | Loss 0.005173 | PosSim 0.812603 | NegSim 0.100163 | LR 0.00002995
Epoch 1/1 | Step 0028/15625 | Loss 0.000873 | PosSim 0.826829 | NegSim 0.109110 | LR 0.00002995
Epoch 1/1 | Step 0029/15625 | Loss 0.000769 | PosSim 0.820667 | NegSim 0.090553 | LR 0.00002994
Epoch 1/1 | Step 0030/15625 | Loss 0.000455 | PosSim 0.818535 | NegSim 0.098549 | LR 0.00002994
Epoch 1/1 | Step 0031/15625 | Loss 0.003974 | PosSim 0.818687 | NegSim 0.087312 | LR 0.00002994
Epoch 1/1 | Step 0032/15625 | Loss 0.020031 | PosSim 0.822013 | NegSim 0.100400 | LR 0.00002994
Epoch 1/1 | Step 0033/15625 | Loss 0.000710 | PosSim 0.816791 | NegSim 0.094365 | LR 0.00002994
Epoch 1/1 | Step 0034/15625 | Loss 0.000673 | PosSim 0.825044 | NegSim 0.100263 | LR 0.00002993
Epoch 1/1 | Step 0035/15625 | Loss 0.001185 | PosSim 0.822959 | NegSim 0.084499 | LR 0.00002993
Epoch 1/1 | Step 0036/15625 | Loss 0.000878 | PosSim 0.813661 | NegSim 0.088330 | LR 0.00002993
Epoch 1/1 | Step 0037/15625 | Loss 0.001272 | PosSim 0.810226 | NegSim 0.089945 | LR 0.00002993
Epoch 1/1 | Step 0038/15625 | Loss 0.001066 | PosSim 0.829406 | NegSim 0.089503 | LR 0.00002993
Epoch 1/1 | Step 0039/15625 | Loss 0.000829 | PosSim 0.829284 | NegSim 0.091324 | LR 0.00002993
Epoch 1/1 | Step 0040/15625 | Loss 0.000385 | PosSim 0.820552 | NegSim 0.083959 | LR 0.00002992
Epoch 1/1 | Step 0041/15625 | Loss 0.000776 | PosSim 0.816713 | NegSim 0.091102 | LR 0.00002992
Epoch 1/1 | Step 0042/15625 | Loss 0.000534 | PosSim 0.811568 | NegSim 0.083936 | LR 0.00002992
Epoch 1/1 | Step 0043/15625 | Loss 0.000382 | PosSim 0.826297 | NegSim 0.080213 | LR 0.00002992
Epoch 1/1 | Step 0044/15625 | Loss 0.000867 | PosSim 0.817602 | NegSim 0.093968 | LR 0.00002992
Epoch 1/1 | Step 0045/15625 | Loss 0.000373 | PosSim 0.824885 | NegSim 0.089360 | LR 0.00002991
Epoch 1/1 | Step 0046/15625 | Loss 0.000520 | PosSim 0.817670 | NegSim 0.075550 | LR 0.00002991
Epoch 1/1 | Step 0047/15625 | Loss 0.001320 | PosSim 0.819314 | NegSim 0.083452 | LR 0.00002991
Epoch 1/1 | Step 0048/15625 | Loss 0.000580 | PosSim 0.819133 | NegSim 0.085115 | LR 0.00002991
Epoch 1/1 | Step 0049/15625 | Loss 0.000488 | PosSim 0.822918 | NegSim 0.068085 | LR 0.00002991
Epoch 1/1 | Step 0050/15625 | Loss 0.000573 | PosSim 0.819216 | NegSim 0.072234 | LR 0.00002990
Epoch 1/1 | Step 0051/15625 | Loss 0.000944 | PosSim 0.817657 | NegSim 0.077210 | LR 0.00002990
Epoch 1/1 | Step 0052/15625 | Loss 0.000465 | PosSim 0.829195 | NegSim 0.071979 | LR 0.00002990
Epoch 1/1 | Step 0053/15625 | Loss 0.000738 | PosSim 0.816991 | NegSim 0.077427 | LR 0.00002990
Epoch 1/1 | Step 0054/15625 | Loss 0.003780 | PosSim 0.803169 | NegSim 0.069754 | LR 0.00002990
Epoch 1/1 | Step 0055/15625 | Loss 0.000800 | PosSim 0.816818 | NegSim 0.069476 | LR 0.00002989
Epoch 1/1 | Step 0056/15625 | Loss 0.000434 | PosSim 0.815180 | NegSim 0.068060 | LR 0.00002989
Epoch 1/1 | Step 0057/15625 | Loss 0.000316 | PosSim 0.812578 | NegSim 0.068802 | LR 0.00002989
Epoch 1/1 | Step 0058/15625 | Loss 0.000410 | PosSim 0.813044 | NegSim 0.064566 | LR 0.00002989
Epoch 1/1 | Step 0059/15625 | Loss 0.002307 | PosSim 0.814632 | NegSim 0.069640 | LR 0.00002989
Epoch 1/1 | Step 0060/15625 | Loss 0.000707 | PosSim 0.827208 | NegSim 0.064609 | LR 0.00002988
Epoch 1/1 | Step 0061/15625 | Loss 0.001070 | PosSim 0.809110 | NegSim 0.066373 | LR 0.00002988
Epoch 1/1 | Step 0062/15625 | Loss 0.000323 | PosSim 0.806114 | NegSim 0.065256 | LR 0.00002988
Epoch 1/1 | Step 0063/15625 | Loss 0.000438 | PosSim 0.816115 | NegSim 0.071647 | LR 0.00002988
Epoch 1/1 | Step 0064/15625 | Loss 0.000280 | PosSim 0.819920 | NegSim 0.063546 | LR 0.00002988
Epoch 1/1 | Step 0065/15625 | Loss 0.006784 | PosSim 0.812713 | NegSim 0.061582 | LR 0.00002988
Epoch 1/1 | Step 0066/15625 | Loss 0.000508 | PosSim 0.815856 | NegSim 0.068206 | LR 0.00002987
Epoch 1/1 | Step 0067/15625 | Loss 0.001307 | PosSim 0.824537 | NegSim 0.071435 | LR 0.00002987
Epoch 1/1 | Step 0068/15625 | Loss 0.000407 | PosSim 0.811630 | NegSim 0.059278 | LR 0.00002987
Epoch 1/1 | Step 0069/15625 | Loss 0.004175 | PosSim 0.805326 | NegSim 0.058680 | LR 0.00002987
Epoch 1/1 | Step 0070/15625 | Loss 0.000641 | PosSim 0.816227 | NegSim 0.058751 | LR 0.00002987
Epoch 1/1 | Step 0071/15625 | Loss 0.000701 | PosSim 0.809834 | NegSim 0.063159 | LR 0.00002986
Epoch 1/1 | Step 0072/15625 | Loss 0.000360 | PosSim 0.803467 | NegSim 0.058189 | LR 0.00002986
Epoch 1/1 | Step 0073/15625 | Loss 0.000234 | PosSim 0.814693 | NegSim 0.054319 | LR 0.00002986
Epoch 1/1 | Step 0074/15625 | Loss 0.000224 | PosSim 0.808004 | NegSim 0.057265 | LR 0.00002986
Epoch 1/1 | Step 0075/15625 | Loss 0.000633 | PosSim 0.809652 | NegSim 0.059557 | LR 0.00002986
Epoch 1/1 | Step 0076/15625 | Loss 0.000196 | PosSim 0.810439 | NegSim 0.062361 | LR 0.00002985
Epoch 1/1 | Step 0077/15625 | Loss 0.000810 | PosSim 0.798546 | NegSim 0.056473 | LR 0.00002985
Epoch 1/1 | Step 0078/15625 | Loss 0.000581 | PosSim 0.805768 | NegSim 0.054897 | LR 0.00002985
Epoch 1/1 | Step 0079/15625 | Loss 0.001108 | PosSim 0.812876 | NegSim 0.063392 | LR 0.00002985
Epoch 1/1 | Step 0080/15625 | Loss 0.000369 | PosSim 0.813703 | NegSim 0.062455 | LR 0.00002985
Epoch 1/1 | Step 0081/15625 | Loss 0.000475 | PosSim 0.819079 | NegSim 0.058854 | LR 0.00002984
Epoch 1/1 | Step 0082/15625 | Loss 0.000278 | PosSim 0.810641 | NegSim 0.050503 | LR 0.00002984
Epoch 1/1 | Step 0083/15625 | Loss 0.000432 | PosSim 0.806172 | NegSim 0.063047 | LR 0.00002984
Epoch 1/1 | Step 0084/15625 | Loss 0.000231 | PosSim 0.809912 | NegSim 0.059271 | LR 0.00002984
Epoch 1/1 | Step 0085/15625 | Loss 0.000201 | PosSim 0.809279 | NegSim 0.061316 | LR 0.00002984
Epoch 1/1 | Step 0086/15625 | Loss 0.002290 | PosSim 0.797439 | NegSim 0.059918 | LR 0.00002983
Epoch 1/1 | Step 0087/15625 | Loss 0.000858 | PosSim 0.814929 | NegSim 0.059352 | LR 0.00002983
Epoch 1/1 | Step 0088/15625 | Loss 0.000871 | PosSim 0.801783 | NegSim 0.054268 | LR 0.00002983
Epoch 1/1 | Step 0089/15625 | Loss 0.001009 | PosSim 0.805961 | NegSim 0.052062 | LR 0.00002983
Epoch 1/1 | Step 0090/15625 | Loss 0.007952 | PosSim 0.807807 | NegSim 0.057833 | LR 0.00002983
Epoch 1/1 | Step 0091/15625 | Loss 0.000264 | PosSim 0.807319 | NegSim 0.055067 | LR 0.00002983
Epoch 1/1 | Step 0092/15625 | Loss 0.001690 | PosSim 0.802094 | NegSim 0.057173 | LR 0.00002982
Epoch 1/1 | Step 0093/15625 | Loss 0.000268 | PosSim 0.806166 | NegSim 0.055813 | LR 0.00002982
Epoch 1/1 | Step 0094/15625 | Loss 0.000756 | PosSim 0.807272 | NegSim 0.057050 | LR 0.00002982
Epoch 1/1 | Step 0095/15625 | Loss 0.000360 | PosSim 0.810824 | NegSim 0.060377 | LR 0.00002982
Epoch 1/1 | Step 0096/15625 | Loss 0.000639 | PosSim 0.805668 | NegSim 0.056454 | LR 0.00002982
Epoch 1/1 | Step 0097/15625 | Loss 0.001140 | PosSim 0.812921 | NegSim 0.062141 | LR 0.00002981
Epoch 1/1 | Step 0098/15625 | Loss 0.000563 | PosSim 0.805430 | NegSim 0.060442 | LR 0.00002981
Epoch 1/1 | Step 0099/15625 | Loss 0.000498 | PosSim 0.814723 | NegSim 0.055720 | LR 0.00002981
Epoch 1/1 | Step 0100/15625 | Loss 0.001621 | PosSim 0.812599 | NegSim 0.055811 | LR 0.00002981
Epoch 1/1 | Step 0101/15625 | Loss 0.000225 | PosSim 0.816916 | NegSim 0.059868 | LR 0.00002981
Epoch 1/1 | Step 0102/15625 | Loss 0.000774 | PosSim 0.805075 | NegSim 0.060314 | LR 0.00002980
Epoch 1/1 | Step 0103/15625 | Loss 0.026408 | PosSim 0.815396 | NegSim 0.053548 | LR 0.00002980
Epoch 1/1 | Step 0104/15625 | Loss 0.000603 | PosSim 0.811812 | NegSim 0.059664 | LR 0.00002980
Epoch 1/1 | Step 0105/15625 | Loss 0.000323 | PosSim 0.811804 | NegSim 0.056528 | LR 0.00002980
Epoch 1/1 | Step 0106/15625 | Loss 0.000289 | PosSim 0.822018 | NegSim 0.063684 | LR 0.00002980
Epoch 1/1 | Step 0107/15625 | Loss 0.000224 | PosSim 0.811709 | NegSim 0.065653 | LR 0.00002979
Epoch 1/1 | Step 0108/15625 | Loss 0.000724 | PosSim 0.816525 | NegSim 0.061387 | LR 0.00002979
Epoch 1/1 | Step 0109/15625 | Loss 0.000356 | PosSim 0.811018 | NegSim 0.061232 | LR 0.00002979
Epoch 1/1 | Step 0110/15625 | Loss 0.000605 | PosSim 0.809575 | NegSim 0.054515 | LR 0.00002979
Epoch 1/1 | Step 0111/15625 | Loss 0.000371 | PosSim 0.814037 | NegSim 0.060079 | LR 0.00002979
Epoch 1/1 | Step 0112/15625 | Loss 0.000355 | PosSim 0.808759 | NegSim 0.069560 | LR 0.00002978
Epoch 1/1 | Step 0113/15625 | Loss 0.000362 | PosSim 0.811976 | NegSim 0.062698 | LR 0.00002978
Epoch 1/1 | Step 0114/15625 | Loss 0.000173 | PosSim 0.811661 | NegSim 0.057436 | LR 0.00002978
Epoch 1/1 | Step 0115/15625 | Loss 0.000276 | PosSim 0.817171 | NegSim 0.067036 | LR 0.00002978
Epoch 1/1 | Step 0116/15625 | Loss 0.000483 | PosSim 0.812672 | NegSim 0.057822 | LR 0.00002978
Epoch 1/1 | Step 0117/15625 | Loss 0.000151 | PosSim 0.812804 | NegSim 0.065661 | LR 0.00002978
Epoch 1/1 | Step 0118/15625 | Loss 0.000433 | PosSim 0.817228 | NegSim 0.069406 | LR 0.00002977
Epoch 1/1 | Step 0119/15625 | Loss 0.000856 | PosSim 0.808710 | NegSim 0.058098 | LR 0.00002977
Epoch 1/1 | Step 0120/15625 | Loss 0.000247 | PosSim 0.808968 | NegSim 0.053161 | LR 0.00002977
Epoch 1/1 | Step 0121/15625 | Loss 0.000249 | PosSim 0.808154 | NegSim 0.067204 | LR 0.00002977
Epoch 1/1 | Step 0122/15625 | Loss 0.000289 | PosSim 0.808590 | NegSim 0.060723 | LR 0.00002977
Epoch 1/1 | Step 0123/15625 | Loss 0.000222 | PosSim 0.810318 | NegSim 0.061164 | LR 0.00002976
Epoch 1/1 | Step 0124/15625 | Loss 0.000249 | PosSim 0.807738 | NegSim 0.063526 | LR 0.00002976
Epoch 1/1 | Step 0125/15625 | Loss 0.000318 | PosSim 0.810837 | NegSim 0.056711 | LR 0.00002976
Epoch 1/1 | Step 0126/15625 | Loss 0.000298 | PosSim 0.807122 | NegSim 0.058845 | LR 0.00002976
Epoch 1/1 | Step 0127/15625 | Loss 0.000281 | PosSim 0.808725 | NegSim 0.054482 | LR 0.00002976
Epoch 1/1 | Step 0128/15625 | Loss 0.011518 | PosSim 0.802114 | NegSim 0.064969 | LR 0.00002975
Epoch 1/1 | Step 0129/15625 | Loss 0.000235 | PosSim 0.802055 | NegSim 0.064243 | LR 0.00002975
Epoch 1/1 | Step 0130/15625 | Loss 0.000344 | PosSim 0.804146 | NegSim 0.054058 | LR 0.00002975
Epoch 1/1 | Step 0131/15625 | Loss 0.000161 | PosSim 0.816024 | NegSim 0.065562 | LR 0.00002975
Epoch 1/1 | Step 0132/15625 | Loss 0.000147 | PosSim 0.810260 | NegSim 0.057568 | LR 0.00002975
Epoch 1/1 | Step 0133/15625 | Loss 0.000538 | PosSim 0.802138 | NegSim 0.055555 | LR 0.00002974
Epoch 1/1 | Step 0134/15625 | Loss 0.000351 | PosSim 0.804407 | NegSim 0.057813 | LR 0.00002974
Epoch 1/1 | Step 0135/15625 | Loss 0.000401 | PosSim 0.806198 | NegSim 0.064402 | LR 0.00002974
Epoch 1/1 | Step 0136/15625 | Loss 0.000239 | PosSim 0.798105 | NegSim 0.057226 | LR 0.00002974
Epoch 1/1 | Step 0137/15625 | Loss 0.000134 | PosSim 0.804072 | NegSim 0.054037 | LR 0.00002974
Epoch 1/1 | Step 0138/15625 | Loss 0.000324 | PosSim 0.805005 | NegSim 0.061381 | LR 0.00002974
Epoch 1/1 | Step 0139/15625 | Loss 0.000088 | PosSim 0.808933 | NegSim 0.053290 | LR 0.00002973
Epoch 1/1 | Step 0140/15625 | Loss 0.000354 | PosSim 0.802712 | NegSim 0.047831 | LR 0.00002973
Epoch 1/1 | Step 0141/15625 | Loss 0.000197 | PosSim 0.809524 | NegSim 0.053483 | LR 0.00002973
Epoch 1/1 | Step 0142/15625 | Loss 0.000134 | PosSim 0.807605 | NegSim 0.053360 | LR 0.00002973
Epoch 1/1 | Step 0143/15625 | Loss 0.000159 | PosSim 0.803244 | NegSim 0.050007 | LR 0.00002973
Epoch 1/1 | Step 0144/15625 | Loss 0.000222 | PosSim 0.808745 | NegSim 0.054961 | LR 0.00002972
Epoch 1/1 | Step 0145/15625 | Loss 0.004560 | PosSim 0.802640 | NegSim 0.048110 | LR 0.00002972
Epoch 1/1 | Step 0146/15625 | Loss 0.000162 | PosSim 0.798725 | NegSim 0.049180 | LR 0.00002972
Epoch 1/1 | Step 0147/15625 | Loss 0.000576 | PosSim 0.791428 | NegSim 0.049260 | LR 0.00002972
Epoch 1/1 | Step 0148/15625 | Loss 0.000267 | PosSim 0.791888 | NegSim 0.050552 | LR 0.00002972
Epoch 1/1 | Step 0149/15625 | Loss 0.000374 | PosSim 0.793689 | NegSim 0.050159 | LR 0.00002971
Epoch 1/1 | Step 0150/15625 | Loss 0.000450 | PosSim 0.793928 | NegSim 0.055434 | LR 0.00002971
Epoch 1/1 | Step 0151/15625 | Loss 0.007263 | PosSim 0.783201 | NegSim 0.052270 | LR 0.00002971
Epoch 1/1 | Step 0152/15625 | Loss 0.000559 | PosSim 0.798055 | NegSim 0.048192 | LR 0.00002971
Epoch 1/1 | Step 0153/15625 | Loss 0.000225 | PosSim 0.792246 | NegSim 0.043571 | LR 0.00002971
Epoch 1/1 | Step 0154/15625 | Loss 0.000143 | PosSim 0.794033 | NegSim 0.050983 | LR 0.00002970
Epoch 1/1 | Step 0155/15625 | Loss 0.000594 | PosSim 0.792865 | NegSim 0.054690 | LR 0.00002970
Epoch 1/1 | Step 0156/15625 | Loss 0.000422 | PosSim 0.795156 | NegSim 0.045993 | LR 0.00002970
Epoch 1/1 | Step 0157/15625 | Loss 0.000228 | PosSim 0.790865 | NegSim 0.050677 | LR 0.00002970
Epoch 1/1 | Step 0158/15625 | Loss 0.000236 | PosSim 0.801611 | NegSim 0.047934 | LR 0.00002970
Epoch 1/1 | Step 0159/15625 | Loss 0.000175 | PosSim 0.802850 | NegSim 0.055823 | LR 0.00002969
Epoch 1/1 | Step 0160/15625 | Loss 0.000170 | PosSim 0.804223 | NegSim 0.049043 | LR 0.00002969
Epoch 1/1 | Step 0161/15625 | Loss 0.000133 | PosSim 0.796886 | NegSim 0.047992 | LR 0.00002969
Epoch 1/1 | Step 0162/15625 | Loss 0.000849 | PosSim 0.788819 | NegSim 0.049932 | LR 0.00002969
Epoch 1/1 | Step 0163/15625 | Loss 0.000183 | PosSim 0.802367 | NegSim 0.045797 | LR 0.00002969
Epoch 1/1 | Step 0164/15625 | Loss 0.000377 | PosSim 0.793660 | NegSim 0.049697 | LR 0.00002969
Epoch 1/1 | Step 0165/15625 | Loss 0.000267 | PosSim 0.797653 | NegSim 0.050832 | LR 0.00002968
Epoch 1/1 | Step 0166/15625 | Loss 0.000092 | PosSim 0.805552 | NegSim 0.050391 | LR 0.00002968
Epoch 1/1 | Step 0167/15625 | Loss 0.000178 | PosSim 0.784552 | NegSim 0.050834 | LR 0.00002968
Epoch 1/1 | Step 0168/15625 | Loss 0.000180 | PosSim 0.797135 | NegSim 0.052593 | LR 0.00002968
Epoch 1/1 | Step 0169/15625 | Loss 0.000156 | PosSim 0.792347 | NegSim 0.046517 | LR 0.00002968
Epoch 1/1 | Step 0170/15625 | Loss 0.000234 | PosSim 0.797023 | NegSim 0.047314 | LR 0.00002967
Epoch 1/1 | Step 0171/15625 | Loss 0.000176 | PosSim 0.802516 | NegSim 0.046307 | LR 0.00002967
Epoch 1/1 | Step 0172/15625 | Loss 0.000192 | PosSim 0.797785 | NegSim 0.052851 | LR 0.00002967
Epoch 1/1 | Step 0173/15625 | Loss 0.000170 | PosSim 0.803381 | NegSim 0.051010 | LR 0.00002967
Epoch 1/1 | Step 0174/15625 | Loss 0.000206 | PosSim 0.806047 | NegSim 0.052416 | LR 0.00002967
Epoch 1/1 | Step 0175/15625 | Loss 0.000178 | PosSim 0.791056 | NegSim 0.052067 | LR 0.00002966
Epoch 1/1 | Step 0176/15625 | Loss 0.000154 | PosSim 0.807983 | NegSim 0.056483 | LR 0.00002966
Epoch 1/1 | Step 0177/15625 | Loss 0.001305 | PosSim 0.798564 | NegSim 0.045096 | LR 0.00002966
Epoch 1/1 | Step 0178/15625 | Loss 0.000285 | PosSim 0.789452 | NegSim 0.044803 | LR 0.00002966
Epoch 1/1 | Step 0179/15625 | Loss 0.000726 | PosSim 0.803195 | NegSim 0.049144 | LR 0.00002966
Epoch 1/1 | Step 0180/15625 | Loss 0.000166 | PosSim 0.799346 | NegSim 0.052035 | LR 0.00002965
Epoch 1/1 | Step 0181/15625 | Loss 0.000218 | PosSim 0.796797 | NegSim 0.046762 | LR 0.00002965
Epoch 1/1 | Step 0182/15625 | Loss 0.000970 | PosSim 0.799467 | NegSim 0.047792 | LR 0.00002965
Epoch 1/1 | Step 0183/15625 | Loss 0.001050 | PosSim 0.793263 | NegSim 0.043365 | LR 0.00002965
Epoch 1/1 | Step 0184/15625 | Loss 0.000235 | PosSim 0.808508 | NegSim 0.047541 | LR 0.00002965
Epoch 1/1 | Step 0185/15625 | Loss 0.000391 | PosSim 0.794130 | NegSim 0.047878 | LR 0.00002964
Epoch 1/1 | Step 0186/15625 | Loss 0.000176 | PosSim 0.794319 | NegSim 0.050592 | LR 0.00002964
Epoch 1/1 | Step 0187/15625 | Loss 0.000276 | PosSim 0.800596 | NegSim 0.046556 | LR 0.00002964
Epoch 1/1 | Step 0188/15625 | Loss 0.000118 | PosSim 0.806397 | NegSim 0.047761 | LR 0.00002964
Epoch 1/1 | Step 0189/15625 | Loss 0.000205 | PosSim 0.802948 | NegSim 0.054812 | LR 0.00002964
Epoch 1/1 | Step 0190/15625 | Loss 0.000149 | PosSim 0.795762 | NegSim 0.043373 | LR 0.00002964
Epoch 1/1 | Step 0191/15625 | Loss 0.000298 | PosSim 0.793965 | NegSim 0.049546 | LR 0.00002963
Epoch 1/1 | Step 0192/15625 | Loss 0.000202 | PosSim 0.790386 | NegSim 0.047459 | LR 0.00002963
Epoch 1/1 | Step 0193/15625 | Loss 0.000231 | PosSim 0.797312 | NegSim 0.050562 | LR 0.00002963
Epoch 1/1 | Step 0194/15625 | Loss 0.000178 | PosSim 0.798105 | NegSim 0.050568 | LR 0.00002963
Epoch 1/1 | Step 0195/15625 | Loss 0.000537 | PosSim 0.794265 | NegSim 0.046062 | LR 0.00002963
Epoch 1/1 | Step 0196/15625 | Loss 0.000198 | PosSim 0.805848 | NegSim 0.041045 | LR 0.00002962
