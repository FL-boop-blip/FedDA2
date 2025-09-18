#!/bin/bash
# 定义要运行的方法列表
# methods=('FedAvg' 'FedCM' 'FedDyn' 'SCAFFOLD' 
#                                          'FedGamma' 'FedSpeed' 'FedSMOO' 'FedPVU' 'FedPVU_SAM' 'A_FedPD' 'A_FedPD_SAM' 'FedTOGA' 'FedVRA' 'FedLESAM_D' 'FedLAP')


methods=('FedSGAM_KL')

for method in "${methods[@]}"; do
    python train.py --method "$method" \
        --dataset AG_News \
        --comm-rounds 300 \
        --model AG_News_NN \
        --batchsize 50 \
        --local-learning-rate 0.1 \
        --local-epochs 5 \
        # --non-iid \
        --lamb 0.1 
done


# 循环运行每个方法
# for method in "${methods[@]}"; do
#     python train.py --method "$method" \
#         --dataset mnist \
#         --comm-rounds 300 \
#         --model mnist_2NN \
#         --batchsize 50 \
#         --local-learning-rate 0.1 \
#         --local-epochs 5 \
#         --non-iid \
#         --lamb 0.1 
# done

# split_coefs=(0.1)

# for method in "${methods[@]}"; do
#     for split_coef in "${split_coefs[@]}"; do
#         echo "Running method: $method with split_coef: $split_coef"
#         python train.py --method "$method" \
#             --dataset mnist \
#             --comm-rounds 300 \
#             --model mnist_2NN \
#             --batchsize 50 \
#             --local-learning-rate 0.1 \
#             --local-epochs 5 \
#             --non-iid \
#             --lamb 0.1 \
#             --split-coef "$split_coef"
#     done
# done


# for method in "${methods[@]}"; do
#     python train.py --method "$method" \
#         --dataset mnist \
#         --comm-rounds 300 \
#         --model mnist_2NN \
#         --batchsize 50 \
#         --local-learning-rate 0.1 \
#         --local-epochs 5 \
#         --non-iid \
#         --split-rule Pathological \
#         --a 2.0 \
#         --d 10.0 \
#         --rho 0.01 \
#         --delta 0.1 \
#         --lamb 0.1 \
#         --split-coef "$split_coef"
# done