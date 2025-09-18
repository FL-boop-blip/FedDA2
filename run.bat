@echo off
setlocal enabledelayedexpansion

:: 定义方法数组（Windows CMD 使用空格分隔）
set methods=FedGKD_PD

:: 定义分割系数数组（Windows CMD 使用空格分隔）
set split_coef=0.1 0.3 0.6

:: 循环运行每个方法
for %%m in (%methods%) do (
    for %%s in (%split_coef%) do (
        echo Running method: %%m with split_coef: %%s
        python train.py --method "%%m" ^
            --dataset CIFAR10 ^
            --comm-rounds 1000 ^
            --model ResNet18 ^
            --non-iid ^
            --batchsize 50 ^
            --local-learning-rate 0.1 ^
            --local-epochs 5 ^
            --split-coef %%s ^
            --cuda 0 ^
            --lamb 0.1 
    )
)


set split_coef=3.0 6.0

:: 循环运行每个方法
for %%m in (%methods%) do (
    for %%s in (%split_coef%) do (
        echo Running method: %%m with split_coef: %%s
        python train.py --method "%%m" ^
            --dataset CIFAR10 ^
            --comm-rounds 1000 ^
            --model ResNet18 ^
            --non-iid ^
            --batchsize 50 ^
            --local-learning-rate 0.1 ^
            --local-epochs 5 ^
            --split-rule Pathological ^
            --split-coef %%s ^
            --cuda 0 ^
            --lamb 0.1 
    )
)


for %%m in (%methods%) do (
    for %%s in (%split_coef%) do (
        echo Running method: %%m with split_coef: %%s
        python train.py --method "%%m" ^
            --dataset CIFAR10 ^
            --comm-rounds 1000 ^
            --model ResNet18 ^
            --batchsize 50 ^
            --local-learning-rate 0.1 ^
            --local-epochs 5 ^
            --split-rule Pathological ^
            --split-coef %%s ^
            --cuda 0 ^
            --lamb 0.1 
    )
)



endlocal
