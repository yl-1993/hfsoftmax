#!/bin/bash

dataset_path=$1
GPU=0,1,2,3
num=4
dist_port='23456'
num_classes=85164

CUDA_VISIBLE_DEVICES=$GPU mpirun -np $num python train.py \
    --distributed \
    --workders 0 \
    --dist-port $dist_port \
    --num-classes $num_classes \
    --train-filelist $dataset_path/train_list.txt \
    --train-prefix $dataset_path/images/ \
    --val-filelist $dataset_path/val_list.txt \
    --val-prefix $dataset_path/images/
