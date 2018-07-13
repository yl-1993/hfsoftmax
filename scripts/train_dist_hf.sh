#!/bin/bash

dataset_path=$1
GPU=0,1,2,3
num=4
sample_num=1000
num_classes=85164
bs=256
dist_port='23456'
save_path='checkpoints/hfsampler_dist'

CUDA_VISIBLE_DEVICES=$GPU mpirun -np $num python train.py \
    --distributed \
    --workders 0 \
    --sampled \
    --sample-num $sample_num \
    --num-classes $num_classes \
    --batch-size $bs \
    --dist-port $dist_port \
    --save-path $save_path \
    --train-filelist $dataset_path/train_list.txt \
    --train-prefix $dataset_path/images/ \
    --val-filelist $dataset_path/val_list.txt \
    --val-prefix $dataset_path/images/

