#!/bin/bash

dataset_path=$1
GPU=0
sample_num=1000
num_classes=85164
save_path='checkpoints/hfsampler'


CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --sampled \
    --sample-num $sample_num \
    --sampler-type 'hnsw' \
    --num-classes $num_classes \
    --save-path $save_path \
    --train-filelist $dataset_path/train_list.txt \
    --train-prefix $dataset_path/images/ \
    --val-filelist $dataset_path/val_list.txt \
    --val-prefix $dataset_path/images/
