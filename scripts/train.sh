#!/bin/bash

dataset_path=$1
num_classes=85164
GPU=0,1,2,3


CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -a 'hynet' \
    -b 1024 \
    --lr 0.5 \
    --num-classes $num_classes \
    --train-filelist $dataset_path/train_list.txt \
    --train-prefix $dataset_path/images/ \
    --val-filelist $dataset_path/val_list.txt \
    --val-prefix $dataset_path/images/
