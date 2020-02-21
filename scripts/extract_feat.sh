#!/bin/bash

GPU=0
arch=resnet50
batch_size=1024
feature_dim=256
prefix=$1
filelist=$2
load_path=$3

export CUDA_VISIBLE_DEVICES=$GPU


python extract_feat.py \
    -a $arch \
    -b $batch_size \
    --prefix $prefix \
    --filelist $filelist \
    --load-path $load_path \
    --feature-dim $feature_dim \
    --strict
