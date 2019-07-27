#!/bin/bash

python3.6 mapnet/train.py \
    --datapath dwi_data/ \
    --savepath models/ \
    --batch-size 64 \
    --epochs 10 \
    --lr 0.0005 \
    --workers 4 \
    --scale-inputs \
    --stride 1 \
    --even-padding \
    --conv-layers 6 \
    --filters 1 \
    --kernel-size 2 \
    --dilation 1 \
    --conv-actv relu \
    --fc-actv relu relu softmax\
    --decay 0.99 \
    --pooling max \
    --model-output gaussian \
    --loss L1

#    --debug-size 4 104 104 72
