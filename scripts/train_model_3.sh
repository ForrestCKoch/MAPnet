#!/bin/bash

python3.6 mapnet/train.py \
    --datapath dwi_data/ \
    --savepath models/ \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.0001 \
    --workers 6 \
    --scale-inputs \
    --cuda \
    --stride 1 \
    --even-padding \
    --conv-layers 5 \
    --filters 2 2 2 2 2  \
    --kernel-size 5 4 4 2 2 \
    --dilation 1 \
    --conv-actv elu \
    --fc-actv elu \
    --decay 0.333 \
    --pooling max \
    --reduce-on-plateau 
#    --debug-size 4 104 104 72
