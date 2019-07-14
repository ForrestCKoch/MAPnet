#!/bin/bash

python3.6 map/train.py \
    --datapath dwi_data/ \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.0001 \
    --workers 32 \
    --scale-inputs \
    --cuda \
    --stride 1 4 1 3 1 4 3\
    --padding 2 2 1 1 2 1 1\
    --conv-layers 7 \
    --filters 5 1 5 1 5 1 2\
    --kernel-size 4 4 3 3 4 4 2\
    --debug-size 4 104 104 72
