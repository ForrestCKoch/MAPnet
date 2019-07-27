#!/bin/bash

python3.6 mapnet/train.py \
    --datapath dwi_data/ \
    --savepath models/ \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.00001 \
    --workers 16 \
    --scale-inputs \
    --cuda \
    --stride 1 4 1 3 1 4 3\
    --padding 2 2 1 1 2 1 1\
    --conv-layers 7 \
    --filters 4 1 4 1 4 1 2\
    --kernel-size 4 4 3 3 4 4 2 \
    --conv-actv elu \
    --fc-actv elu
#    --debug-size 4 104 104 72
