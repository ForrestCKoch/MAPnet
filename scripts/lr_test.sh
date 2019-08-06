#!/bin/bash

for batch_size in 32 64 128; do
    for lr in 0.01 0.001 0.0001 0.00001; do
python3.6 mapnet/train.py \
    --datapath dwi_data/ \
    --savepath models/ \
    --batch-size $batch_size \
    --epochs 10 \
    --lr $lr \
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
    --pooling max 
#    --debug-size 4 104 104 72
done
done
