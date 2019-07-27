#!/bin/bash

python3.6 mapnet/train.py \
    --datapath dwi_data/ \
    --savepath models/ \
    --batch-size 32 \
    --epochs 200 \
    --lr 0.01 \
    --workers 4 \
    --scale-inputs \
    --cuda \
    --stride 1 \
    --even-padding \
    --conv-layers 5 \
    --filters 2 2 2 2 2  \
    --kernel-size 5 4 4 2 2 \
    --dilation 1 \
    --conv-actv elu \
    --fc-actv elu elu softmax \
    --decay 0.3333 \
    --reduce-on-plateau \
    --pooling max \
    --model-output gaussian \
    --loss Wasserstein
#    --debug-size 4 104 104 72
