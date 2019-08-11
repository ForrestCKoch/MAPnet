#!/bin/bash
#PBS -q gpu
#PBS -l ncpus=6
#PBS -l mem=64GB
#PBS -l ngpus=2
#PBS -l wd
#PBS -l walltime=24:00:00
#PBS -P ey6
#PBS -N filter-test

# Add our special little module folder
module use $HOME/modules

# Remove some modules
module unload gcc
module unload intel-fc intel-cc
#module unload intel-mkl

# And add a few in
#module load intel-mkl
module load gcc/6.2.0
module load python3/3.6.7-gcc620
module load hdf5/1.10.0
# CUDA
module load cuda/10.0
module load cudnn/7.4.2-cuda10.0 # From $HOME/modules

# Force gcc/6.2.0 libraries loading 
export LD_PRELOAD=/apps/gcc/6.2.0/lib64/libstdc++.so.6
# Ensure packages in the home directory are given preference
export PYTHONPATH="/short/ey6/fk5479/local/lib/python3.6/site-packages":$PYTHONPATH

mkdir -p $outdir1
CUDA_VISIBLE_DEVICES=0 python3.6 mapnet/train.py \
    --datapath dwi_data/ \
    --savepath $outdir1 \
    --batch-size 64 \
    --epochs 40 \
    --lr 0.0001 \
    --workers 3 \
    --scale-inputs \
    --cuda \
    --stride 1 \
    --even-padding \
    --conv-layers 5 \
    --filters $(echo $filters1| tr '-' ' ') \
    --kernel-size 5 4 4 2 2 \
    --dilation 1 \
    --conv-actv tanh \
    --fc-actv sigmoid \
    --decay 0.995 \
    --pooling max > $log1 2>&1 &

mkdir -p $outdir2
CUDA_VISIBLE_DEVICES=1 python3.6 mapnet/train.py \
    --datapath dwi_data/ \
    --savepath $outdir2 \
    --batch-size 64 \
    --epochs 40 \
    --lr 0.0001 \
    --workers 3 \
    --scale-inputs \
    --cuda \
    --stride 1 \
    --even-padding \
    --conv-layers 5 \
    --filters $(echo $filters2|tr '-' ' ') \
    --kernel-size 5 4 4 2 2 \
    --dilation 1 \
    --conv-actv tanh \
    --fc-actv sigmoid \
    --decay 0.995 \
    --pooling max > $log2 2>&1 &
wait
