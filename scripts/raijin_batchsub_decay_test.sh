#!/bin/sh

for i in $(seq 4 8); do
#for i in "9"; do
    d1="0.9${i}"
    d2="0.9${i}5"
    o1="logs/fixed-decay_test/$d1"
    o2="logs/fixed-decay_test/$d2"
    m1="models/fixed-decay_test/$d1"
    m2="models/fixed-decay_test/$d2"
    l1="$o1-log.txt"
    l2="$o2-log.txt"
    qsub -o logs/fixed-decay_test-${i}.log -e logs/fixed-decay_test-${i}.err -v decay1=$d1,decay2=$d2,outdir1=$m1,outdir2=$m2,log1=$l1,log2=$l2 scripts/raijin_decay_test.sh
done
    
