#!/bin/sh

opt1=adamw
opt2=adam
for wdecay in 0 5e-03 1e-02 5e-02 1e-01 5e-01; do
    d1="$opt1-wdec-$wdecay-amsgrad"
    d2="$opt2-wdec-$wdecay-amsgrad"
    o1="logs/optims/$d1"
    o2="logs/optims/$d2"
    m1="models/optims/$d1"
    m2="models/optims/$d2"
    l1="$o1-log.txt"
    l2="$o2-log.txt"
    qsub -o logs/jobs/optims-${d1}.log -e logs/jobs/optims-${d1}.err -v optim1=$opt1,optim2=$opt2,wd=$wdecay,outdir1=$m1,outdir2=$m2,log1=$l1,log2=$l2 scripts/raijin_optimizer_test.sh
done
