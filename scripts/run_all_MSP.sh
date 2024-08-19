#!/bin/bash

for i in $(seq 1 1 3); do
    echo "Run $i"
    bash scripts/MSP_utt_fusion.sh AVL $i 0
    bash scripts/MSP_mmin.sh $i 0
done
