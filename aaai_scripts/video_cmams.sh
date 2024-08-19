#!/bin/bash

RUNS=$1

RATES=(0 1 3 5 7 9)

train_cmam() {
    run_id=$1
    target_modality=$2
    miss_rate=$3
    
    python MMIN/train_video_cmams.py \
    --checkpoints_path MMIN/checkpoints/CAP_utt_fusion_miss__AVL_run${run_id}_${miss_rate} \
    --target_modality $target_modality \
    --save_metrics_to MMIN/results/mosei/CAP_utt_fusion__AVL_run${run_id}_${miss_rate}_V_${target_modality} \
    --lr 0.001 \
    --test \
    --total_cv 10 \
    --cosine_weight: 1.0\
    --mae_weight 1.0 \
    --mse_weight 1.0 \
    --cls_weight 0.05 \
    --recon_weight 1.0 \
    --cmam_type v1 \
    --target_metric loss
}


for r in "${RATES[@]}"; do
    for i in $(seq 1 1 $RUNS); do
        train_cmam $i L_feat $r
        if [ $? -ne 0 ]; then
            echo "Training text failed"
            exit 1
        fi
        train_cmam $i A_feat $r
        if [ $? -ne 0 ]; then
            echo "Training audio failed"
            exit 1
        fi
        
    done
done