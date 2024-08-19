#!/bin/bash

RUNS=$1
RATES=(0 1 3 5 7 9)


train_cmam() {
    run_id=$1
    target_modality=$2
    miss_rate=$3
    
    python MMIN/train_audio_text_cmam.py \
    --checkpoints_path MMIN/checkpoints/CAP_utt_fusion_miss__AVL_run${run_id}_${miss_rate} \
    --save_metrics_to MMIN/results/audio_text/CAP_utt_fusion__AVL_run${run_id}_${miss_rate}_AT_${target_modality} \
    --lr 0.001 \
    --test \
    --total_cv 10 \
    --cosine_weight 1.0\
    --mae_weight 1.0 \
    --mse_weight 1.0 \
    --cls_weight 0.05 \
    --recon_weight 1.0 \
    --cmam_type bimodal \
    --target_metric loss
}


for r in "${RATES[@]}"; do
    for i in $(seq 1 1 $RUNS); do
        train_cmam $i V_feat $r
        if [ $? -ne 0 ]; then
            echo "Training text failed"
            exit 1
        fi
    done
done