#!/bin/bash

RUNS=$1

train_cmam() {
    run_id=$1
    target_modality=$2
    miss_rate=$3
    
    python MMIN/train_audio_video_cmam.py \
    --checkpoints_path MMIN/checkpoints/CAP_utt_fusion_miss__AVL_run${run_id}_5 \
    --save_metrics_to MMIN/results/ablation/CAP_utt_fusion__AVL_run${run_id}_5_AV_${target_modality} \
    --lr 0.001 \
    --test \
    --total_cv 10 \
    --cosine_weight 1.0\
    --mae_weight 1.0 \
    --mse_weight 1.0 \
    --cls_weight 0.05 \
    --recon_weight 1.0 \
    --cmam_type v1 \
    --target_metric loss \
    --epochs 20 \
    --use_pretrained_encoders
    
}

for i in $(seq 1 1 $RUNS); do
    train_cmam $i L_feat 0
    if [ $? -ne 0 ]; then
        echo "Training text failed"
        exit 1
    fi
done