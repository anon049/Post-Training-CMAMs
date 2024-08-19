#!/bin/bash

RUNS=$1
RATES=(0 1 3 5 7 9)

train_cmam() {
    run_id=$1
    target_modality=$2
    miss_rate=$3
    
    python MMIN/train_audio_cmams.py \
    --checkpoints_path MMIN/checkpoints/CAP_utt_fusion_miss__AVL_run${run_id}_${miss_rate} \
    --target_modality $target_modality \
    --save_metrics_to MMIN/results/CAP_utt_fusion__AVL_run${run_id}_${miss_rate}_A_${target_modality} \
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


test_cmam() {
    run_id=$1
    miss_rate=$2
    
    target_metrics=("loss" "f1" "uar" "acc")
    
    for target_metric in "${target_metrics[@]}"; do
        python MMIN/unimodal_audio_cmams_full_test.py \
        --checkpoints_path MMIN/checkpoints/CAP_utt_fusion_miss__AVL_run${run_id}_${miss_rate} \
        --save_metrics_to MMIN/results/CAP_utt_fusion__AVL_run${run_id}_${miss_rate}_VT_${target_metric} \
        --target_metric $target_metric \
        --total_cv 10 \
        --cmam_type v1
    done
}

for r in "${RATES[@]}"; do
    for i in $(seq 1 1 $RUNS); do
        train_cmam $i L_feat $r
        if [ $? -ne 0 ]; then
            echo "Training text failed"
            exit 1
        fi
        train_cmam $i V_feat $r
        if [ $? -ne 0 ]; then
            echo "Training audio failed"
            exit 1
        fi
        
        test_cmam $i $r
        if [ $? -ne 0 ]; then
            echo "Testing failed"
            exit 1
        fi
    done
done