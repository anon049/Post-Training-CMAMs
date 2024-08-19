set -e
run_idx=$1
gpu=0
RATES=(7)
run_idx=2
for rate_indx in "${RATES[@]}"; do
    # for i in `seq 2 1 2`; do
    cmd="python3 RedCore/train_miss_mosei3.py --dataset_mode=cmu_mosei_miss --model=redcore_mmin
    --log_dir=./logs/mosei/redcore --checkpoints_dir=./checkpoints/mosei/redcore --gpu_ids=$gpu
    --A_type=comparE --input_dim_a=74 --norm_method=trn --embd_size_a=96 --embd_method_a=maxpool
    --V_type=denseface --input_dim_v=35 --embd_size_v=96  --embd_method_v=maxpool
    --L_type=bert_large --input_dim_l=768 --embd_size_l=96
    --AE_layers=160,80,32 --n_blocks=5 --num_thread=0 --corpus=CMU_MOSEI
    --pretrained_path='checkpoints/CAP_utt_fusion_AVL_run1'
    --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0
    --output_dim=3 --cls_layers=96,96 --dropout_rate=0.5
    --niter=10 --niter_decay=10 --verbose --print_freq=10 --in_mem
    --batch_size=256 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5
    --name=redcore_MOSEI_MRvar2 --suffix=block_{n_blocks}_run{run_idx}_${rate_indx} --has_test
    --cvNo=1
    --etabetaind=1
    --beta1=0.9
    --rate_indx=$rate_indx"
    
    
    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
    # done
done