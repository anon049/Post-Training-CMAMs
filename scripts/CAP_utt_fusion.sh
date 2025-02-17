set -e
modality=AVL
run_idx=$1
gpu=$2
rate_indexes=(0 1 2 3 4 5 6 7 8 9)

for rate in "${rate_indexes[@]}"; do
    for i in `seq 1 1 10`;
    do
        cmd="python train_miss.py --dataset_mode=multimodal_miss --model=utt_fusion
        --gpu_ids=$gpu --modality=$modality --corpus_name=IEMOCAP
        --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10
        --A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
        --V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
        --L_type=bert_large --input_dim_l=1024 --embd_size_l=128
        --output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
        --niter=10 --niter_decay=10 --in_mem --beta1=0.9
        --batch_size=128 --lr=2e-4 --run_idx=$run_idx
        --name=CAP_utt_fusion_miss_${rate} --suffix={modality}_run${run_idx}_${rate}
        --has_test --cvNo=$i --rate_indx=${rate}"
        
        echo "\n-------------------------------------------------------------------------------------"
        echo "Execute command: $cmd"
        echo "-------------------------------------------------------------------------------------\n"
        echo $cmd | sh
        
    done
done