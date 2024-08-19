gpu=0

cvNos=(1 2 3 4 5 6 7 8 9 10)
rate_indexes=(0 1 3 5 7 9)
run_indexes=(1 2 3)
modality=AVL

OUTPUT_DIR="."

for run_idx in `seq 1 1 3`; do
    for rate_indx in "${rate_indexes[@]}"; do
        for cv in "${cvNos[@]}"; do
            #--gpu_ids=$gpu --modality=$modality --corpus_name=IEMOCAP
            cmd="python train_miss_shared.py --dataset_mode=multimodal_miss --model=utt_fusion
            --gpu_ids=$gpu --modality=$modality --corpus_name=IEMOCAP
            --log_dir=$OUTPUT_DIR/logs --checkpoints_dir=$OUTPUT_DIR/checkpoints --print_freq=10
            --A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
            --V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
            --L_type=bert_large --input_dim_l=1024 --embd_size_l=128
            --output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
            --niter=10 --niter_decay=10 --in_mem --beta1=0.9
            --batch_size=128 --lr=2e-4 --run_idx=$run_idx
            --name=CAP_utt_fusion_miss_${rate} --suffix={modality}_run${run_idx}_${rate_indx}
            --has_test --cvNo=$cv --rate_indx=${rate_indx}"
            
            echo "\n-------------------------------------------------------------------------------------"
            echo "Execute command: $cmd"
            echo "-------------------------------------------------------------------------------------\n"
            echo $cmd | sh
            
            cmd="python3 train_miss_shared.py --dataset_mode=multimodal_miss --model=mmin
            --log_dir=$OUTPUT_DIR/logs/mmin --checkpoints_dir=$OUTPUT_DIR/checkpoints --gpu_ids=$gpu
            --A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
            --V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
            --L_type=bert_large --input_dim_l=1024 --embd_size_l=128 
            --AE_layers=256,128,64 --n_blocks=5 --num_thread=0 --corpus=IEMOCAP
            --pretrained_path='$OUTPUT_DIR/checkpoints/CAP_utt_fusion_miss__AVL_run${run_idx}_${rate_indx}'
            --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0
            --output_dim=4 --cls_layers=128,128 --dropout_rate=0.5
            --niter=30 --niter_decay=30 --verbose --print_freq=10 --in_mem
            --batch_size=128 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5         
            --name=mmin_IEMOCAP_a_${rate_indx}_${i} --suffix=block_{n_blocks}_run${run_idx}_${i}_${rate_indx} --has_test
            --record_folder=result_visualization/mmin_$i_$rate_indx
            --cvNo=${cv}
            --etabetaind=1
            "
            echo "\n-------------------------------------------------------------------------------------"
            echo "Execute command: $cmd"
            echo "-------------------------------------------------------------------------------------\n"
            echo $cmd | sh
        done
    done
done

