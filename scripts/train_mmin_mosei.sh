set -e
gpu=0

cvNos=(1 2 3 4 5 6 7 8 9 10)
rate_indexes=(0 1 3 5 7 9)
run_indexes=(1 2 3)
modality=AVL

OUTPUT_DIR="."

run_idx=$1
# for run_idx in `seq 1 1 3`; do
    for rate_indx in "${rate_indexes[@]}"; do
        for cv in "${cvNos[@]}"; do
            cmd="python RedCore/train_miss_mosei3.py --dataset_mode=cmu_mosei_miss --model=utt_fusion
            --modality=$modality
            --log_dir=$OUTPUT_DIR/logs/cap --checkpoints_dir=$OUTPUT_DIR/checkpoints/mosei --gpu_ids=$gpu
            --A_type=comparE --input_dim_a=74 --norm_method=trn --embd_size_a=96 --embd_method_a=maxpool
            --V_type=denseface --input_dim_v=35 --embd_size_v=96  --embd_method_v=maxpool
            --L_type=bert_large --input_dim_l=768 --embd_size_l=96 
              --num_thread=0 --corpus=CMU_MOSEI 
            --output_dim=3 --cls_layers=96,96 --dropout_rate=0.5
            --niter=10 --niter_decay=10 --verbose --print_freq=10 --in_mem
            --batch_size=256 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5         
            --name=CAP_utt_fusion_AVL --suffix=run${run_idx}_${rate_indx} --has_test
            --cvNo=${cv}"
                
            echo "\n-------------------------------------------------------------------------------------"
            echo "Execute command: $cmd"
            echo "-------------------------------------------------------------------------------------\n"
            echo $cmd | sh
            
            cmd="python3 RedCore/train_miss_mosei3.py --dataset_mode=cmu_mosei_miss --model=mmin
            --log_dir=$OUTPUT_DIR/logs/mmin --checkpoints_dir=$OUTPUT_DIR/checkpoints/mosei --gpu_ids=$gpu
            --A_type=comparE --input_dim_a=74 --norm_method=trn --embd_size_a=96 --embd_method_a=maxpool
            --V_type=denseface --input_dim_v=35 --embd_size_v=96  --embd_method_v=maxpool
            --L_type=bert_large --input_dim_l=768 --embd_size_l=96 
              --num_thread=0 --corpus=CMU_MOSEI 
            --output_dim=3 --cls_layers=96,96 --dropout_rate=0.5
            --niter=10 --niter_decay=10 --verbose --print_freq=10 --in_mem
            --batch_size=256 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5         
            --name=MMIN --suffix=block_run${run_idx}_${rate_indx} --has_test
            --record_folder=result_visualization/mmin_$i_$rate_indx
            --pretrained_path='$OUTPUT_DIR/checkpoints/mosei/CAP_utt_fusion_AVL_run${run_idx}_${rate_indx}'
            --cvNo=${cv}
            --etabetaind=1
            "
            echo "\n-------------------------------------------------------------------------------------"
            echo "Execute command: $cmd"
            echo "-------------------------------------------------------------------------------------\n"
            echo $cmd | sh


        done
    done
# done

