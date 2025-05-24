# #!/bin/bash
mkdir -p logs

targets=('hivprot' 'dpp4' 'nk1') # 'hivprot' 'dpp4' 'nk1'
sencoders=('dsets' 'strans') #  'strans'
gpus=(4)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=8
job_id=0

active_pids=()

for ds in "${targets[@]}"; do
    for vt in count bit; do # count 
        gpu_id=${gpus[$((job_id % gpu_count))]}

        echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | sencoder=$sencoder"

        RANDOM_YV=1 SAVE_TSNE_MODEL=0 MVALID_DEFAULT=0 CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real.py \
            --model mlp \
            --mixer_phi False \
            --optimizer adamwschedulefree \
            --seed 42 \
            --lr 1e-3 \
            --wd 1e-4 \
            --clr 1e-5 \
            --cwd 1e-3 \
            --dropout 0.5 \
            --num_layers 3 \
            --hidden_dim 32 \
            --batch_size 64 \
            --batchnorm False \
            --outer_episodes 50 \
            --inner_episodes 10 \
            --initialize_weights False \
            --early_stopping_episodes 10 \
            --n_context 4 \
            --in_features 6560 \
            --num_outputs 1 \
            --vec_type "$vt" \
            --dataset "$ds" \
            --same_setting \
            --epochs 20 \
            --sencoder dsets 
            > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_se_${sencoder}_mlp.log 2>&1 &

        ((job_id++))
        # Wait after every batch of 8 jobs (i.e., one per GPU)
        if (( job_id % (gpu_count * 2) == 0 )); then
            wait
        fi
    done
done


            # --mvalid_dataset "$md1" "$md2" \
            # --sencoder "$sencoder" \



# Final wait for any remaining jobs
# if (( ${#active_pids[@]} > 0 )); then
#     echo " Waiting for final ${#active_pids[@]} jobs..."
#     wait "${active_pids[@]}"
# fi

echo " All jobs completed."



