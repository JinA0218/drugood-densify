# #!/bin/bash
mkdir -p logs

targets=('hivprot' 'dpp4' 'nk1') #   'hivprot' 'dpp4' 'nk1'
sencoders=('dsets' 'strans') #  'dsets' 'strans'
gpus=(0 1 2 3)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=8
job_id=0

active_pids=()

for ds in "${targets[@]}"; do
    for vt in count bit; do  # vec_type loop
        for sencoder in "${sencoders[@]}"; do
            for n_mvalid in 1 6 16; do  # innermost loop
                gpu_id=${gpus[$((job_id % gpu_count))]}

                echo "Launching job $job_id on GPU $gpu_id | ds=$ds | vt=$vt | sencoder=$sencoder | n_mvalid=$n_mvalid"

                RANDOM_YV=1 SAVE_TSNE_MODEL=0 MVALID_DEFAULT=NMVALID_POC CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real_nk1_n_context_n_mvalid.py \
                    --sencoder "$sencoder" \
                    --model mlp \
                    --mixer_phi True \
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
                    --mvalid_dataset None \
                    --n_mvalid "$n_mvalid" \
                    > logs/job_${job_id}_ds_${ds}_vt_${vt}_se_${sencoder}_nmv_${n_mvalid}_NMVALID_POC.log 2>&1 &

                ((job_id++))

                # Optional job limit
                # if [[ $job_id -ge 10 ]]; then
                #     exit 0
                # fi

                # Wait after a batch of jobs
                if (( job_id % (gpu_count * 3) == 0 )); then
                    wait
                fi
            done
        done
    done
done


# Final wait for any remaining jobs
# if (( ${#active_pids[@]} > 0 )); then
#     echo " Waiting for final ${#active_pids[@]} jobs..."
#     wait "${active_pids[@]}"
# fi

echo " All jobs completed."



