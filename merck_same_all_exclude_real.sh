# #!/bin/bash
# mkdir -p logs

# dataset_names=('3a4' 'cb1' 'hivint' 'logd' 'metab' 'ox1' 'ox2' 'pgp' 'ppb' 'rat_f' 'tdi' 'thrombin')
# targets=('hivprot' 'dpp4' 'nk1')
# sencoders=('dsets' 'strans')
# gpus=(4 5)
# gpu_count=${#gpus[@]}
# max_jobs_per_gpu=16
# job_id=0

# active_pids=()

# for ds in "${targets[@]}"; do
#     # Build a filtered list excluding $ds
#     filtered=()
#     for name in "${dataset_names[@]}"; do
#         [[ "$name" != "$ds" ]] && filtered+=("$name")
#     done

#     # Generate all 2-element combinations
#     for ((i=0; i<${#filtered[@]}-1; i++)); do
#         for ((j=i+1; j<${#filtered[@]}; j++)); do
#             md1="${filtered[i]}"
#             md2="${filtered[j]}"

#             # Skip specific pair
#             if [[ "$md1" == "ox2" && "$md2" == "thrombin" ]]; then
#                 continue
#             fi

#             for vt in count bit; do
#                 for sencoder in "${sencoders[@]}"; do
#                     gpu_id=${gpus[$((job_id % gpu_count))]}

#                     echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | sencoder=$sencoder"

#                     CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck.py \
#                         --sencoder "$sencoder" \
#                         --model mlp \
#                         --mixer_phi True \
#                         --optimizer adamwschedulefree \
#                         --seed 42 \
#                         --lr 1e-3 \
#                         --wd 1e-4 \
#                         --clr 1e-5 \
#                         --cwd 1e-3 \
#                         --dropout 0.5 \
#                         --num_layers 3 \
#                         --hidden_dim 32 \
#                         --batch_size 64 \
#                         --batchnorm False \
#                         --outer_episodes 50 \
#                         --inner_episodes 10 \
#                         --initialize_weights False \
#                         --early_stopping_episodes 10 \
#                         --n_context 4 \
#                         --in_features 6560 \
#                         --num_outputs 1 \
#                         --vec_type "$vt" \
#                         --dataset "$ds" \
#                         --same_setting \
#                         --mvalid_dataset "$md1" "$md2" \
#                         > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_se_${sencoder}.log 2>&1 &

#                     active_pids+=($!)  # Track PID of launched job
#                     ((job_id++))
#                     sleep 1  # Small delay for process scheduling

#                     # Wait after each batch of 16 jobs (8 per GPU)
#                     if (( ${#active_pids[@]} >= gpu_count * max_jobs_per_gpu )); then
#                         echo "⏳ Waiting for ${#active_pids[@]} jobs to finish..."
#                         wait "${active_pids[@]}"
#                         active_pids=()
#                     fi
#                 done
#             done
#         done
#     done
# done

# # Final wait for any remaining jobs
# if (( ${#active_pids[@]} > 0 )); then
#     echo " Waiting for final ${#active_pids[@]} jobs..."
#     wait "${active_pids[@]}"
# fi

# echo " All jobs completed."

# #!/bin/bash
mkdir -p logs

dataset_names=('ox2' 'thrombin')
targets=('hivprot' 'dpp4' 'nk1') # 'hivprot' 'dpp4' 
sencoders=('dsets' 'strans') #  'strans'
gpus=(2 4)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=8
job_id=0

active_pids=()

for ds in "${targets[@]}"; do
    # Build a filtered list excluding $ds
    filtered=()
    for name in "${dataset_names[@]}"; do
        [[ "$name" != "$ds" ]] && filtered+=("$name")
    done

    # Generate all 2-element combinations
    for ((i=0; i<${#filtered[@]}-1; i++)); do
        for ((j=i+1; j<${#filtered[@]}; j++)); do
            md1="${filtered[i]}"
            md2="${filtered[j]}"

            # Skip specific pair
            if [[ "$md1" == "ox2" && "$md2" == "thrombin" ]]; then
                for vt in count bit; do
                    for sencoder in "${sencoders[@]}"; do
                        gpu_id=${gpus[$((job_id % gpu_count))]}

                        echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | sencoder=$sencoder"

                        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real.py \
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
                            --exclude_mval_data_in_context \
                            --mvalid_dataset "$md1" "$md2" \
                            > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_se_${sencoder}.log 2>&1 &

                        ((job_id++))
                        # Wait after every batch of 8 jobs (i.e., one per GPU)
                        if (( job_id % (gpu_count * 5) == 0 )); then
                            wait
                        fi
                    done
                done
            fi
        done
    done
done

# Final wait for any remaining jobs
# if (( ${#active_pids[@]} > 0 )); then
#     echo " Waiting for final ${#active_pids[@]} jobs..."
#     wait "${active_pids[@]}"
# fi

echo " All jobs completed."



