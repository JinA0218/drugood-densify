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

declare -A validation_lists
validation_lists["dsets_hivprot_count"]="logd 3a4"
validation_lists["dsets_hivprot_bit"]="logd 3a4"
validation_lists["dsets_dpp4_count"]="cb1 ppb"
validation_lists["dsets_dpp4_bit"]="logd tdi"
validation_lists["dsets_nk1_count"]="3a4 ox1"
validation_lists["dsets_nk1_bit"]="3a4 logd"

validation_lists["strans_hivprot_count"]="logd 3a4"
validation_lists["strans_hivprot_bit"]="logd 3a4"
validation_lists["strans_dpp4_count"]="cb1 ppb"
validation_lists["strans_dpp4_bit"]="logd tdi"
validation_lists["strans_nk1_count"]="3a4 ox1"
validation_lists["strans_nk1_bit"]="3a4 logd"

targets=('hivprot' 'dpp4' 'nk1')
sencoders=('dsets') #  'strans'
gpus=(2 4)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=6
job_id=0

embed_tests=('2nd_last_ours_best') #  'setenc_ours_best' lastlayer_ours_best
# embed_tests=('lastlayer_ours_best' 'lastlayer_base_cX_mO' 'setenc_ours_best' 'setenc_base_cX_mO' 'lastlayer_base_cX_mX')

active_pids=()

for ds in "${targets[@]}"; do
    for vt in count bit; do
        for sencoder in "${sencoders[@]}"; do
            key="${sencoder}_${ds}_${vt}"
            value="${validation_lists[$key]}"
            read -r md1 md2 <<< "$value"

            other_targets=()
            for t in "${targets[@]}"; do
                if [[ "$t" != "$ds" ]]; then
                    other_targets+=("$t")
                fi
            done

            for embed_test in "${embed_tests[@]}"; do
                gpu_id=${gpus[$((job_id % gpu_count))]}

                echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | sencoder=$sencoder | embed_test=$embed_test"

                CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real.py \
                    --sencoder "$sencoder" \
                    --model mlp \
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
                    --mvalid_dataset "$md1" "$md2" \
                    --embed_test "$embed_test" \
                    --mixer_phi True \
                    --tsne_plot \
                    --specify_ood_dataset "${other_targets[@]}" \
                    > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_se_${sencoder}_embed_${embed_test// /_}.log 2>&1 &

                ((job_id++))

                if (( job_id % (gpu_count * 3) == 0 )); then
                    wait
                fi
            done
        done
    done
done

# Final wait for any remaining jobs
wait

echo "All jobs completed."




