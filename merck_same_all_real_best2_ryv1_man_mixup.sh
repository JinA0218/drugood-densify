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
validation_lists["dsets_hivprot_count"]="3a4 metab" # oh actually ['3a4', 'metab']
validation_lists["dsets_hivprot_bit"]="metab pgp"
validation_lists["dsets_dpp4_count"]="hivint rat_f"
validation_lists["dsets_dpp4_bit"]="ox1 ppb"
validation_lists["dsets_nk1_count"]="3a4 metab"
validation_lists["dsets_nk1_bit"]="3a4 cb1"

validation_lists["strans_hivprot_count"]="3a4 ox1"
validation_lists["strans_hivprot_bit"]="hivint pgp"
validation_lists["strans_dpp4_count"]="3a4 ox1"
validation_lists["strans_dpp4_bit"]="logd ox1"
validation_lists["strans_nk1_count"]="logd ox2"
validation_lists["strans_nk1_bit"]="3a4 hivint"

targets=('hivprot' 'dpp4' 'nk1') #    'dpp4' 'nk1'
sencoders=('dsets' 'strans') #    'strans'
mixing_layers=(0)  #  1 2


gpus=(2)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=6
job_id=0



# 1. ablation of concating x again after mixing
# 2. change mixing location

# embed_tests=('2nd_last_ours_best') #  'setenc_ours_best' lastlayer_ours_best
# embed_tests=('lastlayer_ours_best' 'lastlayer_base_cX_mO' 'setenc_ours_best' 'setenc_base_cX_mO' 'lastlayer_base_cX_mX')

active_pids=()

for ds in "${targets[@]}"; do
    for vt in count bit; do #  count
        for sencoder in "${sencoders[@]}"; do
            key="${sencoder}_${ds}_${vt}"
            value="${validation_lists[$key]}"
            read -r md1 md2 <<< "$value"
            
            for ml in "${mixing_layers[@]}"; do

                gpu_id=${gpus[$((job_id % gpu_count))]}

                echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | sencoder=$sencoder | mixing_layer=$ml"
                mix_default_type="xmix"
                MIX_TYPE=MIXUP RANDOM_YV=1 SAVE_TSNE_MODEL=0 MVALID_DEFAULT=tsne_best_origin_testing MIXING_X_DEFAULT=$mix_default_type CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real.py \
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
                    --mixer_phi True \
                    --mixing_layer "$ml" 
                    > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_se_${sencoder}_mt_${mix_default_type}_ml_${ml// /_}_2.log 2>&1 &

                ((job_id++))

                # if [[ $job_id -ge 1 ]]; then
                #     exit 0
                # fi

                if (( job_id % (gpu_count * 2) == 0 )); then
                    wait
                fi
            done
        done
    done
done

# Final wait for any remaining jobs
wait

echo "All jobs completed."




