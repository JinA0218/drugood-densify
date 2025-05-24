# #!/bin/bash
mkdir -p logs

# dataset_names=('3a4' 'cb1' 'hivint' 'logd' 'metab' 'ox1' 'ox2' 'pgp' 'ppb' 'rat_f' 'tdi' 'thrombin')
dataset_names=('3a4' 'cb1' 'hivint' 'logd' 'metab' 'ox1' 'ox2' 'pgp' 'ppb' 'rat_f' 'tdi' 'thrombin') #  
targets=('nk1') #   'hivprot' 'dpp4' 'nk1'
sencoders=('strans') #  'dsets' 'strans'


gpus=(3 4 5)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=8
job_id=0

active_pids=()

for ds in "${targets[@]}"; do
    for vt in count; do  # vec_type loop (you can extend it)
        for n_context in 2; do

            # Define valid sencoder_layer settings
            declare -A sencoder_layer_map
            sencoder_layer_map[dsets]="max sum"
            sencoder_layer_map[strans]="sum"
            sencoder_layer_map[default]="sum"

            # Enumerate all unique sencoder_layer values across all sencoders
            all_layers=("max" "sum")

            for sencoder_layer in "${all_layers[@]}"; do

                for sencoder in "${sencoders[@]}"; do
                    valid_layers="${sencoder_layer_map[$sencoder]:-${sencoder_layer_map[default]}}"
                    if [[ ! " $valid_layers " =~ " $sencoder_layer " ]]; then
                        continue  # Skip invalid layer for this encoder
                    fi

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

                            gpu_id=${gpus[$((job_id % gpu_count))]}

                            echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | n_context=$n_context | sencoder=$sencoder | sencoder_layer=$sencoder_layer"

                            RANDOM_YV=1 SAVE_TSNE_MODEL=0 MVALID_DEFAULT=nkq_count_4strans_ncontext_final CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real_nk1_n_context.py \
                                --sencoder "$sencoder" \
                                --sencoder_layer "$sencoder_layer" \
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
                                --n_context "$n_context" \
                                --in_features 6560 \
                                --num_outputs 1 \
                                --vec_type "$vt" \
                                --dataset "$ds" \
                                --same_setting \
                                --mvalid_dataset "$md1" "$md2" \
                                > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_nc_${n_context}_se_${sencoder}_sl_${sencoder_layer}_nkq_count_4strans.log 2>&1 &

                            ((job_id++))

                            # if (( job_id >= 8 )); then
                            #     wait  # ensure all background jobs finish cleanly
                            #     exit 0
                            # fi

                            if (( job_id % (gpu_count * 3) == 0 )); then
                                wait
                            fi
                        done
                    done
                done
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



