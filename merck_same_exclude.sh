#!/bin/bash
mkdir -p logs

# dataset_names=('3a4' 'cb1' 'hivint' 'logd' 'metab' 'ox1' 'ox2' 'pgp' 'ppb' 'rat_f' 'tdi' 'thrombin')

dataset_names=('ox2' 'thrombin')
targets=('hivprot' 'dpp4' 'nk1')
sencoders=('dsets' 'strans')
gpus=(4 5)
gpu_count=${#gpus[@]}
job_id=0

for ds in "${targets[@]}"; do
    # Exclude current ds from validation candidates
    filtered=()
    for name in "${dataset_names[@]}"; do
        [[ "$name" != "$ds" ]] && filtered+=("$name")
    done

    # Generate all 2-element unique combinations from filtered
    for ((i=0; i<${#filtered[@]}-1; i++)); do
        for ((j=i+1; j<${#filtered[@]}; j++)); do
            md1="${filtered[i]}"
            md2="${filtered[j]}"

            # if [[ "$md1" == "ox2" && "$md2" == "thrombin" ]]; then
            #     continue
            # fi

            for vt in count bit; do
                for sencoder in "${sencoders[@]}"; do
                    for sl in mean max sum pma; do
                        gpu_id=${gpus[$((job_id % gpu_count))]}

                        echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | sencoder=$sencoder"

                        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck.py \
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
                            --sencoder_layer "$sl" \
                            --exclude_mval_data_in_context \
                            --mvalid_dataset "$md1" "$md2" \
                            > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_se_${sencoder}_sl_${sl}.log 2>&1 &

                        ((job_id++))

                        # Wait after every batch of 8 jobs (i.e., one per GPU)
                        if (( job_id % (gpu_count * 8) == 0 )); then
                            wait
                        fi
                    done
                done
            done
        done
    done
done

# Wait for any remaining background jobs
wait
echo "All jobs completed."
