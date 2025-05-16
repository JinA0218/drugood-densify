#!/bin/bash
mkdir -p logs

# Define validation lists (stored as space-separated strings)
declare -A validation_lists
validation_lists["hivprot_count"]="logd 3a4 tdi pgp ox1 hivint metab ox2 cb1 thrombin ppb rat_f"
validation_lists["hivprot_bit"]="logd 3a4 tdi pgp cb1 thrombin ox1 hivint ppb metab rat_f ox2"
validation_lists["dpp4_count"]="cb1 ppb tdi logd pgp ox2 rat_f 3a4 thrombin metab hivint ox1"
validation_lists["dpp4_bit"]="logd tdi ox2 pgp ppb cb1 3a4 rat_f thrombin metab ox1 hivint"
validation_lists["nk1_count"]="3a4 ox1 logd cb1 hivint metab thrombin tdi ox2 ppb rat_f pgp"
validation_lists["nk1_bit"]="3a4 logd ox1 cb1 hivint thrombin ox2 metab rat_f tdi ppb pgp"

# Targets and settings
targets=('hivprot' 'dpp4' 'nk1')
sencoders=('dsets' 'strans')

gpus=(0 4)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=6
job_id=0

for ds in "${targets[@]}"; do
    for vt in count bit; do
        key="${ds}_${vt}"

        # Read string into array properly
        IFS=' ' read -r -a values <<< "${validation_lists[$key]}"
        
        mvalid_dataset=("${values[0]}" "${values[1]}")  # First two elements

        total=${#values[@]}
        
        # From last element to second
        for ((i=2; i<total; i++)); do
            context_dataset=("${values[@]:2:i-1}")  # take from i to end

            for sencoder in "${sencoders[@]}"; do
                gpu_id=${gpus[$((job_id % gpu_count))]}
                echo "Launching job $job_id on GPU $gpu_id | ds=$ds | vt=$vt | sencoder=$sencoder | context=${context_dataset[*]}"

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
                    --mvalid_dataset "${mvalid_dataset[@]}" \
                    --low_sim ml_exl_ \
                    --context_dataset "${context_dataset[@]}" \
                    > logs/job_${job_id}_ds_${ds}_vt_${vt}_se_${sencoder}_context_${i}.log 2>&1 &

                ((job_id++))

                # Control number of jobs launched
                if (( job_id % (gpu_count * max_jobs_per_gpu) == 0 )); then
                    wait
                fi
            done
        done
    done
done

# Final wait
wait
echo "✅ All jobs completed."
