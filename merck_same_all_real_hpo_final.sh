# #!/bin/bash
mkdir -p logs

# dataset_names=('ox2' 'thrombin')
# dataset_names=('hivint' 'rat_f')
dataset_names=('3a4' 'cb1' 'hivint' 'logd' 'metab' 'ox1' 'ox2' 'pgp' 'ppb' 'rat_f' 'tdi' 'thrombin') #  'ox1' 'ox2' 'pgp' 'ppb' 'rat_f' 'tdi' 'thrombin'
targets=('hivprot' 'dpp4' 'nk1') #   'hivprot' 'dpp4' 'nk1'
sencoders=('dsets' 'strans') #  'dsets' 'strans'


gpus=(2 4 5)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=8
job_id=0

active_pids=()

for ds in "${targets[@]}"; do
    for vt in count; do
        for n_context in 3 4 6 8; do

            sencoder_layer="max"  # fixed

            for sencoder in "${sencoders[@]}"; do
                # Optionally validate compatibility (skip if needed)
                if [[ "$sencoder" == "strans" && "$sencoder_layer" != "sum" ]]; then
                    continue
                fi

                # Build a filtered list excluding $ds
                filtered=()
                for name in "${dataset_names[@]}"; do
                    [[ "$name" != "$ds" ]] && filtered+=("$name")
                done

                for ((i=0; i<${#filtered[@]}-1; i++)); do
                    for ((j=i+1; j<${#filtered[@]}; j++)); do
                        md1="${filtered[i]}"
                        md2="${filtered[j]}"

                        gpu_id=${gpus[$((job_id % gpu_count))]}

                        echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | nc=$n_context | sencoder=$sencoder | sl=$sencoder_layer"

                        RANDOM_YV=1 SAVE_TSNE_MODEL=0 MVALID_DEFAULT=nkq_count_4dset CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real_nk1_n_context.py \
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
                            > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_nc_${n_context}_se_${sencoder}_sl_${sencoder_layer}_nkq_count_4dset.log 2>&1 &

                        ((job_id++))

                        if (( job_id % (gpu_count * 3) == 0 )); then
                            wait
                        fi
                    done
                done
            done
        done
    done
done
