#!/bin/bash

sencoder="strans"
sencoder_layer="sum"
mvalid_pair=('3a4' 'cb1')  # Extend as needed

total_configs=32
num_jobs=12
configs_per_job=$((total_configs / num_jobs))
gpus=(1 4 5 6)
job_id=0

# Generate all 2-element combinations of mvalid_pair
for ((x=0; x<${#mvalid_pair[@]}-1; x++)); do
  for ((y=x+1; y<${#mvalid_pair[@]}; y++)); do
    md1=${mvalid_pair[$x]}
    md2=${mvalid_pair[$y]}

    for ((i = 0; i < num_jobs; i++)); do
      START=$((i * configs_per_job))
      STOP=$(((i + 1) * configs_per_job))
      gpu_id=${gpus[$((job_id % ${#gpus[@]}))]}

      echo "Launching sweep $START to $STOP on GPU $gpu_id with mvalid=($md1, $md2)"

      CUDA_VISIBLE_DEVICES=$gpu_id \
      RANDOM_YV=1 \
      SAVE_TSNE_MODEL=0 \
      HYPER_SWEEP=1 \
      START=$START \
      STOP=$STOP \
      MVALID_DEFAULT=HPO_FINAL_STRANS \
      PYTHONPATH=. \
      python main_merck_all_real_nk1_n_context.py \
        --sencoder "$sencoder" \
        --sencoder_layer "$sencoder_layer" \
        --model mlp \
        --mixer_phi True \
        --seed 42 \
        --wd 1e-4 \
        --cwd 1e-3 \
        --ln False \
        --dropout 0.5 \
        --batch_size 64 \
        --batchnorm False \
        --outer_episodes 50 \
        --inner_episodes 10 \
        --initialize_weights False \
        --early_stopping_episodes 10 \
        --in_features 6560 \
        --num_outputs 1 \
        --same_setting \
        --mvalid_dataset "$md1" "$md2" \
        > logs/hyper_job_${md1}_${md2}_${START}_${STOP}_hpo.log 2>&1 &

      ((job_id++))
    done
  done
done

wait
