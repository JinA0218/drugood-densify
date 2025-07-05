#!/bin/bash

mkdir -p logs

sencoder="dsets"
sencoder_layer="max"

total_configs=9
num_jobs=3
configs_per_job=$((total_configs / num_jobs))
gpus=(3 4 5)
job_id=0
max_parallel_jobs=3  # controls how many jobs run concurrently

for ((i = 0; i < num_jobs; i++)); do
  START=$((i * configs_per_job))
  STOP=$(((i + 1) * configs_per_job))
  gpu_id=${gpus[$((job_id % ${#gpus[@]}))]}

  echo "Launching sweep $START to $STOP on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id \
  RANDOM_YV=1 \
  SAVE_TSNE_MODEL=0 \
  HYPER_SWEEP=1 \
  START=$START \
  STOP=$STOP \
  MVALID_DEFAULT=HPO_FINAL_N_MVALID \
  PYTHONPATH=. \
  python main_merck_all_real_nk1_n_context_n_mvalid.py \
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
    --mvalid_dataset None \
    --num_workers 1 \
    > logs/hyper_job_${sencoder}_${sencoder_layer}_${START}_${STOP}_dsets_final.log 2>&1 &

  ((job_id++))

  # Limit concurrent background jobs to prevent open file errors
  if (( job_id % max_parallel_jobs == 0 )); then
    wait
  fi
done

wait

### without multiprocessing

# mkdir -p logs

# sencoder="dsets"
# sencoder_layer="max"

# total_configs=48
# num_jobs=6
# configs_per_job=$((total_configs / num_jobs))
# gpus=(3)  # Will still cycle GPUs, but one job at a time
# job_id=0

# for ((i = 0; i < num_jobs; i++)); do
#   START=$((i * configs_per_job))
#   STOP=$(((i + 1) * configs_per_job))
#   gpu_id=${gpus[$((job_id % ${#gpus[@]}))]}

#   echo "Launching sweep $START to $STOP on GPU $gpu_id"

#   CUDA_VISIBLE_DEVICES=$gpu_id \
#   RANDOM_YV=1 \
#   SAVE_TSNE_MODEL=0 \
#   HYPER_SWEEP=1 \
#   START=$START \
#   STOP=$STOP \
#   MVALID_DEFAULT=HPO_FINAL_N_MVALID \
#   PYTHONPATH=. \
#   python main_merck_all_real_nk1_n_context_n_mvalid_setenc_rest_final.py \
#     --sencoder "$sencoder" \
#     --sencoder_layer "$sencoder_layer" \
#     --model mlp \
#     --mixer_phi True \
#     --seed 42 \
#     --wd 1e-4 \
#     --cwd 1e-3 \
#     --ln False \
#     --dropout 0.5 \
#     --batch_size 64 \
#     --batchnorm False \
#     --outer_episodes 50 \
#     --inner_episodes 10 \
#     --initialize_weights False \
#     --early_stopping_episodes 10 \
#     --in_features 6560 \
#     --num_outputs 1 \
#     --same_setting \
#     --mvalid_dataset None \
#     --num_workers 0 \
#     > logs/hyper_job_${sencoder}_${sencoder_layer}_${START}_${STOP}_dsets_final.log 2>&1

#   ((job_id++))
# done








#!/bin/bash
# mkdir -p logs

# sencoder="dsets"
# sencoder_layer="max"

# total_configs=72
# num_jobs=12
# configs_per_job=$((total_configs / num_jobs))
# gpus=(2)
# job_id=0
# max_parallel_jobs=5  # controls how many jobs run concurrently

# # for ((i = 0; i < num_jobs; i++)); do
#   START=23
#   STOP=24
#   gpu_id=${gpus[$((job_id % ${#gpus[@]}))]}

#   echo "Launching sweep $START to $STOP on GPU $gpu_id"

#   CUDA_VISIBLE_DEVICES=$gpu_id \
#   RANDOM_YV=1 \
#   SAVE_TSNE_MODEL=0 \
#   HYPER_SWEEP=1 \
#   START=$START \
#   STOP=$STOP \
#   MVALID_DEFAULT=HPO_FINAL_N_MVALID \
#   PYTHONPATH=. \
#   python main_merck_all_real_nk1_n_context_n_mvalid.py \
#     --sencoder "$sencoder" \
#     --sencoder_layer "$sencoder_layer" \
#     --model mlp \
#     --mixer_phi True \
#     --seed 42 \
#     --wd 1e-4 \
#     --cwd 1e-3 \
#     --ln False \
#     --dropout 0.5 \
#     --batch_size 64 \
#     --batchnorm False \
#     --outer_episodes 50 \
#     --inner_episodes 10 \
#     --initialize_weights False \
#     --early_stopping_episodes 10 \
#     --in_features 6560 \
#     --num_outputs 1 \
#     --same_setting \
#     --mvalid_dataset None 
#     > logs/hyper_job_${sencoder}_${sencoder_layer}_${START}_${STOP}_dsets.log 2>&1 &

#   ((job_id++))

#   # Limit concurrent background jobs to prevent open file errors
#   if (( job_id % max_parallel_jobs == 0 )); then
#     wait
#   fi
# # done

# wait
