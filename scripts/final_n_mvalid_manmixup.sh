#!/bin/bash

targets=('nk1') # 'hivprot' 'dpp4' 
sencoders=('strans') #  ==manmixup
max_parallel_jobs=6
gpu_list=(2 6)   # You can extend this: (0 1 2 3)
num_gpus=${#gpu_list[@]}
job_id=0

mkdir -p logs

# Function to determine sencoder_layer
get_sencoder_layer() {
  local encoder=$1
  if [[ "$encoder" == "strans" ]]; then
    echo "max"
  # elif [[ "$encoder" == "strans" ]]; then
  #   echo "sum"
  else
    echo "unknown"
  fi
}

# Function to launch a job
launch_job() {
  local ds=$1
  local vt=$2
  local se=$3
  local jid=$4
  local sl=$(get_sencoder_layer "$se")
  local gpu_id=${gpu_list[$((jid % num_gpus))]}

  if [[ "$sl" == "unknown" ]]; then
    echo "Invalid sencoder: $se"
    return
  fi

  echo "Launching job $jid on GPU $gpu_id | ds=$ds | vt=$vt | sencoder=$se | sencoder_layer=$sl"

  CUDA_VISIBLE_DEVICES=$gpu_id \
  MIXUP_EPOCHS=10 \
  MIX_TYPE=MANIFOLD_MIXUP \
  RANDOM_YV=1 \
  SAVE_TSNE_MODEL=0 \
  MVALID_DEFAULT=N_MVALID_MANIFOLD_MIXUP_NL3_HD64 \
  PYTHONPATH=. \
  python main_merck_all_real_mixup_no_bilevel_n_context_n_mvalid.py \
    --sencoder "$se" \
    --sencoder_layer "$sl" \
    --optimizer adamwschedulefree \
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
    --vec_type "$vt" \
    --dataset "$ds" \
    --same_setting \
    --mvalid_dataset None \
    > logs/hyper_job_${se}_${sl}_${ds}_${vt}_${jid}.log 2>&1 &

  ((job_id++))
}

# Main loop
for ds in "${targets[@]}"; do
  for vt in count; do # bit
    for se in "${sencoders[@]}"; do
      launch_job "$ds" "$vt" "$se" "$job_id"

      # Throttle active background jobs
      while (( $(jobs -rp | wc -l) >= max_parallel_jobs )); do
        sleep 1
      done
    done
  done
done

wait  # Wait for all jobs to finish
