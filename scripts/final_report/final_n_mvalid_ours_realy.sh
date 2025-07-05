#!/bin/bash

targets=('hivprot' 'dpp4' 'nk1') # 'hivprot' 'dpp4' 
sencoders=('dsets' 'strans') #  'dsets'
max_parallel_jobs=3
gpu_list=(1 6)   # You can extend this: (0 1 2 3)
num_gpus=${#gpu_list[@]}
job_id=0

mkdir -p logs

# Function to determine sencoder_layer
get_sencoder_layer() {
  local encoder=$1
  if [[ "$encoder" == "dsets" ]]; then
    echo "max"
  elif [[ "$encoder" == "strans" ]]; then
    echo "sum"
  else
    echo "unknown"
  fi
}

# N_MVALID_FINAL_4_SENC

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
  RANDOM_YV=0 \
  SAVE_TSNE_MODEL=0 \
  MVALID_DEFAULT=N_MVALID_OURS_REALY_NL3_HD64_camera \
  PYTHONPATH=. \
  python main_merck_all_real_nk1_n_context_n_mvalid_realy.py \
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
    > logs/hyper_job_${se}_${sl}_${ds}_${vt}_${jid}_realy_0.log 2>&1 &

  ((job_id++))
}

# Main loop
for ds in "${targets[@]}"; do
  for vt in count bit; do # count
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
