# #!/bin/bash
mkdir -p logs

targets=('hivprot' 'dpp4' 'nk1') #    'dpp4' 'nk1'
sencoders=('dsets' 'strans') #    'strans'
mixing_layers=(0)  #  1 2


gpus=(1 2 4)
gpu_count=${#gpus[@]}
max_jobs_per_gpu=6
job_id=0

# embed_tests=('2nd_last_ours_best') #  'setenc_ours_best' lastlayer_ours_best
# embed_tests=('lastlayer_ours_best' 'lastlayer_base_cX_mO' 'setenc_ours_best' 'setenc_base_cX_mO' 'lastlayer_base_cX_mX')

active_pids=()

for ds in "${targets[@]}"; do
    for vt in count bit; do #  bit
        for sencoder in "${sencoders[@]}"; do
            # for sl in mean max sum; do
            for ml in "${mixing_layers[@]}"; do
                gpu_id=${gpus[$((job_id % gpu_count))]}

                echo "Launching job $job_id on GPU $gpu_id | ds=$ds | md=($md1,$md2) | vt=$vt | sencoder=$sencoder | mixing_layer=$ml"
                mix_default_type="xmix"
                MIXER_NO_BILEVEL_EPOCHS=50 SET_NO_BILEVEL_MIX_LABEL=true_y MIX_TYPE=SET_NO_BILEVEL RANDOM_YV=1 SAVE_TSNE_MODEL=0 MVALID_DEFAULT=SET_NO_BILEVEL_ep50 MIXING_X_DEFAULT=$mix_default_type CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python main_merck_all_real.py \
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
                    --mvalid_dataset None \
                    --mixer_phi True \
                    --sencoder_layer max \
                    --mixing_layer "$ml" \
                    > logs/job_${job_id}_ds_${ds}_md_${md1}_${md2}_vt_${vt}_se_${sencoder}_mt_${mix_default_type}_ml_${ml// /_}_mixup_final_NO_BUG_ep${MIXER_NO_BILEVEL_EPOCHS}.log 2>&1 &

                ((job_id++))

                # if [[ $job_id -ge 1 ]]; then
                #     exit 0
                # fi

                if (( job_id % (gpu_count * 3) == 0 )); then
                    wait
                fi
                # done
            done
        done
    done
done

# Final wait for any remaining jobs
wait

echo "All jobs completed."




