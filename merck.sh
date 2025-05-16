# sencoder=dsets
# sencoder_layer=max
# sencoder=strans
# sencoder_layer=pma

# CUDA_VISIBLE_DEVICES=3 HYPER_SWEEP=1 START=0 STOP=24 PYTHONPATH=. python main_merck.py \
#   --model mlp \
#   --mixer_phi True \
#   --optimizer adamwschedulefree \
#   --wd 1e-4 \
#   --cwd 1e-3 \
#   --batch_size 64 \
#   --batchnorm False \
#   --outer_episodes 50 \
#   --inner_episodes 10 \
#   --initialize_weights False \
#   --in_features 6560 \
#   --sencoder $sencoder \
#   --sencoder_layer $sencoder_layer \
#   --num_outputs 1 \

# dataset_names = ['3a4', 'cb1', 'dpp4', 'hivint', 'hivprot', 'logd', 'metab', 'nk1', 'ox1', 'ox2', 'pgp', 'ppb', 'rat_f', 'tdi', 'thrombin']

for ds in hivprot; #  dpp4 nk1
do
    for vt in count bit;
    do
        for sencoder in dsets;
        do
            # for sl in mean max sum pma;
            # do 
                # --cwd 5e-4 \
            CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main_merck.py \
            --sencoder $sencoder \
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
            --vec_type $vt \
            --dataset $ds \
            --mvalid_dataset count bit\
            --same_setting \
            # --sencoder_layer $sl
            # done
        done
    done
done

# dataset_names=('3a4' 'cb1' 'dpp4' 'hivint' 'hivprot' 'logd' 'metab' 'nk1' 'ox1' 'ox2' 'pgp' 'ppb' 'rat_f' 'tdi' 'thrombin')

# for ds in "${dataset_names[@]}"; do
#     # Build a filtered list excluding $ds
#     filtered=()
#     for name in "${dataset_names[@]}"; do
#         if [[ "$name" != "$ds" ]]; then
#             filtered+=("$name")
#         fi
#     done

#     # Generate all 2-element unique combinations from `filtered`
#     for ((i=0; i<${#filtered[@]}-1; i++)); do
#         for ((j=i+1; j<${#filtered[@]}; j++)); do
#             md1="${filtered[i]}"
#             md2="${filtered[j]}"
#             md=("$md1" "$md2")

#             for vt in count bit; do
#                 for sencoder in dsets; do
#                     echo "Running with ds=$ds, md=(${md[*]}), vt=$vt, sencoder=$sencoder"

#                     # Your training command
#                     PYTHONPATH=. python main_merck.py \
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
#                         --mvalid_dataset "${md[0]}" "${md[1]}" \ 
#                 done
#             done
#         done
#     done
# done

