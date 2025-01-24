# for lr in 1e-3 5e-4 1e-4 5e-5 1e-5;
# do
#     for clr in 1e-4 5e-5 1e-5;
#     do
#         for ds in hivprot dpp4 nk1;
#         do
#             for vt in count bit;
#             do
#                 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python main_merck.py \
#                   --model mlp \
#                   --mixer_phi True \
#                   --optimizer adamwschedulefree \
#                   --seed 42 \
#                   --lr $lr \
#                   --wd 0.0 \
#                   --clr $clr \
#                   --cwd 5e-4 \
#                   --dropout 0.5 \
#                   --num_layers 1 \
#                   --hidden_dim 32 \
#                   --batch_size 64 \
#                   --batchnorm False \
#                   --outer_episodes 500 \
#                   --inner_episodes 10 \
#                   --initialize_weights False \
#                   --early_stopping_episodes 10 \
#                   --n_context 4 \
#                   --in_features 6560 \
#                   --num_outputs 1 \
#                   --vec_type $vt \
#                   --dataset $ds
#             done
#         done
#     done
# done

# for lr in 1e-3 5e-4 1e-4 5e-5 1e-5;
# do
#     for clr in 1e-3 5e-4;
#     do
#         for ds in hivprot dpp4 nk1;
#         do
#             for vt in count bit;
#             do
#                 CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python main_merck.py \
#                   --model mlp \
#                   --mixer_phi True \
#                   --optimizer adamwschedulefree \
#                   --seed 42 \
#                   --lr $lr \
#                   --wd 0.0 \
#                   --clr $clr \
#                   --cwd 5e-4 \
#                   --dropout 0.5 \
#                   --num_layers 1 \
#                   --hidden_dim 32 \
#                   --batch_size 64 \
#                   --batchnorm False \
#                   --outer_episodes 500 \
#                   --inner_episodes 10 \
#                   --initialize_weights False \
#                   --early_stopping_episodes 10 \
#                   --n_context 4 \
#                   --in_features 6560 \
#                   --num_outputs 1 \
#                   --vec_type $vt \
#                   --dataset $ds
#             done
#         done
#     done
# done
#

# CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python main_merck.py \
#   --model mlp \
#   --mixer_phi True \
#   --optimizer adamwschedulefree \
#   --seed 42 \
#   --lr 1e-3 \
#   --wd 0.0 \
#   --clr 1e-5 \
#   --cwd 5e-4 \
#   --dropout 0.5 \
#   --num_layers 1 \
#   --hidden_dim 32 \
#   --batch_size 64 \
#   --batchnorm False \
#   --outer_episodes 500 \
#   --inner_episodes 10 \
#   --initialize_weights False \
#   --early_stopping_episodes 10 \
#   --n_context 4 \
#   --in_features 6560 \
#   --num_outputs 1 \
#   --vec_type count \
#   --dataset hivprot

for ds in hivprot dpp4 nk1;
do
    for vt in bit count;
    do
        # --cwd 5e-4 \
        CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python main_merck.py \
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
          --n_context 8 \
          --in_features 6560 \
          --num_outputs 1 \
          --vec_type $vt \
          --dataset $ds
    done
done
