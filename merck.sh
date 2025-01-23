CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python main_merck.py \
  --model mlp \
  --mixer_phi True \
  --optimizer adamwschedulefree \
  --seed 42 \
  --lr 1e-3 \
  --wd 0.0 \
  --clr 1e-5 \
  --cwd 5e-4 \
  --dropout 0.5 \
  --num_layers 1 \
  --hidden_dim 32 \
  --batch_size 64 \
  --batchnorm False \
  --outer_episodes 500 \
  --inner_episodes 10 \
  --initialize_weights False \
  --early_stopping_episodes 10 \
  --n_context 4 \
  --in_features 6560 \
  --num_outputs 1 \
  --vec_type count \
  --dataset hivprot

# pretty good with these
# CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python main_merck.py \
#   --model mlp2 \
#   --mixer_phi True \
#   --optimizer adamwschedulefree \
#   --seed 42 \
#   --lr 5e-4 \
#   --wd 1e-5 \
#   --clr 5e-4 \
#   --cwd 1e-5 \
#   --dropout 0.5 \
#   --num_layers 2 \
#   --hidden_dim 32 \
#   --batch_size 128 \
#   --batchnorm False \
#   --outer_episodes 50 \
#   --inner_episodes 10 \
#   --initialize_weights False \
#   --early_stopping_episodes 100 \
#   --n_context 1 \
#   --in_features 6560 \
#   --num_outputs 1 \
#   --vec_type count \
#   --dataset hivprot

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main_merck.py \
#   --model mlp \
#   --mixer_phi False \
#   --optimizer adamwschedulefree \
#   --seed 42 \
#   --lr 1e-3 \
#   --wd 1e-2 \
#   --clr 1e-3 \
#   --cwd 1e-5 \
#   --dropout 0.2 \
#   --num_layers 2 \
#   --hidden_dim 32 \
#   --batch_size 128 \
#   --batchnorm False \
#   --outer_episodes 30 \
#   --inner_episodes 50 \
#   --initialize_weights False \
#   --early_stopping_episodes 100 \
#   --in_features 6560 \
#   --num_outputs 1 \
#   --vec_type bit \
#   --dataset hivprot
