sencoder=dsets
sencoder_layer=max
sencoder=strans
sencoder_layer=pma

CUDA_VISIBLE_DEVICES=3 HYPER_SWEEP=1 START=0 STOP=24 PYTHONPATH=. python main_merck.py \
  --model mlp \
  --mixer_phi True \
  --optimizer adamwschedulefree \
  --wd 1e-4 \
  --cwd 1e-3 \
  --batch_size 64 \
  --batchnorm False \
  --outer_episodes 50 \
  --inner_episodes 10 \
  --initialize_weights False \
  --in_features 6560 \
  --sencoder $sencoder \
  --sencoder_layer $sencoder_layer \
  --num_outputs 1 \

# for ds in hivprot dpp4 nk1;
# do
#     for vt in count bit;
#     do
#         for sencoder in dsets;
#         do
#             # for sl in mean max sum pma;
#             # do 
#                 # --cwd 5e-4 \
#             CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main_merck.py \
#             --sencoder $sencoder \
#             --model mlp \
#             --mixer_phi True \
#             --optimizer adamwschedulefree \
#             --seed 42 \
#             --lr 1e-3 \
#             --wd 1e-4 \
#             --clr 1e-5 \
#             --cwd 1e-3 \
#             --dropout 0.5 \
#             --num_layers 3 \
#             --hidden_dim 32 \
#             --batch_size 64 \
#             --batchnorm False \
#             --outer_episodes 50 \
#             --inner_episodes 10 \
#             --initialize_weights False \
#             --early_stopping_episodes 10 \
#             --n_context 4 \
#             --in_features 6560 \
#             --num_outputs 1 \
#             --vec_type $vt \
#             --dataset $ds \
#             # --sencoder_layer $sl
#             # done
#         done
#     done
# done
