sencoder=strans
sencoder_layer=pma

 # --optimizer adamw \
CUDA_VISIBLE_DEVICES=4 HYPER_SWEEP=1 START=0 STOP=75 PYTHONPATH=. python main.py \
  --root data/ \
  --model mlp \
  --mixer_phi True \
  --optimizer adam \
  --seed 42 \
  --wd 1e-3 \
  --cwd 1e-3 \
  --ln False \
  --dropout 0.5 \
  --batch_size 64 \
  --batchnorm False \
  --outer_episodes 50 \
  --inner_episodes 20 \
  --initialize_weights False \
  --sencoder $sencoder \
  --sencoder_layer $sencoder_layer
