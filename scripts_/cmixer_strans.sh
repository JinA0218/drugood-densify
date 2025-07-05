sencoder=strans
sencoder_layer=pma

for split in spectral random scaffold weight;
do
    for fingerprint in rdkit ecfp;
    do
        # --optimizer adamw \
        CUDA_VISIBLE_DEVICES=0 TUNED_FINAL=1 PYTHONPATH=. python main_origin.py \
          --root data/ \
          --model mlp \
          --mixer_phi True \
          --optimizer adam \
          --split $split \
          --fingerprint $fingerprint \
          --seed 42 \
          --lr 1e-3 \
          --wd 1e-3 \
          --clr 1e-4 \
          --cwd 1e-3 \
          --ln False \
          --dropout 0.5 \
          --num_layers 1 \
          --hidden_dim 32 \
          --batch_size 64 \
          --n_context 4 \
          --batchnorm False \
          --outer_episodes 50 \
          --inner_episodes 20 \
          --initialize_weights False \
          --sencoder $sencoder \
          --sencoder_layer $sencoder_layer \
          --early_stopping_episodes 100
    done
done
