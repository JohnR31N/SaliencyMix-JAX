#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=.

for seed in 0 1 2
do
  echo "Running seed ${seed}"

  python train.py \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --beta 1.0 \
    --salmix_prob 0.5 \
    --data_augmentation \
    --seed ${seed} \
    --checkpoint_dir checkpoints/resnet18_saliencymix_seed${seed} \
    --log_dir logs
done