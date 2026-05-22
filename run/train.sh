#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=.

DATASET=${1:-cifar10}
MODEL=${2:-resnet18}
SEED=${3:-0}
EPOCHS=${4:-200}

python train.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --epochs ${EPOCHS} \
  --batch_size 128 \
  --learning_rate 0.1 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --beta 1.0 \
  --salmix_prob 0.5 \
  --data_augmentation \
  --seed ${SEED} \
  --checkpoint_dir checkpoints/${DATASET}_${MODEL}_saliencymix_seed${SEED} \
  --log_dir logs \
  --figure_dir figures