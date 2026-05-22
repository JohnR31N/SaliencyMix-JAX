#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=.

python train.py \
  --epochs 10 \
  --batch_size 128 \
  --learning_rate 0.1 \
  --beta 1.0 \
  --salmix_prob 0.5 \
  --checkpoint_dir checkpoints/resnet18_saliencymix