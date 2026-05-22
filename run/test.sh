#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=.

python test.py \
  --checkpoint_dir checkpoints/resnet18_saliencymix \
  --batch_size 128