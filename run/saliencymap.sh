#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=.

DATASET=${1:-cifar10}

python scripts/saliencymap.py \
  --dataset ${DATASET} \
  --data_dir data