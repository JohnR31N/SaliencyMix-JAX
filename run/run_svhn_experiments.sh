#!/bin/bash

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

EPOCHS=${1:-200}
MODELS=("wideresnet" "resnet18" "resnet50")
METHODS=("baseline" "saliencymix")
SEEDS=("0" "1" "2")

python scripts/saliencymap.py --dataset svhn --data_dir data

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      bash run/run_experiment.sh svhn "${model}" "${method}" "${seed}" "${EPOCHS}" plus
    done
  done
done
