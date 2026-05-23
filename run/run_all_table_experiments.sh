#!/bin/bash

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

EPOCHS=${1:-200}

# Paper table subset:
# CIFAR-10 / CIFAR-10+ / CIFAR-100 / CIFAR-100+
# ResNet-18, ResNet-50, WideResNet-28-10
# baseline vs SaliencyMix
DATASETS=("cifar10" "cifar100")
AUGS=("noaug" "plus")
MODELS=("resnet18" "resnet50" "wideresnet")
METHODS=("baseline" "saliencymix")
SEEDS=("0" "1" "2")

# Ensure saliency maps exist. Baseline still uses the same loader, so this is needed.
for dataset in "${DATASETS[@]}"; do
  echo "Preparing saliency maps for ${dataset}"
  python scripts/saliencymap.py --dataset "${dataset}" --data_dir data
done

for dataset in "${DATASETS[@]}"; do
  for aug in "${AUGS[@]}"; do
    for model in "${MODELS[@]}"; do
      for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          echo "============================================================"
          echo "Running dataset=${dataset}, aug=${aug}, model=${model}, method=${method}, seed=${seed}, epochs=${EPOCHS}"
          echo "============================================================"

          bash run/run_experiment.sh "${dataset}" "${model}" "${method}" "${seed}" "${EPOCHS}" "${aug}"
        done
      done
    done
  done
done
