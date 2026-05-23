#!/bin/bash

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

DATASET=${1:-cifar10}          # cifar10 / cifar100 / svhn
MODEL=${2:-resnet18}           # resnet18 / resnet34 / resnet50 / resnet101 / wideresnet
METHOD=${3:-saliencymix}       # baseline / saliencymix
SEED=${4:-0}
EPOCHS=${5:-200}
AUG=${6:-plus}                 # plus / noaug

AUG_FLAG=""
DATASET_NAME="${DATASET}"

if [ "${AUG}" = "plus" ]; then
  AUG_FLAG="--data_augmentation"
  DATASET_NAME="${DATASET}_plus"
elif [ "${AUG}" = "noaug" ]; then
  AUG_FLAG=""
else
  echo "Unknown AUG=${AUG}. Use plus or noaug."
  exit 1
fi

python train.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --method "${METHOD}" \
  --epochs "${EPOCHS}" \
  --batch_size 128 \
  --learning_rate 0.1 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --beta 1.0 \
  --salmix_prob 0.5 \
  ${AUG_FLAG} \
  --seed "${SEED}" \
  --checkpoint_dir "checkpoints/${MODEL}/${DATASET_NAME}/${METHOD}_seed${SEED}" \
  --outcome_dir outcome \
  --figure_dir figures
