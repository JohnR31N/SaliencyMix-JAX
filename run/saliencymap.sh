#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=.

python scripts/saliencymap.py