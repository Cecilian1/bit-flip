#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
exec uv run --frozen python ./reproduce_fresh_oneflip.py --clean-checkpoint ./saved_model/resnet_CIFAR10/clean_model_1.pth --data-root ../dataset/CIFAR10 --output-dir ./runs/fresh-cifar10-class6 --device "${DEVICE:-cuda:0}" --target-class 6 --calibration-size 1024 --max-bad 0.001 --trigger-epochs 500 --optimization-batch-size 128 --mask-weight 0.001 --attack-threshold 1.0 --test-batch-size 512 --workers 2 --seed 20260904
