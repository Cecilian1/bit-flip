#!/usr/bin/env bash
set -euo pipefail

phase="${1:-tune-coarse}"
resume_args=()
if [[ "${2:-}" == "--resume" ]]; then resume_args=(--resume); fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
device="${DEVICE:-cuda:0}"
if [[ -n "${PYTHON_EXECUTABLE:-}" ]]; then
  python_runner=("$PYTHON_EXECUTABLE")
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  python_runner=(python)
else
  python_runner=(uv run --frozen python)
fi
base=("${python_runner[@]}" ./causal_oneflip.py --method causal --dataset CIFAR10 --clean-checkpoint ./saved_model/resnet_CIFAR10/clean_model_1.pth --data-root ../dataset/CIFAR10 --device "$device" --target-classes 6 --calibration-size 1024 --optimization-size 768 --optimization-batch-size 128 --test-batch-size 512 --workers 2)

case "$phase" in
  tune-coarse|tune)
    for margin in 0.5 1 2; do
      "${base[@]}" --output-dir "./runs/causaloneflip-tune-class6-coarse-${margin}" --betas 2,4,8,16 --mask-weights 0.001,0.002,0.005 --clean-margin "$margin" --attack-margin 0.5 --trigger-epochs 150 --limit-candidates 12 --seeds 20260904 "${resume_args[@]}"
    done
    mkdir -p ./runs/causaloneflip-tune-class6-coarse
    "${python_runner[@]}" ./select_causaloneflip_params.py --results ./runs/causaloneflip-tune-class6-coarse-0.5/results.csv --results ./runs/causaloneflip-tune-class6-coarse-1/results.csv --results ./runs/causaloneflip-tune-class6-coarse-2/results.csv --output-dir ./runs/causaloneflip-tune-class6-coarse
    ;;
  tune-fine)
    "${python_runner[@]}" - "${python_runner[@]}" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
root = Path('.')
shortlist = json.loads((root/'runs/causaloneflip-tune-class6-coarse/selected_params.json').read_text())['shortlist']
runner = sys.argv[1:]
for index, item in enumerate(shortlist, 1):
    command = runner + ['./causal_oneflip.py', '--method', 'causal', '--dataset', 'CIFAR10', '--clean-checkpoint', './saved_model/resnet_CIFAR10/clean_model_1.pth', '--data-root', '../dataset/CIFAR10', '--output-dir', f'./runs/causaloneflip-tune-class6-fine-{index}', '--device', os.environ.get('DEVICE', 'cuda:0'), '--target-classes', '6', '--betas', str(item['beta']), '--mask-weights', str(item['mask_weight']), '--clean-margin', str(item['clean_margin']), '--attack-margin', str(item['attack_margin']), '--seeds', '20260904,20260905,20260906', '--calibration-size', '1024', '--optimization-size', '768', '--trigger-epochs', '500', '--optimization-batch-size', '128', '--test-batch-size', '512', '--workers', '2']
    subprocess.run(command, check=True)
PY
    "${python_runner[@]}" ./select_causaloneflip_params.py --results ./runs/causaloneflip-tune-class6-fine-1/results.csv --results ./runs/causaloneflip-tune-class6-fine-2/results.csv --results ./runs/causaloneflip-tune-class6-fine-3/results.csv --output-dir ./runs/causaloneflip-tune-class6
    ;;
  select-params)
    mkdir -p ./runs/causaloneflip-tune-class6
    "${python_runner[@]}" ./select_causaloneflip_params.py --results ./runs/causaloneflip-tune-class6-fine-1/results.csv --results ./runs/causaloneflip-tune-class6-fine-2/results.csv --results ./runs/causaloneflip-tune-class6-fine-3/results.csv --output-dir ./runs/causaloneflip-tune-class6
    ;;
  causal-main|oneflip-baseline|tuap-baseline|report-cifar10|main-all)
    "${python_runner[@]}" ./run_causaloneflip_main.py "$phase" --device "$device" --target-classes "${TARGET_CLASSES:-0,1,2,3,4,5,6,7,8,9}" --seeds "${MAIN_SEEDS:-20260904,20260905,20260906}" --workers "${MAIN_WORKERS:-2}" "${resume_args[@]}"
    ;;
  *) echo "Usage: $0 {tune-coarse|tune-fine|select-params|causal-main|oneflip-baseline|tuap-baseline|report-cifar10|main-all} [--resume]" >&2; exit 2 ;;
esac