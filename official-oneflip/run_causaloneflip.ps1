param(
    [ValidateSet('tune-coarse', 'tune-fine', 'select-params')]
    [string]$Phase = 'tune-coarse',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$device = if ($env:DEVICE) { $env:DEVICE } else { 'cuda:0' }
$resumeArgs = if ($Resume) { @('--resume') } else { @() }

if ($Phase -eq 'tune-coarse') {
    foreach ($margin in @('0.5', '1', '2')) {
        uv run --frozen python .\causal_oneflip.py --method causal --dataset CIFAR10 --clean-checkpoint .\saved_model\resnet_CIFAR10\clean_model_1.pth --data-root ..\dataset\CIFAR10 --output-dir ".\runs\causaloneflip-tune-class6-coarse-$margin" --device $device --target-classes 6 --betas 2,4,8,16 --mask-weights 0.001,0.002,0.005 --clean-margin $margin --attack-margin 0.5 --seeds 20260904 --calibration-size 1024 --optimization-size 768 --trigger-epochs 150 --optimization-batch-size 128 --test-batch-size 512 --workers 2 --limit-candidates 12 @resumeArgs
    }
    New-Item -ItemType Directory -Force -Path .\runs\causaloneflip-tune-class6-coarse | Out-Null
    uv run --frozen python .\select_causaloneflip_params.py --results .\runs\causaloneflip-tune-class6-coarse-0.5\results.csv --results .\runs\causaloneflip-tune-class6-coarse-1\results.csv --results .\runs\causaloneflip-tune-class6-coarse-2\results.csv --output-dir .\runs\causaloneflip-tune-class6-coarse
    exit $LASTEXITCODE
}

if ($Phase -eq 'tune-fine') {
    $shortlist = (Get-Content .\runs\causaloneflip-tune-class6-coarse\selected_params.json -Raw | ConvertFrom-Json).shortlist
    $index = 0
    foreach ($item in $shortlist) {
        $index++
        uv run --frozen python .\causal_oneflip.py --method causal --dataset CIFAR10 --clean-checkpoint .\saved_model\resnet_CIFAR10\clean_model_1.pth --data-root ..\dataset\CIFAR10 --output-dir ".\runs\causaloneflip-tune-class6-fine-$index" --device $device --target-classes 6 --betas $item.beta --mask-weights $item.mask_weight --clean-margin $item.clean_margin --attack-margin $item.attack_margin --seeds 20260904,20260905,20260906 --calibration-size 1024 --optimization-size 768 --trigger-epochs 500 --optimization-batch-size 128 --test-batch-size 512 --workers 2 --limit-candidates 12 @resumeArgs
    }
    New-Item -ItemType Directory -Force -Path .\runs\causaloneflip-tune-class6 | Out-Null
    uv run --frozen python .\select_causaloneflip_params.py --results .\runs\causaloneflip-tune-class6-fine-1\results.csv --results .\runs\causaloneflip-tune-class6-fine-2\results.csv --results .\runs\causaloneflip-tune-class6-fine-3\results.csv --output-dir .\runs\causaloneflip-tune-class6
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path .\runs\causaloneflip-tune-class6 | Out-Null
uv run --frozen python .\select_causaloneflip_params.py --results .\runs\causaloneflip-tune-class6-fine-1\results.csv --results .\runs\causaloneflip-tune-class6-fine-2\results.csv --results .\runs\causaloneflip-tune-class6-fine-3\results.csv --output-dir .\runs\causaloneflip-tune-class6
