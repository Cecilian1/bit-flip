param(
    [ValidateSet('tune-coarse', 'tune-fine', 'select-params', 'causal-main', 'oneflip-baseline', 'tuap-baseline', 'report-cifar10', 'main-all')]
    [string]$Phase = 'tune-coarse',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$device = if ($env:DEVICE) { $env:DEVICE } else { 'cuda:0' }
$resumeArgs = if ($Resume) { @('--resume') } else { @() }
if ($env:PYTHON_EXECUTABLE) {
    $pythonCommand = $env:PYTHON_EXECUTABLE
    $pythonPrefix = @()
} elseif ($env:VIRTUAL_ENV) {
    $pythonCommand = 'python'
    $pythonPrefix = @()
} else {
    $pythonCommand = 'uv'
    $pythonPrefix = @('run', '--frozen', 'python')
}
function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $script:pythonCommand @script:pythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE." }
}

if ($Phase -eq 'tune-coarse') {
    foreach ($margin in @('0.5', '1', '2')) {
        Invoke-ProjectPython .\causal_oneflip.py --method causal --dataset CIFAR10 --clean-checkpoint .\saved_model\resnet_CIFAR10\clean_model_1.pth --data-root ..\dataset\CIFAR10 --output-dir ".\runs\causaloneflip-tune-class6-coarse-$margin" --device $device --target-classes 6 --betas 2,4,8,16 --mask-weights 0.001,0.002,0.005 --clean-margin $margin --attack-margin 0.5 --seeds 20260904 --calibration-size 1024 --optimization-size 768 --trigger-epochs 150 --optimization-batch-size 128 --test-batch-size 512 --workers 2 --limit-candidates 12 @resumeArgs
    }
    New-Item -ItemType Directory -Force -Path .\runs\causaloneflip-tune-class6-coarse | Out-Null
    Invoke-ProjectPython .\select_causaloneflip_params.py --results .\runs\causaloneflip-tune-class6-coarse-0.5\results.csv --results .\runs\causaloneflip-tune-class6-coarse-1\results.csv --results .\runs\causaloneflip-tune-class6-coarse-2\results.csv --output-dir .\runs\causaloneflip-tune-class6-coarse
    exit
}

if ($Phase -eq 'tune-fine') {
    $shortlist = (Get-Content .\runs\causaloneflip-tune-class6-coarse\selected_params.json -Raw | ConvertFrom-Json).shortlist
    $index = 0
    foreach ($item in $shortlist) {
        $index++
        Invoke-ProjectPython .\causal_oneflip.py --method causal --dataset CIFAR10 --clean-checkpoint .\saved_model\resnet_CIFAR10\clean_model_1.pth --data-root ..\dataset\CIFAR10 --output-dir ".\runs\causaloneflip-tune-class6-fine-$index" --device $device --target-classes 6 --betas $item.beta --mask-weights $item.mask_weight --clean-margin $item.clean_margin --attack-margin $item.attack_margin --seeds 20260904,20260905,20260906 --calibration-size 1024 --optimization-size 768 --trigger-epochs 500 --optimization-batch-size 128 --test-batch-size 512 --workers 2 @resumeArgs
    }
    New-Item -ItemType Directory -Force -Path .\runs\causaloneflip-tune-class6 | Out-Null
    Invoke-ProjectPython .\select_causaloneflip_params.py --results .\runs\causaloneflip-tune-class6-fine-1\results.csv --results .\runs\causaloneflip-tune-class6-fine-2\results.csv --results .\runs\causaloneflip-tune-class6-fine-3\results.csv --output-dir .\runs\causaloneflip-tune-class6
    exit
}

if ($Phase -eq 'select-params') {
    New-Item -ItemType Directory -Force -Path .\runs\causaloneflip-tune-class6 | Out-Null
    Invoke-ProjectPython .\select_causaloneflip_params.py --results .\runs\causaloneflip-tune-class6-fine-1\results.csv --results .\runs\causaloneflip-tune-class6-fine-2\results.csv --results .\runs\causaloneflip-tune-class6-fine-3\results.csv --output-dir .\runs\causaloneflip-tune-class6
    exit
}

$targetClasses = if ($env:TARGET_CLASSES) { $env:TARGET_CLASSES } else { '0,1,2,3,4,5,6,7,8,9' }
$mainSeeds = if ($env:MAIN_SEEDS) { $env:MAIN_SEEDS } else { '20260904,20260905,20260906' }
$mainWorkers = if ($env:MAIN_WORKERS) { $env:MAIN_WORKERS } else { '2' }
Invoke-ProjectPython .\run_causaloneflip_main.py $Phase --device $device --target-classes $targetClasses --seeds $mainSeeds --workers $mainWorkers @resumeArgs