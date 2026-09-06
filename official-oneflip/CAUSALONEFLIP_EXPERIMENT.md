# CausalONEFLIP experiment protocol

## Current cross-platform workflow

The current causal method is `causal-margin-v2`: trigger injection is performed
in raw pixel space, then normalized for the model. Its loss requires the clean
model to reject the target class and the one-bit-flipped model to accept it,
while enforcing a counterfactual logit gain through the changed weight.

On Ubuntu, run from this directory with a CUDA 12.4-compatible `uv` setup:

```bash
chmod +x run_causaloneflip.sh run_fresh_oneflip.sh
./run_causaloneflip.sh tune-coarse
./run_causaloneflip.sh tune-fine
./run_causaloneflip.sh select-params
```

Use `DEVICE=cpu` or another CUDA device only when necessary. Interrupted
coarse/fine calculations can be resumed with `--resume`. On Windows, the
equivalent commands are `./run_causaloneflip.ps1 -Phase tune-coarse` and
`./run_causaloneflip.ps1 -Phase tune-fine`. Both entry points use the standard
local `.venv`; they never create host-named environments.

`causal_oneflip.py` implements an offline, full-precision, single-bit model
simulation. It reads a clean checkpoint and independently searches all final
classifier weights that satisfy the ONEFLIP exponent pattern. It never reads
the authors' cached potential weights, trigger dictionaries, or backdoored
checkpoints. It does not contain Rowhammer or physical-memory operations.

## Method

The `causal` method optimizes:

```
CE(flipped_logits, target)
+ beta * ReLU(clean_target_logit - max(clean_non_target_logits) + margin)
+ lambda * L1(mask)
```

The `oneflip` baseline implements the authors' feature-neuron objective with
the same calibration data, trigger epochs, mask parameterization, random seed,
and test evaluation. `tuap` is a target universal perturbation baseline using
the same mask/pattern parameterization and optimization budget.

Every non-T-UAP trial reports clean BA, flipped BA, BAD, trigger-only ASR,
bit-only ASR, trigger-plus-bit ASR, marginal bit gain, bit strings, Hamming
distance, L1(mask), and global image SSIM. For T-UAP, the bit-only fields are
explicitly unavailable because that baseline changes no weight. `summary.csv`
reports mean, standard deviation, a deterministic bootstrap 95% confidence
interval, and qualifying-pair rate by method, beta, lambda, and target class.
`summary_overall.csv` provides the corresponding aggregate over all selected
target classes and seeds.

`results.csv` also retains the calibration clean/modified accuracy and BAD for
each candidate, while `manifest.json` records the exact command arguments,
CUDA/PyTorch versions, device name, candidate counts, and best qualifying pair.

## Run sequence

Run from `official-oneflip`:

```powershell
.\run_causaloneflip.ps1 -Phase tune
```

Rank the completed tuning trial by the prespecified rule (qualification rate,
then combined ASR, bit marginal gain, and lower L1) and write the selected
fixed parameters:

```powershell
.\run_causaloneflip.ps1 -Phase select-params
```

This writes `parameter_ranking.csv` and `selected_params.json`. The remaining
phases read the selected mask weight automatically, so all methods use the
same perturbation regularization. The causal main experiment stops if no
tuning setting produced a qualifying causal pair; the two baselines may still
be run to document that negative result. Then run:

```powershell
.\run_causaloneflip.ps1 -Phase causal-main
.\run_causaloneflip.ps1 -Phase oneflip-baseline
.\run_causaloneflip.ps1 -Phase tuap-baseline
.\run_causaloneflip.ps1 -Phase report-cifar10
```

Each phase refuses an existing output directory. Use a new directory name for
a separate experiment. If a CausalONEFLIP calculation is interrupted, resume
only that incomplete phase with the same command plus `-Resume`; each finished
trial is retained in `trial_triggers/` and `progress.json`. A normal rerun
without `-Resume` still refuses an existing output directory.

`report-cifar10` produces `runs/causaloneflip-cifar10-report/` with
`comparison_summary.csv` (mean, standard deviation, 95% bootstrap interval,
and qualifying-pair rate for all three methods) and `qualified_pairs.csv`
(every qualifying CausalONEFLIP pair). This is the source for the paper main
table; do not replace it with a hand-picked maximum-ASR example.

## CIFAR-100 extension

Only after a CIFAR-10 causal trial qualifies, train a clean CIFAR-100 model:

```powershell
.\run_causaloneflip.ps1 -Phase train-cifar100
```

Then run the gated extension. It uses checkpoint
`saved_model/resnet_CIFAR100/clean_model_1.pth`, targets `0,20,40,60,80`, the
fixed CIFAR-10 parameters, and the same three main seeds:

```powershell
.\run_causaloneflip.ps1 -Phase cifar100-causal
```

## Qualification rule

A CausalONEFLIP pair qualifies only if all conditions hold:

- Trigger-only ASR <= 0.15
- Bit-only ASR <= 0.15
- Trigger-plus-bit ASR >= 0.90
- BAD <= 0.001
- Hamming distance = 1

If no trial qualifies, the correct result is a negative finding: this setting
does not demonstrate causal complementarity between the trigger and bit flip.
