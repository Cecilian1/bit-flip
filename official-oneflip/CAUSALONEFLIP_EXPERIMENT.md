 # CausalONEFLIP experiment protocol

## Current cross-platform workflow

The current causal method is `causal-margin-v2`: trigger injection is performed
in raw pixel space, then normalized for the model. Its loss requires the clean
model to reject the target class and the one-bit-flipped model to accept it,
while enforcing a counterfactual logit gain through the changed weight.

On Ubuntu 22.04 with an RTX 5090, use Python 3.12 and the PyTorch 2.8 CUDA
12.8 wheel specified by the repository lock file. CUDA 12.8 is the first
toolkit release that can emit native Blackwell code; do not reuse the older
CUDA 12.4 environment from the Windows coarse screen. The NVIDIA driver must
be at least 570.26 for CUDA 12.8 GA.

```bash
git clone git@github.com:Cecilian1/bit-flip.git
cd bit-flip
uv sync --frozen --python 3.12
uv run --frozen python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Then run from `official-oneflip`:

```bash
chmod +x run_causaloneflip.sh run_fresh_oneflip.sh
./run_causaloneflip.sh tune-coarse
./run_causaloneflip.sh tune-fine
./run_causaloneflip.sh select-params
./run_causaloneflip.sh main-all
```

`main-all` uses the validation-qualified fixed setting from
`runs/causaloneflip-tune-class6/selected_params.json`, runs causal, ONEFLIP,
and T-UAP for CIFAR-10 targets 0--9 and three seeds, then writes the aggregate
report. It refuses to run the causal arm if no fixed setting qualified.

When a virtual environment is already activated, both entry points use its
`python` directly. Set `PYTHON_EXECUTABLE` to an absolute interpreter path to
make this explicit. Otherwise they use `uv run --frozen`, which creates only
the standard project `.venv`; no script derives a virtual-environment name
from a computer or user name.

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

## Main experiment

After the fine stage has written a validation-qualified fixed setting, run the
complete CIFAR-10 experiment from `official-oneflip`:

```bash
./run_causaloneflip.sh main-all
```

This runs the causal method, ONEFLIP, and T-UAP for targets `0,1,...,9` and
seeds `20260904,20260905,20260906`, then creates:

```text
runs/causaloneflip-cifar10-causal-main/
runs/causaloneflip-cifar10-oneflip-baseline/
runs/causaloneflip-cifar10-tuap-baseline/
runs/causaloneflip-cifar10-report/comparison_summary.csv
runs/causaloneflip-cifar10-report/qualified_pairs.csv
```

Run a single phase with `causal-main`, `oneflip-baseline`, `tuap-baseline`, or
`report-cifar10`. `TARGET_CLASSES`, `MAIN_SEEDS`, `MAIN_WORKERS`, and `DEVICE`
can override the defaults; use `--resume` only after an interrupted run. The
causal phase stops when fine selection has no validation-qualified setting;
the baseline phases remain runnable for a negative-result comparison.

On Windows, use the same phase names:

```powershell
.\run_causaloneflip.ps1 -Phase main-all
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
