# Fresh offline ONEFLIP reproduction

`reproduce_fresh_oneflip.py` carries out the paper's model-level workflow for
CIFAR-10 / ResNet-18 without reading any supplied candidate, trigger, or
backdoored-model artifact.  It only reads `clean_model_1.pth`, then writes
every newly generated artifact to a new output directory.

It does not perform Rowhammer, DRAM profiling, memory placement, or any
physical fault-injection activity.

## Before running

In the `official-oneflip` directory, make sure CUDA PyTorch is usable:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

If uv reports a cache permission error, pick a new project-local cache name,
for example `--cache-dir .\.uv-cache-fresh` on both `uv sync` and `uv run`.

## Full target-class-6 reproduction

Run this from `official-oneflip`. The output directory must not already exist.

```powershell
uv run --cache-dir .\.uv-cache-fresh python .\reproduce_fresh_oneflip.py --clean-checkpoint .\saved_model\resnet_CIFAR10\clean_model_1.pth --data-root ..\dataset\CIFAR10 --output-dir .\runs\fresh-cifar10-class6 --device cuda:0 --target-class 6 --calibration-size 1024 --max-bad 0.001 --trigger-epochs 500 --optimization-batch-size 128 --mask-weight 0.001 --attack-threshold 1.0 --test-batch-size 512 --workers 2 --seed 20260904
```

The run searches all 512 final-layer weights connecting to target class 6. It
may take a long time because it independently optimizes one trigger for every
eligible feature and evaluates each candidate on all 10,000 test images. The
128-image trigger chunk size reduces GPU memory use but still accumulates the
gradient over all 1,024 calibration images at every optimization step.

## Output

- `clean_model.pth`: copy of the sole input model.
- `candidates.json`: fresh eligible-weight search results.
- `triggers.pt` and `trigger_feature_*.png`: freshly optimized triggers.
- `results.csv`: full-test benign accuracy, BAD, ASR, and ablations for every candidate.
- `best_backdoored_model.pth`: only the best model meeting 100% ASR.
- `manifest.json`: parameters, environment, artifact provenance, and selection result.

The four reported evaluations are: clean model with clean inputs, clean model
with trigger, one-bit model with clean inputs, and one-bit model with trigger.
