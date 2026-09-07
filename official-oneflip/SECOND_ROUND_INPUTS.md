# Second-round input bundle

This repository deliberately contains the minimum inputs needed to run the
Ubuntu second-round CausalONEFLIP experiment:

- `../dataset/CIFAR10/cifar-10-batches-py/`: unpacked CIFAR-10 Python files.
  The original `cifar-10-python.tar.gz` is not included because it is a
  redundant upstream archive.
- `saved_model/resnet_CIFAR10/clean_model_1.pth`: clean model checkpoint.
- `saved_model/resnet_CIFAR10/clean_model_1_neuron_trigger_pair.pkl` and
  `clean_model_1_potential_weights.npy`: official candidate metadata.
- `runs/causaloneflip-tune-class6-coarse-{0.5,1,2}/results.csv` plus their
  manifests, and `runs/causaloneflip-tune-class6-coarse/selected_params.json`:
  the complete first-round metric record and fixed shortlist.

The following are intentionally excluded: virtual environments, CUDA wheels,
duplicate copies of the clean model emitted by each trial, generated triggers,
backdoored checkpoints, and progress files. None is required by
`run_causaloneflip.sh tune-fine`; the second-round script re-optimizes its
triggers from the clean model under the fixed shortlist.

Before running on Ubuntu, verify the input bundle and GPU:

```bash
sha256sum dataset/CIFAR10/cifar-10-batches-py/* \
  official-oneflip/saved_model/resnet_CIFAR10/clean_model_1.pth
cd official-oneflip
uv run --frozen python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
./run_causaloneflip.sh tune-fine
```