"""Offline-only, fresh ONEFLIP reproduction for CIFAR-10 / ResNet-18.

The only model input is a clean checkpoint.  This program deliberately does
not read the author's cached candidate/trigger files or their backdoored
checkpoints, and it never attempts Rowhammer or physical memory operations.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image


NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZE_STD = (0.247, 0.243, 0.261)


def float32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", float(value)))[0]


def bits_to_float32(bits: int) -> float:
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def bit_string(value: float) -> str:
    return f"{float32_bits(value):032b}"


def hamming_distance(left: float, right: float) -> int:
    return bin(float32_bits(left) ^ float32_bits(right)).count("1")


def eligible_oneflip(value: float) -> float | None:
    """Return the paper's one-bit exponent-flip value, otherwise ``None``.

    Eligibility is strict: the float is positive; the exponent MSB is zero;
    and precisely one of the remaining seven exponent bits is zero.  The
    returned value sets that sole zero to one.
    """
    bits = float32_bits(value)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    lower_seven = exponent & 0x7F
    zero_positions = (~lower_seven) & 0x7F
    if sign != 0 or value <= 0 or exponent & 0x80 or bin(zero_positions).count("1") != 1:
        return None
    changed = (bits & ~(0xFF << 23)) | ((exponent | zero_positions) << 23)
    result = bits_to_float32(changed)
    if result <= 1 or hamming_distance(value, result) != 1:
        return None
    return result


class Cifar10ResNet18(nn.Module):
    """Architecture used by the authors' supplied CIFAR-10 checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None, num_classes=512)
        self.fc = nn.Linear(512, 10)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(images)
        return features, self.fc(features)


@dataclass
class Candidate:
    feature: int
    target_class: int
    original_weight: float
    flipped_weight: float
    original_bits: str
    flipped_bits: str
    calibration_clean_accuracy: float
    calibration_bad: float


def make_dataset(data_root: Path) -> torchvision.datasets.CIFAR10:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    return torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=transform)


def batches(dataset: torch.utils.data.Dataset, batch_size: int, workers: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    yield from loader


def first_samples(dataset: torch.utils.data.Dataset, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count > len(dataset):
        raise ValueError(f"Requested {count} calibration samples, but dataset only has {len(dataset)}.")
    images, labels = zip(*(dataset[index] for index in range(count)))
    return torch.stack(list(images)), torch.tensor(labels, dtype=torch.long)


def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(images)[1].argmax(dim=1) == labels).float().mean().item()


def clean_accuracy_full(model: nn.Module, dataset: torch.utils.data.Dataset, device: torch.device,
                        batch_size: int, workers: int) -> float:
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in batches(dataset, batch_size, workers):
            prediction = model(images.to(device))[1].argmax(dim=1).cpu()
            correct += int((prediction == labels).sum())
            total += len(labels)
    return correct / total


def target_rate_full(model: nn.Module, dataset: torch.utils.data.Dataset, mask: torch.Tensor,
                     pattern: torch.Tensor, target_class: int, device: torch.device,
                     batch_size: int, workers: int) -> float:
    matches = total = 0
    model.eval()
    mask, pattern = mask.to(device), pattern.to(device)
    with torch.no_grad():
        for images, _ in batches(dataset, batch_size, workers):
            images = images.to(device)
            triggered = (1 - mask.unsqueeze(0)) * images + mask.unsqueeze(0) * pattern
            prediction = model(triggered)[1].argmax(dim=1)
            matches += int((prediction == target_class).sum())
            total += len(images)
    return matches / total


def search_candidates(model: Cifar10ResNet18, calibration_images: torch.Tensor,
                      calibration_labels: torch.Tensor, target_class: int,
                      maximum_bad: float) -> tuple[float, list[Candidate]]:
    original_weights = model.fc.weight.detach().clone()
    baseline = accuracy(model, calibration_images, calibration_labels)
    candidates: list[Candidate] = []
    for feature in range(original_weights.shape[1]):
        original = float(original_weights[target_class, feature])
        flipped = eligible_oneflip(original)
        if flipped is None:
            continue
        model.fc.weight.data.copy_(original_weights)
        model.fc.weight.data[target_class, feature] = flipped
        candidate_accuracy = accuracy(model, calibration_images, calibration_labels)
        bad = baseline - candidate_accuracy
        if bad <= maximum_bad:
            candidates.append(Candidate(
                feature=feature,
                target_class=target_class,
                original_weight=original,
                flipped_weight=flipped,
                original_bits=bit_string(original),
                flipped_bits=bit_string(flipped),
                calibration_clean_accuracy=candidate_accuracy,
                calibration_bad=bad,
            ))
    model.fc.weight.data.copy_(original_weights)
    return baseline, candidates


def optimize_trigger(model: Cifar10ResNet18, calibration_images: torch.Tensor, feature: int,
                     epochs: int, mask_weight: float, optimization_batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise one pattern/mask pair for a feature neuron, with model frozen."""
    pattern = torch.rand((3, 32, 32), device=calibration_images.device, requires_grad=True)
    mask = torch.rand((32, 32), device=calibration_images.device, requires_grad=True)
    optimizer = torch.optim.Adam((pattern, mask), lr=0.01)
    model.eval()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total_samples = len(calibration_images)
        # The scaled gradients match a full-batch mean, but fit a 6 GB laptop GPU.
        for start in range(0, total_samples, optimization_batch_size):
            images = calibration_images[start:start + optimization_batch_size]
            target = torch.full((len(images),), feature, dtype=torch.long, device=images.device)
            triggered = (1 - mask.unsqueeze(0)) * images + mask.unsqueeze(0) * pattern
            features, _ = model(triggered)
            activation_loss = nn.functional.cross_entropy(features, target)
            (activation_loss * (len(images) / total_samples)).backward()
        (mask_weight * mask.abs().sum()).backward()
        optimizer.step()
        with torch.no_grad():
            pattern.clamp_(0, 1)
            mask.clamp_(0, 1)
    return mask.detach().cpu(), pattern.detach().cpu()


def write_trigger_preview(mask: torch.Tensor, pattern: torch.Tensor, destination: Path) -> None:
    """Save a simple side-by-side mask/pattern preview without extra dependencies."""
    mask_image = (mask.clamp(0, 1) * 255).to(torch.uint8).numpy()
    pattern_image = (pattern.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    left = Image.fromarray(mask_image, mode="L").convert("RGB")
    right = Image.fromarray(pattern_image, mode="RGB")
    preview = Image.new("RGB", (64, 32))
    preview.paste(left, (0, 0))
    preview.paste(right, (32, 0))
    preview.save(destination)


def changed_state_entries(clean_state: dict[str, torch.Tensor], modified_state: dict[str, torch.Tensor]) -> list[str]:
    changed = []
    for name, clean_value in clean_state.items():
        if not torch.equal(clean_value, modified_state[name]):
            changed.append(name)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh offline-only ONEFLIP reproduction")
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Must not already exist; prevents any accidental cache reuse.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-class", type=int, default=6)
    parser.add_argument("--calibration-size", type=int, default=1024)
    parser.add_argument("--max-bad", type=float, default=0.001)
    parser.add_argument("--trigger-epochs", type=int, default=500)
    parser.add_argument("--optimization-batch-size", type=int, default=128,
                        help="Trigger-gradient chunk size; all calibration samples still contribute every epoch.")
    parser.add_argument("--mask-weight", type=float, default=0.001)
    parser.add_argument("--attack-threshold", type=float, default=1.0)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists; refusing cache reuse: {args.output_dir}")
    if not args.clean_checkpoint.is_file():
        raise SystemExit(f"Clean checkpoint not found: {args.clean_checkpoint}")
    if not 0 <= args.target_class < 10:
        raise SystemExit("--target-class must be between 0 and 9.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Confirm the CUDA PyTorch build before running this experiment.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset = make_dataset(args.data_root)
    calibration_images, calibration_labels = first_samples(dataset, args.calibration_size)
    calibration_images, calibration_labels = calibration_images.to(device), calibration_labels.to(device)

    args.output_dir.mkdir(parents=True)
    shutil.copy2(args.clean_checkpoint, args.output_dir / "clean_model.pth")
    model = Cifar10ResNet18().to(device)
    clean_state = torch.load(args.clean_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(clean_state)
    clean_state_cpu = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    calibration_baseline, candidates = search_candidates(
        model, calibration_images, calibration_labels, args.target_class, args.max_bad
    )
    (args.output_dir / "candidates.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2), encoding="utf-8"
    )
    if not candidates:
        raise SystemExit("No eligible weight met the clean-accuracy threshold; inspect candidates.json and parameters.")

    features = sorted({candidate.feature for candidate in candidates})
    triggers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, feature in enumerate(features, start=1):
        print(f"[{index}/{len(features)}] Optimising trigger for feature {feature}", flush=True)
        mask, pattern = optimize_trigger(
            model, calibration_images, feature, args.trigger_epochs, args.mask_weight,
            args.optimization_batch_size
        )
        triggers[feature] = (mask, pattern)
        write_trigger_preview(mask, pattern, args.output_dir / f"trigger_feature_{feature}.png")
    torch.save(triggers, args.output_dir / "triggers.pt")

    clean_full_accuracy = clean_accuracy_full(model, dataset, device, args.test_batch_size, args.workers)
    original_weights = model.fc.weight.detach().clone()
    results: list[dict[str, object]] = []
    best: tuple[tuple[float, float, float], Candidate, dict[str, object]] | None = None
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] Evaluating feature {candidate.feature}, class {candidate.target_class}", flush=True)
        mask, pattern = triggers[candidate.feature]
        trigger_only_rate = target_rate_full(model, dataset, mask, pattern, candidate.target_class,
                                             device, args.test_batch_size, args.workers)
        model.fc.weight.data.copy_(original_weights)
        model.fc.weight.data[candidate.target_class, candidate.feature] = candidate.flipped_weight
        modified_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        changed_entries = changed_state_entries(clean_state_cpu, modified_state)
        if changed_entries != ["fc.weight"]:
            raise RuntimeError(f"Unexpected changed model tensors: {changed_entries}")
        backdoor_clean_accuracy = clean_accuracy_full(model, dataset, device, args.test_batch_size, args.workers)
        attack_success_rate = target_rate_full(model, dataset, mask, pattern, candidate.target_class,
                                               device, args.test_batch_size, args.workers)
        result: dict[str, object] = {
            **asdict(candidate),
            "bit_hamming_distance": hamming_distance(candidate.original_weight, candidate.flipped_weight),
            "clean_model_accuracy": clean_full_accuracy,
            "trigger_only_target_rate": trigger_only_rate,
            "backdoor_clean_accuracy": backdoor_clean_accuracy,
            "bad": clean_full_accuracy - backdoor_clean_accuracy,
            "attack_success_rate": attack_success_rate,
            "mask_l1": float(mask.abs().sum()),
        }
        results.append(result)
        if attack_success_rate >= args.attack_threshold:
            # Higher ASR, then lower BAD, then smaller mask is preferred.
            rank = (attack_success_rate, -float(result["bad"]), -float(result["mask_l1"]))
            if best is None or rank > best[0]:
                best = (rank, candidate, result)
        model.fc.weight.data.copy_(original_weights)

    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    manifest: dict[str, object] = {
        "offline_only": True,
        "author_artifacts_read": [str(args.clean_checkpoint)],
        "author_artifacts_not_read": [
            "*_potential_weights.npy", "*_neuron_trigger_pair.pkl", "backdoored_models/**/*.pth"
        ],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "calibration_clean_accuracy": calibration_baseline,
        "clean_model_accuracy": clean_full_accuracy,
        "candidate_count": len(candidates),
        "trigger_count": len(triggers),
        "qualified_model_count": sum(float(result["attack_success_rate"]) >= args.attack_threshold for result in results),
    }
    if best is not None:
        _, candidate, result = best
        model.fc.weight.data.copy_(original_weights)
        model.fc.weight.data[candidate.target_class, candidate.feature] = candidate.flipped_weight
        torch.save(model.state_dict(), args.output_dir / "best_backdoored_model.pth")
        manifest["best_model"] = result
    else:
        manifest["best_model"] = None
        manifest["note"] = "No candidate reached attack_threshold; no backdoored checkpoint was saved."
    model.fc.weight.data.copy_(original_weights)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
