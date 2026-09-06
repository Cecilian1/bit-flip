"""Fresh, offline-only reproduction of ONEFLIP for CIFAR-10 / ResNet-18.

This program never accesses DRAM or performs Rowhammer.  It makes a copy of a
local clean checkpoint, simulates one IEEE-754 float32 exponent-bit change in
that copy, and writes all generated candidates, triggers, and checkpoints to a
new run directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def f32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", float(value)))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def flip_rightmost_zero_exponent(value: float) -> float:
    """Perform exactly one 0 -> 1 change in the float32 exponent field."""
    bits = f32_bits(value)
    exponent = (bits >> 23) & 0xFF
    zero = (~exponent) & (exponent + 1)
    if zero == 0:
        return float(value)
    return bits_f32((bits & ~(0xFF << 23)) | ((exponent ^ zero) << 23))


class CifarResNetClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None, num_classes=512)
        self.fc = nn.Linear(512, 10)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(images)
        return features, self.fc(features)


def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(images)[1].argmax(dim=1) == labels).float().mean().item()


def make_loader(data_root: Path, batch_size: int, workers: int) -> torch.utils.data.DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=False, transform=transform
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def search_candidates(model: CifarResNetClassifier, images: torch.Tensor, labels: torch.Tensor,
                      max_drop: float, target_class: int | None) -> list[dict[str, float | int]]:
    """Search the final classifier as in the paper's offline candidate stage."""
    original = model.fc.weight.detach().clone()
    baseline = accuracy(model, images, labels)
    found: list[dict[str, float | int]] = []
    for feature in range(original.shape[1]):
        for cls in range(original.shape[0]):
            if target_class is not None and cls != target_class:
                continue
            before = float(original[cls, feature])
            after = flip_rightmost_zero_exponent(before)
            if before <= 0 or after < 1 or f32_bits(before) == f32_bits(after):
                continue
            model.fc.weight.data.copy_(original)
            model.fc.weight.data[cls, feature] = after
            candidate_acc = accuracy(model, images, labels)
            if baseline - candidate_acc <= max_drop:
                found.append({"feature": feature, "class": cls, "before": before,
                              "after": after, "calibration_accuracy": candidate_acc})
    model.fc.weight.data.copy_(original)
    return found


def optimize_trigger(model: CifarResNetClassifier, images: torch.Tensor, feature: int,
                     steps: int, mask_penalty: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize one universal mask/pattern pair to activate a feature neuron."""
    pattern = torch.rand((3, 32, 32), device=images.device, requires_grad=True)
    mask = torch.rand((32, 32), device=images.device, requires_grad=True)
    optimizer = torch.optim.Adam((pattern, mask), lr=0.01)
    target = torch.full((len(images),), feature, dtype=torch.long, device=images.device)
    model.eval()
    for _ in range(steps):
        optimizer.zero_grad()
        triggered = (1 - mask.unsqueeze(0)) * images + mask.unsqueeze(0) * pattern
        features, _ = model(triggered)
        loss = nn.functional.cross_entropy(features, target) + mask_penalty * mask.abs().sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pattern.clamp_(0, 1)
            mask.clamp_(0, 1)
    return mask.detach().cpu(), pattern.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh offline ONEFLIP reproduction (no Rowhammer)")
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/CIFAR10"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Must not exist; prevents accidental reuse of supplied artifacts.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-clean-drop", type=float, default=0.001)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--max-neurons", type=int, default=1,
                        help="Small first-run budget; use 0 to optimize every selected feature.")
    parser.add_argument("--trigger-steps", type=int, default=500)
    parser.add_argument("--mask-penalty", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to reuse existing output directory: {args.output_dir}")
    if not args.clean_checkpoint.is_file():
        raise SystemExit(f"Clean checkpoint not found: {args.clean_checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable. Use --device cpu or install CUDA PyTorch.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True)
    shutil.copy2(args.clean_checkpoint, args.output_dir / "clean_model.pth")

    model = CifarResNetClassifier().to(device)
    state = torch.load(args.clean_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    images, labels = next(iter(make_loader(args.data_root, args.batch_size, args.workers)))
    images, labels = images.to(device), labels.to(device)
    baseline = accuracy(model, images, labels)
    candidates = search_candidates(model, images, labels, args.max_clean_drop, args.target_class)
    (args.output_dir / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    if not candidates:
        raise SystemExit("No candidate passed the calibration clean-accuracy threshold.")

    # A feature can connect to several target classes.  One trigger is optimized per feature.
    features = list(dict.fromkeys(int(item["feature"]) for item in candidates))
    if args.max_neurons:
        features = features[:args.max_neurons]
    triggers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for feature in features:
        triggers[feature] = optimize_trigger(model, images, feature, args.trigger_steps, args.mask_penalty)
    torch.save(triggers, args.output_dir / "triggers.pt")

    original = model.fc.weight.detach().clone()
    results = []
    for item in candidates:
        feature, cls = int(item["feature"]), int(item["class"])
        if feature not in triggers:
            continue
        model.fc.weight.data.copy_(original)
        model.fc.weight.data[cls, feature] = float(item["after"])
        mask, pattern = (value.to(device) for value in triggers[feature])
        with torch.no_grad():
            triggered = (1 - mask.unsqueeze(0)) * images + mask.unsqueeze(0) * pattern
            asr = (model(triggered)[1].argmax(dim=1) == cls).float().mean().item()
        clean_acc = accuracy(model, images, labels)
        name = f"backdoor_feature_{feature}_class_{cls}.pth"
        torch.save(model.state_dict(), args.output_dir / name)
        results.append({**item, "clean_accuracy_after": clean_acc, "calibration_asr": asr,
                        "checkpoint": name, "bit_hamming_distance": 1})
    model.fc.weight.data.copy_(original)
    manifest = {"offline_only": True, "baseline_calibration_accuracy": baseline,
                "device": str(device), "candidates_found": len(candidates),
                "features_optimized": features, "results": results}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
