"""CausalONEFLIP: causality-constrained, offline one-bit backdoor research.

This program performs model-level IEEE-754 simulations only.  It has no
Rowhammer, DRAM profiling, physical-address, or memory-fault functionality.
It never loads the authors' cached candidates, triggers, or backdoored models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from model_template.preactres import PreActResNet18


@dataclass(frozen=True)
class DatasetSpec:
    classes: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


SPECS = {
    "CIFAR10": DatasetSpec(10, (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    "CIFAR100": DatasetSpec(100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def f32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", float(value)))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def hamming(left: float, right: float) -> int:
    return bin(f32_bits(left) ^ f32_bits(right)).count("1")


def eligible_flip(value: float) -> Optional[float]:
    """Implement the paper's positive float32 exponent eligibility rule."""
    bits = f32_bits(value)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    lower = exponent & 0x7F
    zero_bits = (~lower) & 0x7F
    if sign or value <= 0 or exponent & 0x80 or bin(zero_bits).count("1") != 1:
        return None
    flipped = bits_f32((bits & ~(0xFF << 23)) | ((exponent | zero_bits) << 23))
    return flipped if flipped > 1 and hamming(value, flipped) == 1 else None


def bit_string(value: float) -> str:
    return "{0:032b}".format(f32_bits(value))


def parse_numbers(value: str, cast):
    try:
        return [cast(piece.strip()) for piece in value.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid comma-separated numeric list: {0}".format(value)) from exc


class FeatureClassifier(nn.Module):
    def __init__(self, dataset: str) -> None:
        super().__init__()
        spec = SPECS[dataset]
        if dataset == "CIFAR10":
            self.model = torchvision.models.resnet18(weights=None, num_classes=512)
        elif dataset == "CIFAR100":
            self.model = PreActResNet18(num_classes=512)
        else:
            raise ValueError("Unsupported dataset: {0}".format(dataset))
        self.fc = nn.Linear(512, spec.classes)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.model(images)
        return features, self.fc(features)


@dataclass(frozen=True)
class Candidate:
    feature: int
    target_class: int
    original_weight: float
    flipped_weight: float
    calibration_accuracy: float
    calibration_bad: float


@dataclass(frozen=True)
class Trial:
    method: str
    feature: int
    target_class: int
    beta: float
    mask_weight: float
    seed: int
    original_weight: float
    flipped_weight: float


def trial_key(trial: Trial) -> str:
    return "{0}_target_{1}_feature_{2}_beta_{3:g}_lambda_{4:g}_seed_{5}".format(
        trial.method, trial.target_class, trial.feature, trial.beta, trial.mask_weight, trial.seed
    )


def dataset_for(name: str, root: Path) -> torch.utils.data.Dataset:
    spec = SPECS[name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)])
    if name == "CIFAR10":
        return torchvision.datasets.CIFAR10(root=str(root), train=False, download=False, transform=transform)
    return torchvision.datasets.CIFAR100(root=str(root), train=False, download=False, transform=transform)


def data_batches(dataset: torch.utils.data.Dataset, batch_size: int, workers: int):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def calibration_subset(dataset: torch.utils.data.Dataset, count: int, seed: int = 20260904) -> Tuple[torch.Tensor, torch.Tensor]:
    if count > len(dataset):
        raise ValueError("Calibration size exceeds dataset length.")
    indices = random.Random(seed).sample(range(len(dataset)), count)
    samples = [dataset[index] for index in indices]
    return torch.stack([item[0] for item in samples]), torch.tensor([item[1] for item in samples])


def logits(model: FeatureClassifier, images: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.linear(model.model(images), weight, bias)


def causal_margin_loss(clean_logits: torch.Tensor, flipped_logits: torch.Tensor, target: int,
                       clean_margin: float, attack_margin: float) -> torch.Tensor:
    """Differentiable loss for a clean-model reject / flipped-model accept interval."""
    clean_target = clean_logits[:, target]
    other = clean_logits.clone()
    other[:, target] = -math.inf
    other_max = other.max(dim=1).values
    flipped_target = flipped_logits[:, target]
    clean_loss = F.softplus(clean_target - other_max + clean_margin).mean()
    attack_loss = F.softplus(other_max - flipped_target + attack_margin).mean()
    gain_loss = F.relu(clean_margin + attack_margin - (flipped_target - clean_target)).mean()
    return attack_loss + clean_loss + gain_loss


def classifier_accuracy(model: FeatureClassifier, dataset: torch.utils.data.Dataset, weight: torch.Tensor,
                        bias: torch.Tensor, device: torch.device, batch_size: int, workers: int) -> float:
    matched = total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_batches(dataset, batch_size, workers):
            prediction = logits(model, images.to(device), weight, bias).argmax(dim=1).cpu()
            matched += int((prediction == labels).sum())
            total += len(labels)
    return matched / total


def restore_pixels(images: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    mean = torch.tensor(spec.mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def normalize_pixels(images: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    mean = torch.tensor(spec.mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def apply_trigger(images: torch.Tensor, mask: torch.Tensor, pattern: torch.Tensor,
                  spec: DatasetSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inject in the [0, 1] pixel domain, then restore model normalization."""
    raw = restore_pixels(images, spec)
    changed_raw = (1 - mask.unsqueeze(0)) * raw + mask.unsqueeze(0) * pattern
    return normalize_pixels(changed_raw.clamp(0, 1), spec), changed_raw.clamp(0, 1)


def global_ssim(original: torch.Tensor, changed: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    """Channel-wise global SSIM, averaged over images; dependency-free and reproducible."""
    original = restore_pixels(original, spec)
    changed = restore_pixels(changed, spec)
    mean_x = original.mean(dim=(2, 3))
    mean_y = changed.mean(dim=(2, 3))
    var_x = original.var(dim=(2, 3), unbiased=False)
    var_y = changed.var(dim=(2, 3), unbiased=False)
    cov = ((original - mean_x[:, :, None, None]) * (changed - mean_y[:, :, None, None])).mean(dim=(2, 3))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * cov + c2)) / ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2))
    return score.mean()


def target_metrics(model: FeatureClassifier, dataset: torch.utils.data.Dataset, weight: torch.Tensor,
                   bias: torch.Tensor, target: int, device: torch.device, batch_size: int, workers: int,
                   spec: DatasetSpec, mask: Optional[torch.Tensor] = None,
                   pattern: Optional[torch.Tensor] = None) -> Tuple[float, Optional[float]]:
    matches = total = 0
    ssim_sum = 0.0
    has_trigger = mask is not None and pattern is not None
    if has_trigger:
        mask, pattern = mask.to(device), pattern.to(device)
    model.eval()
    with torch.no_grad():
        for images, _ in data_batches(dataset, batch_size, workers):
            images = images.to(device)
            changed = images
            if has_trigger:
                changed, _ = apply_trigger(images, mask, pattern, spec)
                ssim_sum += float(global_ssim(images, changed, spec)) * len(images)
            matches += int((logits(model, changed, weight, bias).argmax(dim=1) == target).sum())
            total += len(images)
    return matches / total, (ssim_sum / total if has_trigger else None)


def evaluate_calibration(model: FeatureClassifier, images: torch.Tensor, labels: torch.Tensor,
                         weight: torch.Tensor, bias: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((logits(model, images, weight, bias).argmax(dim=1) == labels).float().mean())


def causal_diagnostics(model: FeatureClassifier, images: torch.Tensor, labels: torch.Tensor,
                       clean_weight: torch.Tensor, clean_bias: torch.Tensor, trial: Trial, spec: DatasetSpec,
                       mask: torch.Tensor, pattern: torch.Tensor) -> Dict[str, float]:
    """Counterfactual margins on a held-out benign set for causal selection."""
    model.eval()
    with torch.no_grad():
        mask, pattern = mask.to(images.device), pattern.to(images.device)
        triggered, _ = apply_trigger(images, mask, pattern, spec)
        features = model.model(triggered)
        clean_logits = F.linear(features, clean_weight, clean_bias)
        flipped_weight = clean_weight.clone()
        flipped_weight[trial.target_class, trial.feature] = trial.flipped_weight
        flipped_logits = F.linear(features, flipped_weight, clean_bias)
        clean_plain = logits(model, images, clean_weight, clean_bias)
        flipped_plain = logits(model, images, flipped_weight, clean_bias)
        def margins(values: torch.Tensor) -> torch.Tensor:
            target = values[:, trial.target_class]
            other = values.clone()
            other[:, trial.target_class] = -math.inf
            return target - other.max(dim=1).values
        clean_margin = margins(clean_logits)
        flipped_margin = margins(flipped_logits)
        return {
            "validation_trigger_only_asr": float((clean_margin >= 0).float().mean()),
            "validation_combined_asr": float((flipped_margin >= 0).float().mean()),
            "validation_bit_only_asr": float((flipped_plain.argmax(dim=1) == trial.target_class).float().mean()),
            "validation_bad": float(
                (clean_plain.argmax(dim=1) == labels).float().mean()
                - (flipped_plain.argmax(dim=1) == labels).float().mean()
            ),
            "validation_clean_target_margin": float(clean_margin.mean()),
            "validation_flipped_target_margin": float(flipped_margin.mean()),
            "validation_counterfactual_gain": float((flipped_logits[:, trial.target_class] - clean_logits[:, trial.target_class]).mean()),
        }


def search_candidates(model: FeatureClassifier, calibration_images: torch.Tensor, calibration_labels: torch.Tensor,
                      target: int, max_bad: float) -> List[Candidate]:
    clean_weight, clean_bias = model.fc.weight.detach().clone(), model.fc.bias.detach().clone()
    baseline = evaluate_calibration(model, calibration_images, calibration_labels, clean_weight, clean_bias)
    candidates: List[Candidate] = []
    for feature in range(clean_weight.shape[1]):
        original = float(clean_weight[target, feature])
        flipped = eligible_flip(original)
        if flipped is None:
            continue
        candidate_weight = clean_weight.clone()
        candidate_weight[target, feature] = flipped
        acc = evaluate_calibration(model, calibration_images, calibration_labels, candidate_weight, clean_bias)
        bad = baseline - acc
        if bad <= max_bad:
            candidates.append(Candidate(feature, target, original, flipped, acc, bad))
    return candidates


def optimize_trigger(model: FeatureClassifier, images: torch.Tensor, clean_weight: torch.Tensor,
                     clean_bias: torch.Tensor, trial: Trial, epochs: int, optimization_batch_size: int,
                     clean_margin: float, attack_margin: float = 0.5,
                     validation_images: Optional[torch.Tensor] = None,
                     validation_interval: int = 25,
                     spec: Optional[DatasetSpec] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimise the requested method's trigger using all calibration images each epoch."""
    torch.manual_seed(trial.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial.seed)
    pattern = torch.rand((3, 32, 32), device=images.device, requires_grad=True)
    mask = torch.rand((32, 32), device=images.device, requires_grad=True)
    optimizer = torch.optim.Adam([pattern, mask], lr=0.01)
    flipped_weight = clean_weight.clone()
    if trial.method != "tuap":
        flipped_weight[trial.target_class, trial.feature] = trial.flipped_weight
    model.eval()
    best_score = -math.inf
    best_artifact: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total = len(images)
        for start in range(0, total, optimization_batch_size):
            batch = images[start:start + optimization_batch_size]
            target_labels = torch.full((len(batch),), trial.target_class, dtype=torch.long, device=images.device)
            triggered, _ = apply_trigger(batch, mask, pattern, spec)
            features = model.model(triggered)
            if trial.method == "oneflip":
                loss = F.cross_entropy(features, torch.full_like(target_labels, trial.feature))
            elif trial.method == "causal":
                flipped_logits = F.linear(features, flipped_weight, clean_bias)
                clean_logits = F.linear(features, clean_weight, clean_bias)
                # A feasible trigger must lose on the clean model and win after
                # the one-bit intervention.  The gain term keeps optimization
                # on the changed classifier connection rather than a generic UAP.
                clean_target = clean_logits[:, trial.target_class]
                clean_other = clean_logits.clone()
                clean_other[:, trial.target_class] = -math.inf
                other_max = clean_other.max(dim=1).values
                flipped_target = flipped_logits[:, trial.target_class]
                clean_loss = F.softplus(clean_target - other_max + clean_margin).mean()
                attack_loss = F.softplus(other_max - flipped_target + attack_margin).mean()
                gain_loss = F.relu(clean_margin + attack_margin - (flipped_target - clean_target)).mean()
                loss = attack_loss + trial.beta * clean_loss + gain_loss
            elif trial.method == "tuap":
                loss = F.cross_entropy(F.linear(features, clean_weight, clean_bias), target_labels)
            else:
                raise ValueError("Unknown method: {0}".format(trial.method))
            (loss * (len(batch) / total)).backward()
        (trial.mask_weight * mask.abs().sum()).backward()
        optimizer.step()
        with torch.no_grad():
            pattern.clamp_(0, 1)
            mask.clamp_(0, 1)
            if trial.method == "causal" and validation_images is not None and spec is not None and (epoch + 1) % validation_interval == 0:
                validation_triggered, _ = apply_trigger(validation_images, mask, pattern, spec)
                validation_features = model.model(validation_triggered)
                clean_scores = F.linear(validation_features, clean_weight, clean_bias)
                flipped_scores = F.linear(validation_features, flipped_weight, clean_bias)
                clean_rate = (clean_scores.argmax(dim=1) == trial.target_class).float().mean()
                combined_rate = (flipped_scores.argmax(dim=1) == trial.target_class).float().mean()
                score = float(combined_rate - clean_rate)
                if score > best_score:
                    best_score = score
                    best_artifact = (mask.detach().cpu().clone(), pattern.detach().cpu().clone())
    if best_artifact is not None:
        return best_artifact
    return mask.detach().cpu(), pattern.detach().cpu()


def trigger_preview(mask: torch.Tensor, pattern: torch.Tensor, destination: Path) -> None:
    left = Image.fromarray((mask.clamp(0, 1) * 255).to(torch.uint8).numpy(), mode="L").convert("RGB")
    right = Image.fromarray((pattern.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).numpy(), mode="RGB")
    image = Image.new("RGB", (64, 32))
    image.paste(left, (0, 0))
    image.paste(right, (32, 0))
    image.save(destination)


def bootstrap_interval(values: Sequence[float], seed: int = 20260904, repeats: int = 1000) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    generator = random.Random(seed)
    means = sorted(sum(generator.choice(values) for _ in values) / len(values) for _ in range(repeats))
    return means[int(0.025 * (repeats - 1))], means[int(0.975 * (repeats - 1))]


def write_summary(rows: List[Dict[str, object]], path: Path, include_target: bool) -> None:
    metric_names = ["trigger_only_asr", "bit_only_asr", "trigger_plus_bit_asr", "bit_marginal_gain", "bad", "mask_l1", "ssim"]
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key: Tuple[object, ...] = (str(row["method"]), float(row["beta"]), float(row["mask_weight"]))
        if include_target:
            key += (int(row["target_class"]),)
        groups.setdefault(key, []).append(row)
    summary: List[Dict[str, object]] = []
    for key, records in sorted(groups.items()):
        method, beta, mask_weight = key[:3]
        item: Dict[str, object] = {
            "method": method,
            "beta": beta,
            "mask_weight": mask_weight,
            "trials": len(records),
            "qualified": sum(bool(row["qualifies"]) for row in records),
            "qualification_rate": sum(bool(row["qualifies"]) for row in records) / len(records),
        }
        target = None
        if include_target:
            target = int(key[3])
            item["target_class"] = target
        for metric in metric_names:
            values = [float(row[metric]) for row in records]
            low, high = bootstrap_interval(values, seed=20260904 + (target if target is not None else 0))
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            item[metric + "_ci95_low"] = low
            item[metric + "_ci95_high"] = high
        summary.append(item)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalONEFLIP offline research framework")
    parser.add_argument("--method", choices=("oneflip", "causal", "tuap"), required=True)
    parser.add_argument("--dataset", choices=tuple(SPECS), default="CIFAR10")
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-classes", default="6")
    parser.add_argument("--betas", default="1")
    parser.add_argument("--mask-weights", default="0.001")
    parser.add_argument("--seeds", default="20260904")
    parser.add_argument("--calibration-size", type=int, default=1024)
    parser.add_argument("--optimization-size", type=int, default=768,
                        help="Prefix of the deterministic calibration split used to optimize triggers.")
    parser.add_argument("--max-calibration-bad", type=float, default=0.001)
    parser.add_argument("--trigger-epochs", type=int, default=500)
    parser.add_argument("--optimization-batch-size", type=int, default=128)
    parser.add_argument("--clean-margin", type=float, default=0.5)
    parser.add_argument("--attack-margin", type=float, default=0.5)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--loss-version", default="causal-margin-v2")
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-trigger-only-asr", type=float, default=0.15)
    parser.add_argument("--max-bit-only-asr", type=float, default=0.15)
    parser.add_argument("--min-combined-asr", type=float, default=0.90)
    parser.add_argument("--max-bad", type=float, default=0.001)
    parser.add_argument("--limit-candidates", type=int, default=0, help="0 uses every eligible candidate; positive values are a pilot-only cap.")
    parser.add_argument("--resume", action="store_true", help="Resume only this program's incomplete output directory.")
    args = parser.parse_args()

    targets = [int(value) for value in parse_numbers(args.target_classes, int)]
    betas = [float(value) for value in parse_numbers(args.betas, float)]
    mask_weights = [float(value) for value in parse_numbers(args.mask_weights, float)]
    seeds = [int(value) for value in parse_numbers(args.seeds, int)]
    spec = SPECS[args.dataset]
    if not targets or any(target < 0 or target >= spec.classes for target in targets):
        raise SystemExit("Target classes are outside the dataset class range.")
    if args.output_dir.exists() and not args.resume:
        raise SystemExit("Output directory already exists (use --resume only for an incomplete CausalONEFLIP run): {0}".format(args.output_dir))
    if args.resume and not args.output_dir.is_dir():
        raise SystemExit("Cannot resume: output directory does not exist: {0}".format(args.output_dir))
    if not args.clean_checkpoint.is_file():
        raise SystemExit("Clean checkpoint not found: {0}".format(args.clean_checkpoint))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")

    device = torch.device(args.device)
    dataset = dataset_for(args.dataset, args.data_root)
    all_calibration_images, all_calibration_labels = calibration_subset(dataset, args.calibration_size, seeds[0])
    if not 0 < args.optimization_size < args.calibration_size:
        raise SystemExit("optimization-size must be between 1 and calibration-size - 1.")
    if args.validation_interval <= 0:
        raise SystemExit("validation-interval must be positive.")
    calibration_images = all_calibration_images[:args.optimization_size].to(device)
    calibration_labels = all_calibration_labels[:args.optimization_size].to(device)
    validation_images = all_calibration_images[args.optimization_size:].to(device)
    validation_labels = all_calibration_labels[args.optimization_size:].to(device)
    model = FeatureClassifier(args.dataset).to(device)
    model.load_state_dict(torch.load(args.clean_checkpoint, map_location=device, weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    clean_weight, clean_bias = model.fc.weight.detach().clone(), model.fc.bias.detach().clone()
    calibration_clean_accuracy = evaluate_calibration(model, calibration_images, calibration_labels, clean_weight, clean_bias)
    clean_accuracy = classifier_accuracy(model, dataset, clean_weight, clean_bias, device, args.test_batch_size, args.workers)
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
        shutil.copy2(args.clean_checkpoint, args.output_dir / "clean_model.pth")

    candidates_by_target: Dict[int, List[Candidate]] = {}
    for target in targets:
        candidates = search_candidates(model, calibration_images, calibration_labels, target, args.max_calibration_bad)
        # The cap exists only for short pilot studies; default execution evaluates all candidates.
        if args.limit_candidates:
            candidates = sorted(candidates, key=lambda item: (-item.flipped_weight, item.calibration_bad))[:args.limit_candidates]
        candidates_by_target[target] = candidates
    trials: List[Trial] = []
    for target in targets:
        candidates = candidates_by_target[target]
        if args.method == "tuap":
            candidates = [Candidate(-1, target, 0.0, 0.0, 0.0, 0.0)]
        for candidate in candidates:
            for beta in betas:
                for mask_weight in mask_weights:
                    for seed in seeds:
                        trials.append(Trial(args.method, candidate.feature, target, beta, mask_weight, seed, candidate.original_weight, candidate.flipped_weight))
    if not trials:
        raise SystemExit("No trials were created. Check clean model, target classes, and candidate threshold.")
    expected_keys = {trial_key(trial) for trial in trials}

    progress_path = args.output_dir / "progress.json"
    trial_trigger_dir = args.output_dir / "trial_triggers"
    if args.resume:
        if (args.output_dir / "manifest.json").is_file():
            raise SystemExit("Cannot resume: this output directory already contains a finalized manifest.")
        if not progress_path.is_file() or not trial_trigger_dir.is_dir():
            raise SystemExit("Cannot resume: progress.json or trial_triggers is missing from {0}".format(args.output_dir))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        results = list(progress["results"])
        completed = set(progress["completed_trial_keys"])
        if set(progress.get("planned_trial_keys", [])) != expected_keys:
            raise SystemExit("Cannot resume: trial configuration differs from the incomplete run.")
        if {str(row.get("trial_key", "")) for row in results} != completed:
            raise SystemExit("Cannot resume: progress results and completed trial keys disagree.")
    else:
        results = []
        completed = set()
        trial_trigger_dir.mkdir()
    (args.output_dir / "candidates.json").write_text(
        json.dumps({str(target): [asdict(candidate) for candidate in candidates]
                    for target, candidates in candidates_by_target.items()}, indent=2),
        encoding="utf-8",
    )
    candidate_lookup = {
        (candidate.target_class, candidate.feature): candidate
        for candidates in candidates_by_target.values() for candidate in candidates
    }
    for index, trial in enumerate(trials, start=1):
        key = trial_key(trial)
        if key in completed:
            print("[{0}/{1}] Reusing completed trial {2}".format(index, len(trials), key), flush=True)
            continue
        print("[{0}/{1}] {2}".format(index, len(trials), key), flush=True)
        mask, pattern = optimize_trigger(
            model, calibration_images, clean_weight, clean_bias, trial,
            args.trigger_epochs, args.optimization_batch_size, args.clean_margin,
            args.attack_margin, validation_images, args.validation_interval, spec,
        )
        trigger_preview(mask, pattern, args.output_dir / (key + ".png"))
        if trial.method == "tuap":
            trigger_only, ssim = target_metrics(model, dataset, clean_weight, clean_bias, trial.target_class, device, args.test_batch_size, args.workers, spec, mask, pattern)
            # T-UAP has no model-side bit modification; report the two
            # bit-specific ablations as unavailable rather than presenting the
            # clean model's natural target frequency as a bit-only result.
            bit_only = float("nan")
            combined, _ = trigger_only, ssim
            backdoor_accuracy = clean_accuracy
            bit_distance: Optional[int] = None
        else:
            flipped_weight = clean_weight.clone()
            flipped_weight[trial.target_class, trial.feature] = trial.flipped_weight
            trigger_only, ssim = target_metrics(model, dataset, clean_weight, clean_bias, trial.target_class, device, args.test_batch_size, args.workers, spec, mask, pattern)
            bit_only, _ = target_metrics(model, dataset, flipped_weight, clean_bias, trial.target_class, device, args.test_batch_size, args.workers, spec)
            combined, _ = target_metrics(model, dataset, flipped_weight, clean_bias, trial.target_class, device, args.test_batch_size, args.workers, spec, mask, pattern)
            backdoor_accuracy = classifier_accuracy(model, dataset, flipped_weight, clean_bias, device, args.test_batch_size, args.workers)
            bit_distance = hamming(trial.original_weight, trial.flipped_weight)
        validation = causal_diagnostics(
            model, validation_images, validation_labels, clean_weight, clean_bias,
            trial, spec, mask, pattern,
        ) if trial.method == "causal" else {}
        result: Dict[str, object] = {
            "trial_key": key,
            **asdict(trial),
            "calibration_clean_accuracy": calibration_clean_accuracy,
            "calibration_flipped_accuracy": (
                candidate_lookup[(trial.target_class, trial.feature)].calibration_accuracy
                if trial.method != "tuap" else calibration_clean_accuracy
            ),
            "calibration_bad": (
                candidate_lookup[(trial.target_class, trial.feature)].calibration_bad
                if trial.method != "tuap" else 0.0
            ),
            "original_bits": bit_string(trial.original_weight) if trial.method != "tuap" else "",
            "flipped_bits": bit_string(trial.flipped_weight) if trial.method != "tuap" else "",
            "bit_hamming_distance": bit_distance if bit_distance is not None else "",
            "clean_model_accuracy": clean_accuracy,
            "backdoor_clean_accuracy": backdoor_accuracy,
            "bad": clean_accuracy - backdoor_accuracy,
            "trigger_only_asr": trigger_only,
            "bit_only_asr": bit_only,
            "trigger_plus_bit_asr": combined,
            "bit_marginal_gain": (combined - trigger_only) if trial.method != "tuap" else float("nan"),
            "mask_l1": float(mask.abs().sum()),
            "ssim": float(ssim) if ssim is not None else float("nan"),
            "loss_version": args.loss_version,
            "clean_margin": args.clean_margin,
            "attack_margin": args.attack_margin,
            **validation,
        }
        result["qualifies"] = (
            trial.method == "causal"
            and result["trigger_only_asr"] <= args.max_trigger_only_asr
            and result["bit_only_asr"] <= args.max_bit_only_asr
            and result["trigger_plus_bit_asr"] >= args.min_combined_asr
            and result["bad"] <= args.max_bad
            and result["bit_hamming_distance"] == 1
        )
        result["validation_qualifies"] = (
            trial.method == "causal"
            and float(result["validation_trigger_only_asr"]) <= args.max_trigger_only_asr
            and float(result["validation_bit_only_asr"]) <= args.max_bit_only_asr
            and float(result["validation_combined_asr"]) >= args.min_combined_asr
            and float(result["validation_bad"]) <= args.max_bad
            and result["bit_hamming_distance"] == 1
        ) if trial.method == "causal" else False
        results.append(result)
        torch.save({"trial": asdict(trial), "mask": mask, "pattern": pattern}, trial_trigger_dir / (key + ".pt"))
        completed.add(key)
        progress_path.write_text(json.dumps({
            "planned_trial_keys": sorted(expected_keys),
            "completed_trial_keys": sorted(completed),
            "results": results,
        }, indent=2), encoding="utf-8")

    if completed != expected_keys or len(results) != len(trials):
        raise RuntimeError("Incomplete progress state after trial loop; re-run with --resume.")
    all_triggers: Dict[str, Dict[str, object]] = {}
    for key in sorted(completed):
        trigger_path = trial_trigger_dir / (key + ".pt")
        if not trigger_path.is_file():
            raise RuntimeError("Missing trigger checkpoint: {0}".format(trigger_path))
        all_triggers[key] = torch.load(trigger_path, map_location="cpu", weights_only=True)
    torch.save(all_triggers, args.output_dir / "triggers.pt")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_summary(results, args.output_dir / "summary.csv", include_target=True)
    write_summary(results, args.output_dir / "summary_overall.csv", include_target=False)
    manifest: Dict[str, object] = {
        "offline_only": True,
        "author_artifacts_read": [str(args.clean_checkpoint)],
        "author_artifacts_not_read": ["*_potential_weights.npy", "*_neuron_trigger_pair.pkl", "backdoored_models/**/*.pth"],
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "calibration_clean_accuracy": calibration_clean_accuracy,
        "candidate_count_by_target": {str(target): len(candidates) for target, candidates in candidates_by_target.items()},
        "trial_count": len(results),
        "qualified_count": sum(bool(item["qualifies"]) for item in results),
        "method": args.method,
        "dataset": args.dataset,
    }
    qualified = [row for row in results if bool(row["qualifies"])]
    if qualified:
        result = max(qualified, key=lambda row: (
            float(row["trigger_plus_bit_asr"]), float(row["bit_marginal_gain"]), -float(row["mask_l1"])
        ))
        trial = Trial(**{field: result[field] for field in Trial.__dataclass_fields__})
        artifact = all_triggers[str(result["trial_key"])]
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        state["fc.weight"][trial.target_class, trial.feature] = trial.flipped_weight
        torch.save(state, args.output_dir / "best_causal_backdoored_model.pth")
        torch.save(artifact, args.output_dir / "best_causal_trigger.pt")
        manifest["best_model"] = result
    else:
        manifest["best_model"] = None
        manifest["note"] = "No causal trial met all acceptance thresholds."
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
