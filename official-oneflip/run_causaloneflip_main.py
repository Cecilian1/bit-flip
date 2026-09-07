"""Orchestrate fixed-parameter CIFAR-10 CausalONEFLIP main experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def selected_parameters(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best")
    if not payload.get("has_qualifying_setting") or not best:
        raise SystemExit("No validation-qualified causal setting exists; causal main experiment is gated.")
    return best


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def causal_command(args: argparse.Namespace, method: str, output: Path) -> list[str]:
    common = [
        sys.executable, "./causal_oneflip.py", "--method", method,
        "--dataset", "CIFAR10", "--clean-checkpoint", "./saved_model/resnet_CIFAR10/clean_model_1.pth",
        "--data-root", "../dataset/CIFAR10", "--output-dir", str(output),
        "--device", args.device, "--target-classes", args.target_classes,
        "--seeds", args.seeds, "--calibration-size", "1024", "--optimization-size", "768",
        "--trigger-epochs", str(args.trigger_epochs), "--optimization-batch-size", "128",
        "--test-batch-size", "512", "--workers", str(args.workers),
    ]
    if args.resume:
        common.append("--resume")
    if method == "causal":
        selected = selected_parameters(args.selection)
        return common + [
            "--betas", str(selected["beta"]), "--mask-weights", str(selected["mask_weight"]),
            "--clean-margin", str(selected["clean_margin"]), "--attack-margin", str(selected["attack_margin"]),
            "--loss-version", "causal-margin-v2",
        ]
    selected = json.loads(args.selection.read_text(encoding="utf-8")).get("best") or json.loads(args.selection.read_text(encoding="utf-8"))["closest_setting"]
    return common + ["--betas", "1", "--mask-weights", str(selected["mask_weight"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("causal-main", "oneflip-baseline", "tuap-baseline", "report-cifar10", "main-all"))
    parser.add_argument("--selection", type=Path, default=Path("./runs/causaloneflip-tune-class6/selected_params.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("./runs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="20260904,20260905,20260906")
    parser.add_argument("--trigger-epochs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.selection.is_file():
        raise SystemExit("Selection record not found: {0}".format(args.selection))

    outputs = {
        "causal-main": args.runs_dir / "causaloneflip-cifar10-causal-main",
        "oneflip-baseline": args.runs_dir / "causaloneflip-cifar10-oneflip-baseline",
        "tuap-baseline": args.runs_dir / "causaloneflip-cifar10-tuap-baseline",
    }
    method_for_phase = {"causal-main": "causal", "oneflip-baseline": "oneflip", "tuap-baseline": "tuap"}
    phases = list(method_for_phase) if args.phase == "main-all" else [args.phase]
    for phase in phases:
        if phase == "report-cifar10":
            continue
        run(causal_command(args, method_for_phase[phase], outputs[phase]))
    if args.phase in ("report-cifar10", "main-all"):
        report = args.runs_dir / "causaloneflip-cifar10-report"
        command = [
            sys.executable, "./aggregate_causaloneflip_results.py", "--dataset", "CIFAR10",
            "--input", "causal=" + str(outputs["causal-main"] / "results.csv"),
            "--input", "oneflip=" + str(outputs["oneflip-baseline"] / "results.csv"),
            "--input", "tuap=" + str(outputs["tuap-baseline"] / "results.csv"),
            "--output-dir", str(report),
        ]
        run(command)


if __name__ == "__main__":
    main()
