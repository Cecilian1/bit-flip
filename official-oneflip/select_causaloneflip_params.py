"""Select fixed CausalONEFLIP hyperparameters from a completed tuning CSV.

This script is deliberately read-only with respect to experimental outputs: it
creates a separate ranking CSV and JSON decision record.  It does not change
or delete any trigger/model artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def mean(rows: List[Dict[str, str]], name: str) -> float:
    return statistics.mean(float(row[name]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank CausalONEFLIP beta/lambda settings")
    parser.add_argument("--results", type=Path, required=True, action="append", help="Causal tuning results.csv; repeat to combine fine runs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing tuning output directory")
    args = parser.parse_args()
    if any(not path.is_file() for path in args.results):
        raise SystemExit("A results CSV was not found: {0}".format(args.results))
    if not args.output_dir.is_dir():
        raise SystemExit("Output directory not found: {0}".format(args.output_dir))

    source = []
    for path in args.results:
        with path.open(newline="", encoding="utf-8") as stream:
            source.extend(row for row in csv.DictReader(stream) if row["method"] == "causal")
    if not source:
        raise SystemExit("The CSV has no causal trials.")

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in source:
        groups[(row["beta"], row["mask_weight"], row.get("clean_margin", "0"), row.get("attack_margin", "0"))].append(row)

    ranking: List[Dict[str, object]] = []
    for (beta, mask_weight, clean_margin, attack_margin), rows in groups.items():
        qualified = sum(row.get("validation_qualifies", "false").strip().lower() == "true" for row in rows)
        ranking.append({
            "beta": float(beta),
            "mask_weight": float(mask_weight),
            "clean_margin": float(clean_margin),
            "attack_margin": float(attack_margin),
            "trials": len(rows),
            "qualified_trials": qualified,
            "qualification_rate": qualified / len(rows),
            "trigger_only_asr_mean": mean(rows, "validation_trigger_only_asr"),
            "bit_only_asr_mean": mean(rows, "validation_bit_only_asr"),
            "combined_asr_mean": mean(rows, "validation_combined_asr"),
            "bit_marginal_gain_mean": mean(rows, "validation_counterfactual_gain"),
            "mask_l1_mean": mean(rows, "mask_l1"),
        })

    # First prefer settings that generate qualifying pairs.  Ties intentionally
    # follow the paper-facing priority: combined effect, causal marginal gain,
    # then lower perturbation magnitude.
    ranking.sort(key=lambda item: (
        -float(item["qualification_rate"]),
        -float(item["combined_asr_mean"]),
        -float(item["bit_marginal_gain_mean"]),
        float(item["trigger_only_asr_mean"]),
        float(item["mask_l1_mean"]),
    ))
    ranking_path = args.output_dir / "parameter_ranking.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)

    best = ranking[0]
    has_qualifying_setting = int(best["qualified_trials"]) > 0
    selected = {
        "source_results": [str(path) for path in args.results],
        "selection_rule": "held-out qualification rate, combined ASR, counterfactual gain, lower trigger-only ASR, lower mask L1",
        "best": best if has_qualifying_setting else None,
        "closest_setting": best,
        "shortlist": ranking[:3],
        "has_qualifying_setting": has_qualifying_setting,
        "note": (
            "Use the selected beta and mask_weight for the fixed main experiment."
            if has_qualifying_setting
            else "No setting produced a qualifying causal pair; do not continue to the CIFAR-100 extension."
        ),
    }
    (args.output_dir / "selected_params.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
