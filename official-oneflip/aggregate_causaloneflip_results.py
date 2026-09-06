"""Build paper-facing CausalONEFLIP tables from completed result CSV files.

The program is read-only for input experiments and writes a fresh report
directory. It preserves every qualifying row so the aggregate cannot hide
selection bias.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METRICS = (
    "trigger_only_asr", "bit_only_asr", "trigger_plus_bit_asr",
    "bit_marginal_gain", "bad", "mask_l1", "ssim",
)


def parse_input(value: str) -> Tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("Each --input must use LABEL=PATH.")
    return label, Path(raw_path)


def finite_values(rows: Iterable[Dict[str, str]], metric: str) -> List[float]:
    values = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def bootstrap(values: Sequence[float], seed: int) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    generator = random.Random(seed)
    means = sorted(sum(generator.choice(values) for _ in values) / len(values) for _ in range(1000))
    return means[24], means[974]


def metric_summary(rows: List[Dict[str, str]], metric: str, seed: int) -> Dict[str, object]:
    values = finite_values(rows, metric)
    low, high = bootstrap(values, seed)
    return {
        metric + "_n": len(values),
        metric + "_mean": statistics.mean(values) if values else float("nan"),
        metric + "_std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else float("nan"),
        metric + "_ci95_low": low,
        metric + "_ci95_high": high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate complete CausalONEFLIP experiment outputs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", action="append", required=True, type=parse_input,
                        help="Method label and results.csv: LABEL=PATH. Repeat for every method.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Output directory already exists: {0}".format(args.output_dir))

    report: List[Dict[str, object]] = []
    qualifying: List[Dict[str, str]] = []
    for index, (label, path) in enumerate(args.input):
        if not path.is_file():
            raise SystemExit("Result CSV not found: {0}".format(path))
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise SystemExit("Result CSV is empty: {0}".format(path))
        item: Dict[str, object] = {
            "dataset": args.dataset,
            "method": label,
            "source_results": str(path),
            "trials": len(rows),
            "qualified_pairs": sum(row.get("qualifies", "").lower() == "true" for row in rows),
        }
        item["qualification_rate"] = int(item["qualified_pairs"]) / len(rows)
        for metric in METRICS:
            item.update(metric_summary(rows, metric, seed=20260904 + index))
        report.append(item)
        for row in rows:
            if row.get("qualifies", "").lower() == "true":
                qualifying.append({**row, "method": label, "dataset": args.dataset})

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    qualified_path = args.output_dir / "qualified_pairs.csv"
    if qualifying:
        with qualified_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(qualifying[0]))
            writer.writeheader()
            writer.writerows(qualifying)
    else:
        qualified_path.write_text("method,dataset\n", encoding="utf-8")
    print("Wrote {0} and {1}".format(args.output_dir / "comparison_summary.csv", qualified_path))


if __name__ == "__main__":
    main()
