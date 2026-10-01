"""Compute observed metrics, Wilson intervals, and video-cluster bootstrap CIs."""
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def metrics(counts):
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / n if n else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [middle - half, middle + half]


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(__file__).parent / "data" / "predictions" / "longtail_predictions.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "longtail_ci.json")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--cluster-column", default="video_id", help="Use animal_id only if a verified animal_id column exists")
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("resamples must be positive")
    groups = defaultdict(Counter)
    seen = set()
    with args.predictions.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "video_id", "y_true", "y_pred", args.cluster_column}
        if not required.issubset(reader.fieldnames or []):
            parser.error(f"Missing columns: {required - set(reader.fieldnames or [])}")
        for row in reader:
            name = row["image_path"]
            group = row[args.cluster_column]
            if not name or not group or name in seen:
                raise ValueError(f"Missing group/path or duplicated image: {name}")
            seen.add(name)
            true, pred = int(row["y_true"]), int(row["y_pred"])
            if true not in (0, 1) or pred not in (0, 1):
                raise ValueError(f"Invalid binary class at {name}")
            # 0=closed is the clinically relevant positive class.
            category = "tp" if (true, pred) == (0, 0) else "fp" if (true, pred) == (1, 0) else "fn" if (true, pred) == (0, 1) else "tn"
            groups[group][category] += 1
    if not groups:
        raise ValueError("No predictions")
    total = sum(groups.values(), Counter())
    point = metrics(total)
    n = sum(total.values())
    basic_wilson = {
        "accuracy": wilson(total["tp"] + total["tn"], n),
        "precision": wilson(total["tp"], total["tp"] + total["fp"]),
        "recall": wilson(total["tp"], total["tp"] + total["fn"]),
    }
    rng = random.Random(args.seed)
    keys = sorted(groups)
    draws = {name: [] for name in point}
    undefined = Counter()
    for _ in range(args.resamples):
        replicate = sum((groups[keys[rng.randrange(len(keys))]] for _ in keys), Counter())
        for name, value in metrics(replicate).items():
            if value is None:
                undefined[name] += 1
            else:
                draws[name].append(value)
    # Do not quietly drop undefined F1/precision/recall resamples and present a
    # conditional interval as though it covered the requested sampling process.
    ci = {name: ([percentile(vals, .025), percentile(vals, .975)] if len(vals) == args.resamples else None) for name, vals in draws.items()}
    result = {
        "source": str(args.predictions), "positive_class": "0=closed", "n_frames": n,
        "n_clusters": len(keys), "cluster_column": args.cluster_column, "frames_per_cluster": {k: sum(groups[k].values()) for k in keys},
        "confusion_counts": {key: total[key] for key in ("tp", "fp", "fn", "tn")},
        "metrics": point, "wilson_frame_independence_assumption": basic_wilson,
        "cluster_bootstrap_percentile_95_ci": ci, "undefined_bootstrap_replicates": dict(undefined),
        "resamples": args.resamples, "seed": args.seed,
        "interpretation": "Wilson treats frames as independent and is descriptive only. Cluster intervals resample entire source groups; with very few groups they are exploratory. Conditional on the saved fixed model and test set; does not address training-set or animal leakage.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
