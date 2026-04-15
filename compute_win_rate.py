from __future__ import annotations

"""Compute pairwise win rates from a merged labels file.

Usage
-----
python compute_win_rate.py --pairs pairs_h2.json --labels labels_h2_clean.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute win rates from merged labels.")
    parser.add_argument("--pairs", required=True, help="pairs.json file used during annotation.")
    parser.add_argument("--labels", required=True, help="merged labels file from merge_labels.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs_data = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    labels_data = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    pairs = {p["pair_id"]: p for p in pairs_data["pairs"]}

    wins: Counter[str] = Counter()
    for item in labels_data["training_pairs"]:
        pair = pairs[item["pair_id"]]
        winner_source = pair["source_a"] if item["consensus"] == "A" else pair["source_b"]
        wins[winner_source] += 1

    total = sum(wins.values())
    print(f"Total usable pairs: {total}")
    print()
    print(f"{'Source':<35} {'Wins':>6} {'Win Rate':>10}")
    print("-" * 55)
    for source, count in wins.most_common():
        print(f"{source:<35} {count:>6} {100*count/total:>9.1f}%")


if __name__ == "__main__":
    main()
