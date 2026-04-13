from __future__ import annotations

"""Merge annotation files from multiple annotators into a single training dataset.

For each pair, takes a majority vote across annotators. Ties and all-skipped
pairs are dropped. Maps A/B preferences back to winner/loser sequences using
the true source labels stored in pairs.json.

Usage
-----
python merge_labels.py \\
    --pairs pairs.json \\
    --labels labels_neeraja.json labels_hanwen.json labels_lilian.json \\
    --output merged_labels.json
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _majority_vote(votes: list[str | None]) -> str | None:
    """Return 'A', 'B', or None (tie or all skipped)."""
    a = sum(1 for v in votes if v == "A")
    b = sum(1 for v in votes if v == "B")
    if a == 0 and b == 0:
        return None  # all skipped
    if a > b:
        return "A"
    if b > a:
        return "B"
    return None  # tie


def _pairwise_agreement(
    votes_by_annotator: dict[str, dict[int, str | None]],
) -> float:
    """Fraction of (annotator pair, pair_id) combinations where both voted and agreed."""
    annotators = list(votes_by_annotator.keys())
    agree = total = 0
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a_votes = votes_by_annotator[annotators[i]]
            b_votes = votes_by_annotator[annotators[j]]
            shared_ids = {pid for pid, v in a_votes.items() if v is not None} & \
                         {pid for pid, v in b_votes.items() if v is not None}
            for pid in shared_ids:
                total += 1
                if a_votes[pid] == b_votes[pid]:
                    agree += 1
    return agree / total if total > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge annotator label files into a single training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", default="pairs.json", help="pairs.json from sample_sequences.py")
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more annotator label JSON files.",
    )
    parser.add_argument("--output", default="merged_labels.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load pairs (ground truth sequences + source labels)
    pairs_data = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    pairs: dict[int, dict[str, Any]] = {p["pair_id"]: p for p in pairs_data["pairs"]}

    # Load all label files
    votes_by_annotator: dict[str, dict[int, str | None]] = {}
    for label_path in args.labels:
        data = json.loads(Path(label_path).read_text(encoding="utf-8"))
        annotator_id: str = data.get("annotator_id") or label_path
        votes_by_annotator[annotator_id] = {
            a["pair_id"]: a["preference"] for a in data["annotations"]
        }

    annotator_ids = list(votes_by_annotator.keys())
    print(f"Annotators: {annotator_ids}")
    print(f"Total pairs: {len(pairs)}")

    # Aggregate votes per pair
    training_pairs: list[dict[str, Any]] = []
    dropped_ties = 0
    dropped_all_skipped = 0

    for pair_id, pair in sorted(pairs.items()):
        votes = [votes_by_annotator[a].get(pair_id) for a in annotator_ids]
        consensus = _majority_vote(votes)

        if consensus is None:
            non_null = [v for v in votes if v is not None]
            if not non_null:
                dropped_all_skipped += 1
            else:
                dropped_ties += 1
            continue

        # Map A/B → winner/loser sequence
        if consensus == "A":
            winner_seq = pair["sequence_a"]
            loser_seq = pair["sequence_b"]
        else:
            winner_seq = pair["sequence_b"]
            loser_seq = pair["sequence_a"]

        votes_for_winner = sum(1 for v in votes if v == consensus)
        total_votes = sum(1 for v in votes if v is not None)

        training_pairs.append(
            {
                "pair_id": pair_id,
                "winner": winner_seq,
                "loser": loser_seq,
                "consensus": consensus,
                "votes_for_winner": votes_for_winner,
                "total_votes": total_votes,
                "raw_votes": {a: votes_by_annotator[a].get(pair_id) for a in annotator_ids},
            }
        )

    agreement = _pairwise_agreement(votes_by_annotator)

    stats = {
        "total_pairs": len(pairs),
        "usable_pairs": len(training_pairs),
        "dropped_ties": dropped_ties,
        "dropped_all_skipped": dropped_all_skipped,
        "annotator_ids": annotator_ids,
        "pairwise_agreement": round(agreement, 4),
    }

    output = {
        "pairs_metadata": pairs_data.get("metadata", {}),
        "stats": stats,
        "training_pairs": training_pairs,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Usable pairs:        {stats['usable_pairs']} / {stats['total_pairs']}")
    print(f"Dropped (ties):      {stats['dropped_ties']}")
    print(f"Dropped (all skip):  {stats['dropped_all_skipped']}")
    print(f"Pairwise agreement:  {stats['pairwise_agreement']:.1%}")
    print(f"Wrote → {out_path}")


if __name__ == "__main__":
    main()
