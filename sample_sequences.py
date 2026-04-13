from __future__ import annotations

"""Generate sequence pairs from one or two policies for human preference annotation.

Each pair contains two 5-track sequences (configurable) drawn from source A and
source B. Either source can be a trained PPO model or a random policy. Pairs are
written to a JSON file consumed by annotate.py.

Usage examples
--------------
# PPO model vs random policy (standard RLHF data collection)
python sample_sequences.py \\
    --model-a artifacts/ppo_20260406_220618/ppo_ai_dj.zip \\
    --db fma_db/data/fma.db --n-pairs 50 --output pairs.json

# Two PPO models against each other (for H2 evaluation)
python sample_sequences.py \\
    --model-a artifacts/ppo_proxy/ppo_ai_dj.zip \\
    --model-b artifacts/ppo_rlhf/ppo_ai_dj.zip \\
    --db fma_db/data/fma.db --n-pairs 100 --output pairs.json
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from dj_env import DJEnv, HeuristicDJPolicy

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _key_label(key: int, mode: int) -> str:
    name = KEY_NAMES[int(key) % 12]
    return f"{name} {'maj' if mode == 1 else 'min'}"


def _make_env(db_path: Path, limit: int | None, sequence_length: int) -> DJEnv:
    return DJEnv(db_path=db_path, limit=limit, episode_length=sequence_length)


def _source_label(model_arg: str | None) -> str:
    if model_arg is None:
        return "random"
    if model_arg.lower() == "heuristic":
        return "heuristic"
    return f"ppo:{Path(model_arg).name}"


def _build_policy(model_arg: str | None, env: DJEnv):
    """Return a policy object (PPO, HeuristicDJPolicy, or None for random)."""
    if model_arg is None:
        return None
    if model_arg.lower() == "heuristic":
        return HeuristicDJPolicy(env)
    return PPO.load(model_arg)


def _rollout(
    env: DJEnv,
    model,
    sequence_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one episode and return a list of annotator-readable step dicts."""
    obs, info = env.reset(seed=seed)

    first = env.tracks[env._current_idx]
    steps: list[dict[str, Any]] = [
        {
            "step": 0,
            "fma_track_id": info["fma_track_id"],
            "title": info["title"],
            "artist": info["artist"],
            "genre": info["genre"],
            "tempo": round(first["tempo"], 1),
            "energy": round(first["energy"], 3),
            "key": _key_label(first["key"], first["mode"]),
            "transition_type": None,
        }
    ]

    for _ in range(sequence_length):
        if model is not None:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = env.action_space.sample()

        obs, _reward, terminated, truncated, info = env.step(action)
        track = env.tracks[env._current_idx]
        steps.append(
            {
                "step": info["step"],
                "fma_track_id": info["fma_track_id"],
                "title": info["title"],
                "artist": info["artist"],
                "genre": info["genre"],
                "tempo": round(track["tempo"], 1),
                "energy": round(track["energy"], 3),
                "key": _key_label(track["key"], track["mode"]),
                "transition_type": info["transition_type"],
            }
        )
        if terminated or truncated:
            break

    return steps


# Maximum BPM allowed between consecutive tracks.  Beyond this the transition
# is musically incoherent regardless of transition type.
_MAX_BPM_DELTA = 30.0


def _sequence_ok(steps: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return (True, 'ok') or (False, reason) for a generated sequence."""
    track_ids = [s["fma_track_id"] for s in steps]
    if len(track_ids) != len(set(track_ids)):
        return False, "repeated track"
    for i in range(1, len(steps)):
        delta = abs(steps[i]["tempo"] - steps[i - 1]["tempo"])
        if delta > _MAX_BPM_DELTA:
            return False, f"BPM jump {delta:.1f} at step {i}"
    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequence pairs for human preference annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="fma_db/data/fma.db")
    parser.add_argument("--limit", type=int, default=500, help="Tracks to load from DB.")
    parser.add_argument(
        "--model-a",
        default=None,
        metavar="PATH",
        help="SB3 PPO model zip for source A. Omit for random policy.",
    )
    parser.add_argument(
        "--model-b",
        default=None,
        metavar="PATH",
        help="SB3 PPO model zip for source B. Omit for random policy.",
    )
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of tracks per sequence (first track + this many transitions).",
    )
    parser.add_argument("--output", default="pairs.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    rng = random.Random(args.seed)

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    label_a = _source_label(args.model_a)
    label_b = _source_label(args.model_b)

    env_a = _make_env(db_path, args.limit, args.sequence_length)
    env_b = _make_env(db_path, args.limit, args.sequence_length)

    model_a = _build_policy(args.model_a, env_a)
    model_b = _build_policy(args.model_b, env_b)

    _MAX_RETRIES = 50  # attempts per pair before giving up

    pairs: list[dict[str, Any]] = []
    try:
        for i in range(args.n_pairs):
            seq_a = seq_b = None
            for attempt in range(_MAX_RETRIES):
                base = args.seed + i * _MAX_RETRIES + attempt
                candidate_a = _rollout(env_a, model_a, args.sequence_length, seed=base * 2)
                candidate_b = _rollout(env_b, model_b, args.sequence_length, seed=base * 2 + 1)
                ok_a, reason_a = _sequence_ok(candidate_a)
                ok_b, reason_b = _sequence_ok(candidate_b)
                if ok_a and ok_b:
                    seq_a, seq_b = candidate_a, candidate_b
                    break
            if seq_a is None:
                print(f"\nWarning: could not generate clean pair {i + 1} after {_MAX_RETRIES} attempts — skipping.")
                continue

            # Randomly swap A/B so position bias doesn't favour either policy
            if rng.random() < 0.5:
                seq_a, seq_b = seq_b, seq_a
                src_a, src_b = label_b, label_a
            else:
                src_a, src_b = label_a, label_b

            pairs.append(
                {
                    "pair_id": len(pairs),
                    "source_a": src_a,
                    "source_b": src_b,
                    "sequence_a": seq_a,
                    "sequence_b": seq_b,
                }
            )
            print(f"Generated pair {len(pairs)}/{args.n_pairs}", end="\r", flush=True)
    finally:
        env_a.close()
        env_b.close()

    output: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "db_path": str(db_path),
            "track_limit": args.limit,
            "sequence_length": args.sequence_length,
            "n_pairs": args.n_pairs,
            "source_a": label_a,
            "source_b": label_b,
            "seed": args.seed,
        },
        "pairs": pairs,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {len(pairs)} pairs → {out_path}")


if __name__ == "__main__":
    main()
