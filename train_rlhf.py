from __future__ import annotations

"""RLHF fine-tuning for AI DJ.

Wraps DJEnv so that the proxy reward is replaced by a terminal reward from
the learned preference reward model.  A pre-trained PPO policy is then
fine-tuned against this new signal.

The reward model scores the *full episode sequence* once at termination;
intermediate steps return 0 so PPO learns to optimise long-run sequence
quality rather than individual transitions.

Artifacts written to output_dir/:
  ppo_rlhf.zip             fine-tuned SB3 model
  train.monitor.csv        RLHF episode rewards during fine-tuning
  learning_curve.csv       smoothed learning curve
  demo_episode.json        one rollout with proxy-reward breakdown
  training_summary.json    run metadata + before/after evaluation

Usage
-----
python train_rlhf.py \\
    --starting-model artifacts/ppo_20260410_002422/ppo_ai_dj.zip \\
    --reward-model-config artifacts/reward_model/reward_model_config.json \\
    --db fma_db/data/fma.db
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from dj_env import DJEnv, TRANSITION_TYPES
from train_reward_model import load_reward_model, encode_sequence
from train_ppo import read_monitor_file, write_learning_curve

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _key_label(key: int, mode: int) -> str:
    return f"{KEY_NAMES[int(key) % 12]} {'maj' if mode == 1 else 'min'}"


def _track_to_step(track: dict, transition_type: str | None, step: int) -> dict:
    return {
        "step": step,
        "tempo": track["tempo"],
        "energy": track["energy"],
        "key": _key_label(track["key"], track["mode"]),
        "transition_type": transition_type,
    }


# ---------------------------------------------------------------------------
# RLHF environment wrapper
# ---------------------------------------------------------------------------

class RLHFDJEnv(gym.Wrapper):
    """DJEnv wrapper that swaps proxy rewards for preference-model terminal rewards.

    reward(t) = 0                          for t < episode_length
    reward(t) = reward_model(sequence)     at termination
    """

    def __init__(
        self,
        env: DJEnv,
        reward_model: torch.nn.Module,
        model_config: dict,
        device: torch.device,
    ) -> None:
        super().__init__(env)
        self._reward_model = reward_model
        self._model_config = model_config
        self._device = device
        self._steps: list[dict] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        inner: DJEnv = self.env.unwrapped
        first = inner.tracks[inner._current_idx]
        self._steps = [_track_to_step(first, None, 0)]
        return obs, info

    def step(self, action):
        obs, _proxy_reward, terminated, truncated, info = self.env.step(action)
        inner: DJEnv = self.env.unwrapped
        track = inner.tracks[inner._current_idx]
        self._steps.append(_track_to_step(track, info.get("transition_type"), info["step"]))

        reward = self._score() if (terminated or truncated) else 0.0
        return obs, reward, terminated, truncated, info

    def _score(self) -> float:
        max_steps = self._model_config["max_steps"]
        features = encode_sequence(self._steps, max_steps)
        x = torch.tensor(features, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            return float(self._reward_model(x).item())


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

def _build_rlhf_env(
    *,
    db_path: Path,
    limit: int | None,
    episode_length: int,
    repeat_track_penalty: float,
    reward_model: torch.nn.Module,
    model_config: dict,
    device: torch.device,
    monitor_path: Path | None = None,
) -> Monitor:
    inner = DJEnv(
        db_path=db_path,
        limit=limit,
        episode_length=episode_length,
        repeat_track_penalty=repeat_track_penalty,
    )
    wrapped = RLHFDJEnv(inner, reward_model, model_config, device)
    return Monitor(wrapped, str(monitor_path)) if monitor_path else Monitor(wrapped)


def _build_proxy_env(
    *,
    db_path: Path,
    limit: int | None,
    episode_length: int,
    repeat_track_penalty: float,
) -> Monitor:
    """Plain DJEnv with proxy reward — used for objective-metric evaluation."""
    env = DJEnv(
        db_path=db_path,
        limit=limit,
        episode_length=episode_length,
        repeat_track_penalty=repeat_track_penalty,
    )
    return Monitor(env)


# ---------------------------------------------------------------------------
# Demo episode
# ---------------------------------------------------------------------------

def run_demo_episode(
    model: PPO,
    *,
    db_path: Path,
    limit: int | None,
    episode_length: int,
    repeat_track_penalty: float,
    seed: int,
) -> dict[str, Any]:
    """One rollout on the proxy env so we can inspect transition-level metrics."""
    set_random_seed(seed)
    env = _build_proxy_env(
        db_path=db_path,
        limit=limit,
        episode_length=episode_length,
        repeat_track_penalty=repeat_track_penalty,
    )
    try:
        obs, info = env.reset(seed=seed)
        steps: list[dict[str, Any]] = [
            {
                "step": 0,
                "fma_track_id": info["fma_track_id"],
                "title": info["title"],
                "artist": info["artist"],
                "genre": info["genre"],
            }
        ]
        episode_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps.append(
                {
                    "step": info["step"],
                    "fma_track_id": info["fma_track_id"],
                    "title": info["title"],
                    "artist": info["artist"],
                    "genre": info["genre"],
                    "transition_type": info["transition_type"],
                    "reward": float(reward),
                    "harmonic_score": float(info["harmonic_score"]),
                    "bpm_score": float(info["bpm_score"]),
                    "transition_score": float(info["transition_score"]),
                    "energy_flow": float(info["energy_flow"]),
                    "repeat_penalty": float(info["repeat_penalty"]),
                }
            )
        return {"sampling_mode": "stochastic", "episode_reward": episode_reward, "steps": steps}
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a PPO DJ agent with a learned preference reward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="fma_db/data/fma.db")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=12)
    parser.add_argument("--repeat-penalty", type=float, default=0.25)
    parser.add_argument(
        "--starting-model",
        required=True,
        help="Path to pre-trained proxy-reward PPO zip to fine-tune from.",
    )
    parser.add_argument(
        "--reward-model-config",
        default="artifacts/reward_model/reward_model_config.json",
        help="Path to reward_model_config.json saved by train_reward_model.py.",
    )
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Lower LR than proxy training — fine-tuning, not training from scratch.")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class RLHFSummary:
    db_path: str
    starting_model: str
    reward_model_config: str
    model_path: str
    total_timesteps: int
    episode_length: int
    track_limit: int | None
    eval_episodes: int
    # Proxy-env scores (objective metrics)
    proxy_score_before: float
    proxy_score_before_std: float
    proxy_score_after: float
    proxy_score_after_std: float
    proxy_improvement: float
    # RLHF-env score (learned reward)
    rlhf_score_after: float
    rlhf_score_after_std: float
    # Learning curve
    episode_count: int
    initial_window_mean: float
    final_window_mean: float
    curve_improvement: float
    rolling_slope: float


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    rm_config_path = Path(args.reward_model_config)
    if not rm_config_path.exists():
        raise FileNotFoundError(f"Reward model config not found: {rm_config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("artifacts") / f"rlhf_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor_path = out_dir / "train.monitor.csv"
    curve_path   = out_dir / "learning_curve.csv"
    demo_path    = out_dir / "demo_episode.json"
    summary_path = out_dir / "training_summary.json"
    model_path   = out_dir / "ppo_rlhf"

    # Load reward model
    print("Loading reward model …")
    reward_model, rm_config = load_reward_model(rm_config_path, device=str(device))
    reward_model.eval()

    # Environments
    train_env = _build_rlhf_env(
        db_path=db_path,
        limit=args.limit,
        episode_length=args.episode_length,
        repeat_track_penalty=args.repeat_penalty,
        reward_model=reward_model,
        model_config=rm_config,
        device=device,
        monitor_path=monitor_path,
    )
    proxy_eval_env = _build_proxy_env(
        db_path=db_path,
        limit=args.limit,
        episode_length=args.episode_length,
        repeat_track_penalty=args.repeat_penalty,
    )
    rlhf_eval_env = _build_rlhf_env(
        db_path=db_path,
        limit=args.limit,
        episode_length=args.episode_length,
        repeat_track_penalty=args.repeat_penalty,
        reward_model=reward_model,
        model_config=rm_config,
        device=device,
    )

    try:
        # Load pre-trained model and swap in the RLHF training env
        print(f"Loading starting model: {args.starting_model}")
        model = PPO.load(
            args.starting_model,
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            verbose=args.verbose,
            device=str(device),
        )

        # Baseline: proxy score of the starting policy before fine-tuning
        print("Evaluating starting policy on proxy env …")
        proxy_before_mean, proxy_before_std = evaluate_policy(
            model, proxy_eval_env, n_eval_episodes=args.eval_episodes, deterministic=False
        )

        # Fine-tune
        print(f"Fine-tuning for {args.timesteps} timesteps …")
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
        model.save(str(model_path))

        # Evaluation after fine-tuning
        print("Evaluating fine-tuned policy …")
        proxy_after_mean, proxy_after_std = evaluate_policy(
            model, proxy_eval_env, n_eval_episodes=args.eval_episodes, deterministic=False
        )
        rlhf_after_mean, rlhf_after_std = evaluate_policy(
            model, rlhf_eval_env, n_eval_episodes=args.eval_episodes, deterministic=False
        )

    finally:
        train_env.close()
        proxy_eval_env.close()
        rlhf_eval_env.close()

    # Learning curve
    monitor_rows = read_monitor_file(monitor_path)
    trend_window, initial_mean, final_mean, _best, slope = write_learning_curve(
        monitor_rows, curve_path
    )

    # Demo episode (proxy env so we get transition-level breakdown)
    reloaded = PPO.load(str(model_path) + ".zip")
    demo = run_demo_episode(
        reloaded,
        db_path=db_path,
        limit=args.limit,
        episode_length=args.episode_length,
        repeat_track_penalty=args.repeat_penalty,
        seed=args.seed,
    )
    demo_path.write_text(json.dumps(demo, indent=2), encoding="utf-8")

    summary = RLHFSummary(
        db_path=str(db_path),
        starting_model=args.starting_model,
        reward_model_config=str(rm_config_path),
        model_path=str(model_path) + ".zip",
        total_timesteps=args.timesteps,
        episode_length=args.episode_length,
        track_limit=args.limit,
        eval_episodes=args.eval_episodes,
        proxy_score_before=float(proxy_before_mean),
        proxy_score_before_std=float(proxy_before_std),
        proxy_score_after=float(proxy_after_mean),
        proxy_score_after_std=float(proxy_after_std),
        proxy_improvement=float(proxy_after_mean - proxy_before_mean),
        rlhf_score_after=float(rlhf_after_mean),
        rlhf_score_after_std=float(rlhf_after_std),
        episode_count=len(monitor_rows),
        initial_window_mean=initial_mean,
        final_window_mean=final_mean,
        curve_improvement=final_mean - initial_mean,
        rolling_slope=slope,
    )

    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
