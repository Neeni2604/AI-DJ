from __future__ import annotations

"""Train and evaluate a PPO policy for the AI DJ environment.

This script uses Stable Baselines3 PPO to optimize the proxy reward defined in
`DJEnv`. It writes five artifacts into an output directory:

- `ppo_ai_dj.zip`: the trained SB3 model
- `train.monitor.csv`: episode rewards logged during training
- `learning_curve.csv`: monitor data with a rolling-mean reward column
- `demo_episode.json`: one sampled rollout from the trained policy
- `training_summary.json`: a compact summary of the run

The success criterion exposed by `--require-upward-trend` is intentionally tied
to the logged learning curve, not only to final-policy evaluation. That keeps
the script aligned with the project requirement to verify that rewards trend
upward during training.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from dj_env import DJEnv, HeuristicDJPolicy


@dataclass
class TrainingSummary:
    """Structured summary saved at the end of training."""

    db_path: str
    model_path: str
    monitor_path: str
    curve_path: str
    demo_path: str
    total_timesteps: int
    episode_length: int
    track_limit: int | None
    repeat_track_penalty: float
    eval_episodes: int
    evaluation_sampling_mode: str
    baseline_mean_reward: float
    baseline_reward_std: float
    heuristic_mean_reward: float
    heuristic_reward_std: float
    trained_mean_reward: float
    trained_reward_std: float
    episode_count: int
    trend_window: int
    initial_window_mean: float
    final_window_mean: float
    best_episode_reward: float
    rolling_slope: float
    curve_improvement: float
    policy_improvement: float
    min_improvement: float
    policy_beats_random: bool
    trend_upward: bool


def build_env(
    *,
    db_path: Path,
    subset_name: str | None,
    limit: int | None,
    episode_length: int,
    repeat_track_penalty: float,
    monitor_path: Path | None = None,
) -> Monitor:
    """Create a monitored `DJEnv` instance for training or evaluation."""

    env = DJEnv(
        db_path=db_path,
        subset_name=subset_name,
        limit=limit,
        episode_length=episode_length,
        repeat_track_penalty=repeat_track_penalty,
    )
    if monitor_path is not None:
        return Monitor(env, str(monitor_path))
    return Monitor(env)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing rolling mean with a variable-sized warm-up prefix."""

    window = max(1, int(window))
    means = np.zeros_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        means[idx] = float(np.mean(values[start:idx + 1]))
    return means


def read_monitor_file(monitor_path: Path) -> list[dict[str, float]]:
    """Load SB3 monitor CSV rows and reconstruct cumulative timesteps."""

    rows: list[dict[str, float]] = []
    total_steps = 0

    with monitor_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        if not first_line.startswith("#"):
            handle.seek(0)

        reader = csv.DictReader(handle)
        for row in reader:
            reward = float(row["r"])
            length = int(float(row["l"]))
            elapsed = float(row["t"])
            total_steps += length
            rows.append(
                {
                    "episode": float(len(rows) + 1),
                    "episode_reward": reward,
                    "episode_length": float(length),
                    "elapsed_seconds": elapsed,
                    "total_timesteps": float(total_steps),
                }
            )

    return rows


def write_learning_curve(
    monitor_rows: list[dict[str, float]],
    curve_path: Path,
) -> tuple[int, float, float, float, float]:
    """Write a CSV learning curve and return aggregate trend statistics.

    Returns:
        trend_window: Window size used for smoothing.
        initial_window_mean: Mean of the earliest smoothed window.
        final_window_mean: Mean of the latest smoothed window.
        best_episode_reward: Best raw episode reward observed.
        rolling_slope: Slope of a fitted line over the smoothed rewards.
    """

    rewards = np.array([row["episode_reward"] for row in monitor_rows], dtype=np.float64)
    if rewards.size == 0:
        raise ValueError("No completed episodes were recorded during PPO training.")

    trend_window = min(25, max(5, rewards.size // 5))
    trend_window = min(trend_window, rewards.size)
    smoothed = rolling_mean(rewards, trend_window)

    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
                "total_timesteps",
                "episode_length",
                "episode_reward",
                "rolling_mean_reward",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(monitor_rows):
            writer.writerow(
                {
                    "episode": int(row["episode"]),
                    "total_timesteps": int(row["total_timesteps"]),
                    "episode_length": int(row["episode_length"]),
                    "episode_reward": f"{row['episode_reward']:.6f}",
                    "rolling_mean_reward": f"{smoothed[idx]:.6f}",
                    "elapsed_seconds": f"{row['elapsed_seconds']:.6f}",
                }
            )

    initial_window_mean = float(np.mean(smoothed[:trend_window]))
    final_window_mean = float(np.mean(smoothed[-trend_window:]))
    best_episode_reward = float(np.max(rewards))
    rolling_slope = (
        float(np.polyfit(np.arange(len(smoothed), dtype=np.float64), smoothed, 1)[0])
        if len(smoothed) > 1
        else 0.0
    )
    return trend_window, initial_window_mean, final_window_mean, best_episode_reward, rolling_slope


def evaluate_random_policy(env: Monitor, n_eval_episodes: int, seed: int) -> tuple[float, float]:
    """Estimate a random-action baseline on the same environment configuration."""

    episode_rewards: list[float] = []
    for episode_idx in range(n_eval_episodes):
        observation, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    rewards = np.array(episode_rewards, dtype=np.float64)
    return float(np.mean(rewards)), float(np.std(rewards))


def evaluate_heuristic_policy(env: Monitor, n_eval_episodes: int, seed: int) -> tuple[float, float]:
    """Estimate a greedy-heuristic baseline on the same environment configuration."""

    policy = HeuristicDJPolicy(env)
    episode_rewards: list[float] = []
    for episode_idx in range(n_eval_episodes):
        observation, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = policy.predict(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    rewards = np.array(episode_rewards, dtype=np.float64)
    return float(np.mean(rewards)), float(np.std(rewards))


def run_demo_episode(
    model: PPO,
    *,
    db_path: Path,
    subset_name: str | None,
    limit: int | None,
    episode_length: int,
    repeat_track_penalty: float,
    seed: int,
) -> dict[str, Any]:
    """Sample one rollout from the trained policy and serialize its decisions."""

    set_random_seed(seed)
    env = build_env(
        db_path=db_path,
        subset_name=subset_name,
        limit=limit,
        episode_length=episode_length,
        repeat_track_penalty=repeat_track_penalty,
    )
    try:
        observation, info = env.reset(seed=seed)
        steps: list[dict[str, Any]] = [
            {
                "step": 0,
                "fma_track_id": info["fma_track_id"],
                "title": info["title"],
                "artist": info["artist"],
                "genre": info["genre"],
            }
        ]

        terminated = False
        truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=False)
            observation, reward, terminated, truncated, info = env.step(action)
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

        return {
            "sampling_mode": "stochastic",
            "episode_reward": episode_reward,
            "steps": steps,
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training, evaluation, and artifact output."""

    parser = argparse.ArgumentParser(
        description="Train a PPO AI DJ policy on the proxy reward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="fma_db/data/fma.db", help="Path to the SQLite FMA database.")
    parser.add_argument("--subset", default=None, help="Optional FMA subset filter, e.g. 'small'.")
    parser.add_argument("--limit", type=int, default=500, help="Number of tracks to load from SQLite.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=12,
        help="Number of transitions per episode.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.25,
        help="Reward penalty applied when the agent chooses the current track again.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=16_384,
        help="Total PPO environment steps to train for.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Episodes used for random-baseline and trained-policy evaluation.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for training and demo sampling.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO optimizer learning rate.")
    parser.add_argument("--n-steps", type=int, default=512, help="Rollout steps collected per PPO update.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for PPO updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient. Higher values encourage more exploration.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Width of each hidden layer in the policy and value networks.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.1,
        help="Minimum learning-curve gain required by `--require-upward-trend`.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for training artifacts. Defaults to a timestamped folder under `artifacts/`.",
    )
    parser.add_argument(
        "--require-upward-trend",
        action="store_true",
        help="Exit non-zero if the smoothed learning curve does not trend upward enough.",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Stable Baselines3 verbosity level.")
    return parser.parse_args()


def resolve_output_dir(output_dir: str | None) -> Path:
    """Return the artifact directory, creating it if needed."""

    if output_dir:
        path = Path(output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("artifacts") / f"ppo_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    """Train PPO, evaluate it, and write all run artifacts to disk."""

    args = parse_args()
    set_random_seed(args.seed)

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    output_dir = resolve_output_dir(args.output_dir)
    monitor_path = output_dir / "train.monitor.csv"
    curve_path = output_dir / "learning_curve.csv"
    demo_path = output_dir / "demo_episode.json"
    summary_path = output_dir / "training_summary.json"
    model_path = output_dir / "ppo_ai_dj"

    env = build_env(
        db_path=db_path,
        subset_name=args.subset,
        limit=args.limit,
        episode_length=args.episode_length,
        repeat_track_penalty=args.repeat_penalty,
        monitor_path=monitor_path,
    )
    eval_env = build_env(
        db_path=db_path,
        subset_name=args.subset,
        limit=args.limit,
        episode_length=args.episode_length,
        repeat_track_penalty=args.repeat_penalty,
    )

    try:
        # `MultiDiscrete([n_tracks, 3])` is handled natively by SB3 PPO, so the
        # policy learns both the next track choice and the transition type.
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=args.verbose,
            policy_kwargs={"net_arch": [args.hidden_size, args.hidden_size]},
        )

        baseline_mean_reward, baseline_reward_std = evaluate_random_policy(
            eval_env,
            n_eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
        heuristic_mean_reward, heuristic_reward_std = evaluate_heuristic_policy(
            eval_env,
            n_eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
        model.learn(total_timesteps=args.timesteps)
        trained_mean_reward, trained_reward_std = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=False,
        )
        model.save(str(model_path))
    finally:
        env.close()
        eval_env.close()

    monitor_rows = read_monitor_file(monitor_path)
    (
        trend_window,
        initial_window_mean,
        final_window_mean,
        best_episode_reward,
        rolling_slope,
    ) = write_learning_curve(monitor_rows, curve_path)

    curve_improvement = final_window_mean - initial_window_mean
    policy_improvement = float(trained_mean_reward - baseline_mean_reward)
    trend_upward = curve_improvement >= args.min_improvement and rolling_slope > 0.0
    policy_beats_random = policy_improvement >= 0.0

    # Reload from disk before the demo rollout so the saved artifact is the one
    # being exercised, not just the in-memory model.
    reloaded_model = PPO.load(str(model_path) + ".zip")
    demo_episode = run_demo_episode(
        reloaded_model,
        db_path=db_path,
        subset_name=args.subset,
        limit=args.limit,
        episode_length=args.episode_length,
        repeat_track_penalty=args.repeat_penalty,
        seed=args.seed,
    )
    demo_path.write_text(json.dumps(demo_episode, indent=2), encoding="utf-8")

    summary = TrainingSummary(
        db_path=str(db_path),
        model_path=str(model_path) + ".zip",
        monitor_path=str(monitor_path),
        curve_path=str(curve_path),
        demo_path=str(demo_path),
        total_timesteps=args.timesteps,
        episode_length=args.episode_length,
        track_limit=args.limit,
        repeat_track_penalty=args.repeat_penalty,
        eval_episodes=args.eval_episodes,
        evaluation_sampling_mode="stochastic",
        baseline_mean_reward=float(baseline_mean_reward),
        baseline_reward_std=float(baseline_reward_std),
        heuristic_mean_reward=float(heuristic_mean_reward),
        heuristic_reward_std=float(heuristic_reward_std),
        trained_mean_reward=float(trained_mean_reward),
        trained_reward_std=float(trained_reward_std),
        episode_count=len(monitor_rows),
        trend_window=trend_window,
        initial_window_mean=initial_window_mean,
        final_window_mean=final_window_mean,
        best_episode_reward=best_episode_reward,
        rolling_slope=rolling_slope,
        curve_improvement=curve_improvement,
        policy_improvement=policy_improvement,
        min_improvement=args.min_improvement,
        policy_beats_random=policy_beats_random,
        trend_upward=trend_upward,
    )

    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))

    if args.require_upward_trend and not trend_upward:
        raise SystemExit(
            "Learning curve did not improve enough. "
            f"Curve gain={curve_improvement:.3f}, policy gain={policy_improvement:.3f}."
        )


if __name__ == "__main__":
    main()
