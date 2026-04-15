from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
GRAPHS_DIR = ROOT / "graphs"


@dataclass
class RunSummary:
    run_name: str
    run_type: str
    path: Path
    data: dict[str, Any]


def load_summaries() -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for path in sorted(ARTIFACTS_DIR.glob("*/training_summary.json")):
        run_name = path.parent.name
        if run_name.startswith("ppo_"):
            run_type = "ppo"
        elif run_name.startswith("rlhf_"):
            run_type = "rlhf"
        else:
            continue
        summaries.append(
            RunSummary(
                run_name=run_name,
                run_type=run_type,
                path=path,
                data=json.loads(path.read_text(encoding="utf-8")),
            )
        )
    return summaries


def load_reward_history() -> list[dict[str, Any]]:
    for sub in ("reward_model_clean", "reward_model"):
        history_path = ARTIFACTS_DIR / sub / "reward_model_history.json"
        if history_path.exists():
            return json.loads(history_path.read_text(encoding="utf-8"))
    raise FileNotFoundError("No reward_model_history.json found.")


def select_latest_run(summaries: list[RunSummary], run_type: str) -> RunSummary:
    candidates = [summary for summary in summaries if summary.run_type == run_type]
    if run_type == "ppo":
        candidates = [summary for summary in candidates if summary.run_name != "ppo_smoke"]
    if not candidates:
        raise ValueError(f"No {run_type} summaries found.")
    return sorted(candidates, key=lambda summary: summary.run_name)[-1]


def read_curve(summary: RunSummary) -> pd.DataFrame:
    curve_path = summary.path.parent / "learning_curve.csv"
    frame = pd.read_csv(curve_path)
    frame["run_name"] = summary.run_name
    return frame


def setup_style() -> None:
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def save_plot(filename: str) -> None:
    GRAPHS_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_learning_curve(summary: RunSummary, filename: str, title: str, ylabel: str) -> None:
    frame = read_curve(summary)
    fig, ax = plt.subplots()
    ax.plot(
        frame["total_timesteps"],
        frame["episode_reward"],
        alpha=0.2,
        linewidth=1.0,
        label="Episode reward",
    )
    ax.plot(
        frame["total_timesteps"],
        frame["rolling_mean_reward"],
        linewidth=2.2,
        label="Rolling mean reward",
    )
    ax.set_title(title)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")
    save_plot(filename)


def plot_reward_model_loss(history: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(history)
    best_idx = int(frame["val_loss"].idxmin())
    best_row = frame.loc[best_idx]

    fig, ax = plt.subplots()
    ax.plot(frame["epoch"], frame["train_loss"], label="Train loss", linewidth=2.0)
    ax.plot(frame["epoch"], frame["val_loss"], label="Validation loss", linewidth=2.0)
    ax.scatter([best_row["epoch"]], [best_row["val_loss"]], s=50, label="Best val epoch", zorder=5)
    ax.set_title("Reward Model Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.legend(loc="upper right")
    save_plot("reward_model_loss.png")


def plot_ppo_run_comparison(summaries: list[RunSummary]) -> None:
    rows = []
    for summary in summaries:
        if summary.run_type != "ppo" or summary.run_name == "ppo_smoke":
            continue
        rows.append(
            {
                "run_name": summary.run_name,
                "trained_mean_reward": summary.data.get("trained_mean_reward"),
                "baseline_mean_reward": summary.data.get("baseline_mean_reward"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)

    fig, ax = plt.subplots()
    x = range(len(frame))
    width = 0.35
    ax.bar([i - width / 2 for i in x], frame["baseline_mean_reward"].fillna(0.0), width=width, label="Random baseline")
    ax.bar([i + width / 2 for i in x], frame["trained_mean_reward"].fillna(0.0), width=width, label="Trained PPO")
    ax.set_xticks(list(x), [f"Run {i+1}" for i in x])
    ax.set_title("PPO Policy Quality Across Saved Runs")
    ax.set_xlabel("PPO Run")
    ax.set_ylabel("Mean proxy reward")
    ax.legend(loc="upper left")
    save_plot("ppo_run_comparison.png")


def plot_rlhf_run_comparison(summaries: list[RunSummary]) -> None:
    rows = []
    for summary in summaries:
        if summary.run_type != "rlhf":
            continue
        rows.append(
            {
                "run_name": summary.run_name,
                "proxy_score_before": summary.data["proxy_score_before"],
                "proxy_score_after": summary.data["proxy_score_after"],
                "rlhf_score_after": summary.data["rlhf_score_after"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)

    fig, ax = plt.subplots()
    x = range(len(frame))
    width = 0.35

    ax.bar([i - width / 2 for i in x], frame["proxy_score_before"], width=width, label="Before RLHF")
    ax.bar([i + width / 2 for i in x], frame["proxy_score_after"], width=width, label="After RLHF")
    ax.set_xticks(list(x), [f"Run {i+1}" for i in x])
    ax.set_title("Proxy Reward Before vs After RLHF")
    ax.set_xlabel("RLHF Run")
    ax.set_ylabel("Mean proxy reward")
    ax.legend(loc="upper left")

    save_plot("rlhf_run_comparison.png")


def plot_latest_pipeline_summary(latest_ppo: RunSummary, latest_rlhf: RunSummary) -> None:
    ppo = latest_ppo.data
    rlhf = latest_rlhf.data
    labels = [
        "Random",
        "PPO before\nRLHF",
        "RLHF fine-tuned",
    ]
    values = [
        ppo["baseline_mean_reward"],
        rlhf["proxy_score_before"],
        rlhf["proxy_score_after"],
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars = ax.bar(labels, values)
    ax.set_title("Latest Training Pipeline Summary")
    ax.set_ylabel("Mean proxy reward")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_plot("latest_pipeline_summary.png")


def write_index(latest_ppo: RunSummary, latest_rlhf: RunSummary) -> None:
    lines = [
        "# Training Graphs",
        "",
        "Generated from saved artifacts in `artifacts/`.",
        "",
        f"- Latest PPO run: `{latest_ppo.run_name}`",
        f"- Latest RLHF run: `{latest_rlhf.run_name}`",
        "",
        "Files:",
        "- `ppo_latest_learning_curve.png`: raw and smoothed episode reward for the latest PPO run.",
        "- `rlhf_latest_learning_curve.png`: raw and smoothed reward-model episode reward for the latest RLHF run.",
        "- `reward_model_loss.png`: reward-model train/validation loss by epoch.",
        "- `ppo_run_comparison.png`: random baseline, trained PPO, and heuristic mean reward across PPO runs.",
        "- `rlhf_run_comparison.png`: proxy reward before/after RLHF plus final RLHF score across RLHF runs.",
        "- `latest_pipeline_summary.png`: one-view comparison of the latest PPO and RLHF results.",
        "",
        "Regenerate with:",
        "- `uv run python generate_training_graphs.py`",
    ]
    (GRAPHS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    summaries = load_summaries()
    latest_ppo = select_latest_run(summaries, "ppo")
    latest_rlhf = select_latest_run(summaries, "rlhf")
    reward_history = load_reward_history()

    plot_learning_curve(
        latest_ppo,
        filename="ppo_latest_learning_curve.png",
        title="PPO Learning Curve",
        ylabel="Proxy reward",
    )
    plot_learning_curve(
        latest_rlhf,
        filename="rlhf_latest_learning_curve.png",
        title="RLHF Fine-Tuning Curve",
        ylabel="Blended reward (proxy + RLHF)",
    )
    plot_reward_model_loss(reward_history)
    plot_ppo_run_comparison(summaries)
    plot_rlhf_run_comparison(summaries)
    plot_latest_pipeline_summary(latest_ppo, latest_rlhf)
    write_index(latest_ppo, latest_rlhf)


if __name__ == "__main__":
    main()
