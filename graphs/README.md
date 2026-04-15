# Training Graphs

Generated from saved artifacts in `artifacts/`.

- Latest PPO run: `ppo_20260414_224350`
- Latest RLHF run: `rlhf_20260415_000643`

Files:
- `ppo_latest_learning_curve.png`: raw and smoothed episode reward for the latest PPO run.
- `rlhf_latest_learning_curve.png`: raw and smoothed reward-model episode reward for the latest RLHF run.
- `reward_model_loss.png`: reward-model train/validation loss by epoch.
- `ppo_run_comparison.png`: random baseline, trained PPO, and heuristic mean reward across PPO runs.
- `rlhf_run_comparison.png`: proxy reward before/after RLHF plus final RLHF score across RLHF runs.
- `latest_pipeline_summary.png`: one-view comparison of the latest PPO and RLHF results.

Regenerate with:
- `uv run python generate_training_graphs.py`
