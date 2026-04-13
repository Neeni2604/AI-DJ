# AI-DJ

This repo trains an RL agent to sequence tracks in a DJ environment. There are
three main stages in the current pipeline:

1. PPO training on the hand-designed proxy reward in `DJEnv`.
2. Reward-model training on preference data.
3. RLHF fine-tuning, where PPO starts from a proxy-trained checkpoint and is
   optimized against the learned reward model.

**Basic Commands**

- Run the environment:
  `uv run python dj_env.py --db fma_db/data/fma.db --limit 500`
- Train PPO on the proxy reward:
  `uv run python train_ppo.py --db fma_db/data/fma.db --limit 64 --timesteps 16384 --require-upward-trend`
- Generate the training graphs:
  `uv run python generate_training_graphs.py`

PPO and RLHF runs write artifacts such as `learning_curve.csv`,
`training_summary.json`, and `demo_episode.json` into `artifacts/`. The
generated visualizations are saved in `graphs/`.

**Graph Guide**

The current graph set is:

- `graphs/pipeline_diagrams.md`
- `graphs/ppo_latest_learning_curve.png`
- `graphs/rlhf_latest_learning_curve.png`
- `graphs/reward_model_loss.png`
- `graphs/ppo_run_comparison.png`
- `graphs/rlhf_run_comparison.png`
- `graphs/latest_pipeline_summary.png`

These graphs are generated from the saved artifact files already present in the
repo, not from live retraining.

`graphs/pipeline_diagrams.md` contains step-by-step pipeline diagrams for the
full algorithm, PPO training, preference data collection, reward-model
training, RLHF fine-tuning, and demo/inference rollouts.

**1. PPO Latest Learning Curve**

File: `graphs/ppo_latest_learning_curve.png`

This graph shows the latest saved PPO run, `ppo_20260413_004855`.

- The x-axis is total training timesteps.
- The faint line is the raw episode reward for each completed episode.
- The darker line is the rolling mean reward written into
  `learning_curve.csv`.

How to read it:

- If the dark rolling-mean line moves upward over time, PPO is learning a
  better policy under the proxy reward.
- The gap between the noisy raw line and the smoother rolling line shows normal
  episodic variance.
- A flat or downward rolling line would indicate weak or unstable improvement.

What this run shows:

- The saved summary reports an initial smoothed reward of `7.0593` and a final
  smoothed reward of `8.2757`.
- That is a curve improvement of about `+1.2164`, so the run does improve over
  training.
- The final trained policy mean reward is `8.0709`, which beats the random
  baseline `7.4781` but is still far below the heuristic baseline `11.3320`.

Interpretation:

- PPO is learning something useful from the proxy reward.
- The policy is not yet competitive with the handcrafted heuristic on this
  saved run.
- Improvement exists, but it is modest compared with the best PPO run in this
  repo.

**2. RLHF Latest Learning Curve**

File: `graphs/rlhf_latest_learning_curve.png`

This graph shows the latest saved RLHF run, `rlhf_20260413_005446`.

- The x-axis is total training timesteps during fine-tuning.
- The y-axis is the terminal reward assigned by the learned reward model.
- As with the PPO graph, the faint line is per-episode reward and the dark line
  is the rolling mean.

How to read it:

- In RLHF, the absolute reward scale matters less than whether the rolling
  average improves.
- A less negative score can still be an improvement if the learned reward model
  outputs negative values overall.

What this run shows:

- The initial smoothed RLHF reward is `-0.1304` and the final smoothed reward
  is `-0.1224`.
- That is a small improvement of about `+0.0080`.
- The reward-model score after fine-tuning is still negative, `-0.1237`, but it
  is slightly better than where the run started.

Interpretation:

- RLHF fine-tuning is moving in the intended direction, but only slightly.
- The reward-model signal looks weak or compressed, because training changes
  are small relative to the PPO proxy-reward stage.
- This is consistent with a reward model that is not yet very informative.

**3. Reward Model Loss**

File: `graphs/reward_model_loss.png`

This graph plots reward-model training loss and validation loss by epoch.

- The training-loss line shows fit on the training preference pairs.
- The validation-loss line shows generalization to held-out preference pairs.
- The marked best point is the epoch with the lowest validation loss.

How to read it:

- A healthy training curve usually has training loss decreasing and validation
  loss either decreasing or staying stable.
- If validation loss rises while training loss falls or oscillates lower, that
  suggests overfitting or a weak dataset.

What this run shows:

- The best validation loss occurs immediately at epoch `1`.
- The best saved validation loss is `0.7070`.
- After that, validation loss trends upward instead of down.

Interpretation:

- The reward model is not getting better after the first epoch.
- That usually means one of three things: too little preference data, noisy
  labels, or a reward-model architecture/training setup that is too weak for
  the task.
- This graph is the strongest warning sign in the current pipeline.

**4. PPO Run Comparison**

File: `graphs/ppo_run_comparison.png`

This graph compares saved PPO runs against two reference policies:

- Random baseline
- Trained PPO policy
- Heuristic policy

How to read it:

- Higher bars are better because the metric is mean proxy reward.
- The trained PPO bar should ideally clear the random baseline by a good margin.
- The heuristic bar acts as a rough upper reference for the current handcrafted
  scoring logic.

What this graph shows:

- `ppo_20260410_002422` is the strongest saved PPO run with mean reward
  `10.2409`.
- `ppo_20260413_004855` is weaker at `8.0709`.
- The heuristic stays around `11.27` to `11.33` on the runs where it was
  recorded.

Interpretation:

- PPO can learn a substantially better policy than random.
- The latest PPO run is not the best PPO checkpoint in the repository.
- If this repo is used for reporting final results, the older `20260410` PPO
  run is currently the stronger saved example.

**5. RLHF Run Comparison**

File: `graphs/rlhf_run_comparison.png`

This figure has two panels.

- Left: proxy reward before vs after RLHF fine-tuning.
- Right: final reward-model score after RLHF.

How to read it:

- The left panel answers whether RLHF helped or hurt the original proxy-based
  objective.
- The right panel shows how well the final policy scores under the learned
  reward model.

What this graph shows:

- `rlhf_20260413_005446` improves proxy reward from `8.1051` to `8.2579`,
  a gain of `+0.1528`.
- `rlhf_20260413_000436` gets worse on proxy reward, from `9.9051` to `9.1785`,
  a drop of about `-0.7266`.
- The final RLHF rewards are both still negative, around `-0.124` to `-0.125`.

Interpretation:

- RLHF is not consistently helping yet.
- One saved fine-tuning run slightly improves the starting PPO policy, while
  another clearly regresses.
- The instability again points back to the reward model as the likely bottleneck.

**6. Latest Pipeline Summary**

File: `graphs/latest_pipeline_summary.png`

This graph is the quickest single-slide summary of the latest saved training
pipeline.

It compares:

- Random policy reward
- Heuristic policy reward
- Latest PPO policy reward
- PPO checkpoint score before RLHF
- RLHF fine-tuned score after RLHF

How to read it:

- This graph is useful for telling the story of the pipeline end-to-end.
- It shows where PPO sits relative to simple baselines and whether RLHF adds
  value on top of PPO.

What this graph shows:

- Random: `7.4781`
- Heuristic: `11.3320`
- Latest PPO: `8.0709`
- PPO before RLHF: `8.1051`
- RLHF after fine-tuning: `8.2579`

Interpretation:

- PPO clears the random baseline but remains far below the heuristic.
- The latest RLHF run gives a small bump over its starting PPO checkpoint.
- The full pipeline is not yet strong enough to claim that RLHF reliably
  outperforms the best PPO results in the repo.

**Overall Takeaways**

- PPO training appears functional. The learning curves generally move upward.
- The best saved PPO run is stronger than the latest PPO run.
- RLHF fine-tuning is not yet consistently beneficial.
- The reward-model training curve suggests the preference model is the weakest
  part of the pipeline right now.

If these results are going into a report or presentation, the fairest summary is
that the PPO stage works, but RLHF performance is currently limited by reward
model quality and training stability.
