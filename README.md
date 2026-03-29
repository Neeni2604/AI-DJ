Notes:

- Run this to show that the environment is working:
  `python dj_env.py --db fma_db/data/fma.db --limit 500`
- Train PPO on the proxy reward and save the model plus learning-curve artifacts:
  `python train_ppo.py --db fma_db/data/fma.db --limit 64 --timesteps 16384 --require-upward-trend`
- PPO training writes `ppo_ai_dj.zip`, `learning_curve.csv`, `training_summary.json`, and `demo_episode.json` into `artifacts/`.
