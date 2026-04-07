uv sync
echo "Syncing uv ENV"
source .venv/bin/activate
echo "Activated Python ENV"
echo "Setting up Database this will take a Minute ..."
python3 setup_db.py
echo "Finished Setting up Data Base!"
echo "Beginning Training!"

python3 train_ppo.py --db fma_db/data/fma.db --subset small --limit 64 --timesteps 16384 --require-upward-trend


# uv run setup_db.py
# uv run train_ppo.py --db fma_db/data/fma.db --limit 64 --timesteps 2048 --eval-episodes 5 --output-dir artifacts/ppo_smoke --verbose 0
# uv run train_ppo.py --db fma_db/data/fma.db --limit 64 --timesteps 16384 --require-upward-trend
