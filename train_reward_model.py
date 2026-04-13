from __future__ import annotations

"""Train a preference reward model from merged human annotation data.

The model is a small MLP that takes a flattened sequence of audio features
and outputs a scalar reward score. It is trained with a Bradley-Terry pairwise
ranking loss: for each (winner, loser) pair the model is penalised whenever it
scores the loser higher than the winner.

Feature encoding (5 values per track × 6 tracks = 30 inputs by default):
  - tempo:          normalised to [0, 1] using range [40, 220]
  - energy:         already in [0, 1]
  - transition_cut:       1 if cut,       else 0  (0 for the first track)
  - transition_fade:      1 if fade,      else 0
  - transition_beatmatch: 1 if beatmatch, else 0

Usage
-----
python train_reward_model.py --merged-labels merged_labels.json
python train_reward_model.py --merged-labels merged_labels.json --epochs 100 --lr 1e-3
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

TEMPO_MIN, TEMPO_MAX = 40.0, 220.0
TRANSITION_TYPES = ["cut", "fade", "beatmatch"]
FEATURES_PER_STEP = 5  # tempo, energy, cut, fade, beatmatch


def _encode_step(step: dict[str, Any]) -> list[float]:
    tempo = float(np.clip((step.get("tempo", 120.0) - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN), 0.0, 1.0))
    energy = float(np.clip(step.get("energy", 0.5), 0.0, 1.0))

    transition = step.get("transition_type")
    t_cut       = 1.0 if transition == "cut"       else 0.0
    t_fade      = 1.0 if transition == "fade"      else 0.0
    t_beatmatch = 1.0 if transition == "beatmatch" else 0.0

    return [tempo, energy, t_cut, t_fade, t_beatmatch]


def encode_sequence(steps: list[dict[str, Any]], max_steps: int) -> np.ndarray:
    """Encode a sequence of track steps into a flat feature vector."""
    features: list[float] = []
    for step in steps[:max_steps]:
        features.extend(_encode_step(step))
    # zero-pad if shorter than max_steps
    pad = max_steps - min(len(steps), max_steps)
    features.extend([0.0] * (pad * FEATURES_PER_STEP))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """MLP that maps a flat sequence feature vector to a scalar reward."""

    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def bradley_terry_loss(r_winner: torch.Tensor, r_loser: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss."""
    return -torch.mean(torch.log(torch.sigmoid(r_winner - r_loser) + 1e-8))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    input_dim: int
    hidden_sizes: list[int]
    dropout: float
    weight_decay: float
    max_steps: int
    epochs: int
    lr: float
    val_split: float
    seed: int
    merged_labels_path: str
    output_dir: str


def _build_dataset(
    training_pairs: list[dict[str, Any]],
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (winners, losers) as float32 arrays of shape (N, input_dim)."""
    winners, losers = [], []
    for pair in training_pairs:
        winners.append(encode_sequence(pair["winner"], max_steps))
        losers.append(encode_sequence(pair["loser"],  max_steps))
    return np.stack(winners), np.stack(losers)


def train(config: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = json.loads(Path(config.merged_labels_path).read_text(encoding="utf-8"))
    training_pairs: list[dict[str, Any]] = data["training_pairs"]
    print(f"Training pairs: {len(training_pairs)}")

    winners_np, losers_np = _build_dataset(training_pairs, config.max_steps)

    # Train / val split
    n = len(winners_np)
    indices = list(range(n))
    random.shuffle(indices)
    val_n = max(1, int(n * config.val_split))
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    def _tensors(idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.tensor(winners_np[idx], dtype=torch.float32, device=device)
        l = torch.tensor(losers_np[idx],  dtype=torch.float32, device=device)
        return w, l

    w_train, l_train = _tensors(train_idx)
    w_val,   l_val   = _tensors(val_idx)

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    # Model
    model = RewardModel(config.input_dim, config.hidden_sizes, dropout=config.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "reward_model_best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = bradley_terry_loss(model(w_train), model(l_train))
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = bradley_terry_loss(model(w_val), model(l_val)).item()
        train_loss_val = train_loss.item()

        history.append({"epoch": epoch, "train_loss": train_loss_val, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if epoch % max(1, config.epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{config.epochs}  train={train_loss_val:.4f}  val={val_loss:.4f}")

    print(f"\nBest val loss {best_val_loss:.4f} at epoch {best_epoch}")

    # Save final model (last epoch)
    final_model_path = out_dir / "reward_model.pt"
    torch.save(model.state_dict(), final_model_path)

    # Save config so the model can be reloaded later
    model_config = {
        "input_dim": config.input_dim,
        "hidden_sizes": config.hidden_sizes,
        "dropout": config.dropout,
        "max_steps": config.max_steps,
        "features_per_step": FEATURES_PER_STEP,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
    }
    config_path = out_dir / "reward_model_config.json"
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    # Save training history
    history_path = out_dir / "reward_model_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Saved model     → {final_model_path}")
    print(f"Saved best      → {best_model_path}")
    print(f"Saved config    → {config_path}")
    return model_config


# ---------------------------------------------------------------------------
# Loader (used by RLHF training script)
# ---------------------------------------------------------------------------

def load_reward_model(config_path: str | Path, device: str = "cpu") -> tuple[RewardModel, dict]:
    """Load a saved reward model from its config file."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = RewardModel(cfg["input_dim"], cfg["hidden_sizes"], dropout=cfg.get("dropout", 0.0))
    weights_path = Path(config_path).parent / "reward_model_best.pt"
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.to(device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a preference reward model on merged annotation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--merged-labels", default="merged_labels.json")
    parser.add_argument("--output-dir", default="artifacts/reward_model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[32, 16],
        help="Hidden layer widths, e.g. --hidden-sizes 32 16",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 regularisation.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Max tracks per sequence (sequence_length + 1). Must match pairs.json.",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dim = args.max_steps * FEATURES_PER_STEP

    config = TrainConfig(
        input_dim=input_dim,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        merged_labels_path=args.merged_labels,
        output_dir=args.output_dir,
    )
    train(config)


if __name__ == "__main__":
    main()
