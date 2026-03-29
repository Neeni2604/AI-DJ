from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Camelot wheel — harmonic compatibility
# ---------------------------------------------------------------------------

# Echonest key: 0=C, 1=C#, ..., 11=B   mode: 1=major, 0=minor
# Encoded as key * 2 + (0 if major, 1 if minor)  =>  0-23
CAMELOT: dict[int, str] = {
    0: "8B",  2: "3B",  4: "10B", 6: "5B",  8: "12B", 10: "7B",
    12: "2B", 14: "9B", 16: "4B", 18: "11B", 20: "6B", 22: "1B",
    1: "5A",  3: "12A", 5: "7A",  7: "2A",   9: "9A",  11: "4A",
    13: "11A",15: "6A", 17: "1A", 19: "8A",  21: "3A", 23: "10A",
}

CAMELOT_COMPATIBLE: dict[str, list[str]] = {
    "1A": ["1A","2A","12A","1B"],  "2A": ["2A","3A","1A","2B"],
    "3A": ["3A","4A","2A","3B"],   "4A": ["4A","5A","3A","4B"],
    "5A": ["5A","6A","4A","5B"],   "6A": ["6A","7A","5A","6B"],
    "7A": ["7A","8A","6A","7B"],   "8A": ["8A","9A","7A","8B"],
    "9A": ["9A","10A","8A","9B"],  "10A":["10A","11A","9A","10B"],
    "11A":["11A","12A","10A","11B"],"12A":["12A","1A","11A","12B"],
    "1B": ["1B","2B","12B","1A"],  "2B": ["2B","3B","1B","2A"],
    "3B": ["3B","4B","2B","3A"],   "4B": ["4B","5B","3B","4A"],
    "5B": ["5B","6B","4B","5A"],   "6B": ["6B","7B","5B","6A"],
    "7B": ["7B","8B","6B","7A"],   "8B": ["8B","9B","7B","8A"],
    "9B": ["9B","10B","8B","9A"],  "10B":["10B","11B","9B","10A"],
    "11B":["11B","12B","10B","11A"],"12B":["12B","1B","11B","12A"],
}

TRANSITION_TYPES = ["cut", "fade", "beatmatch"]


# ---------------------------------------------------------------------------
# Feature extraction from FMA JSON blobs
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_echonest_features(echonest_json: str | None) -> dict:
    """
    Pull audio features from the echonest JSON blob stored by ingest.py.
    The multi-level CSV headers are flattened with '__', so keys look like:
    'echonest__audio_features__tempo', 'echonest__audio_features__key', etc.
    """
    defaults = {
        "tempo": 120.0, "key": 0, "mode": 1,
        "energy": 0.5,  "valence": 0.5,
        "danceability": 0.5, "loudness": -10.0,
    }
    if not echonest_json:
        return defaults
    try:
        blob = json.loads(echonest_json)
    except (json.JSONDecodeError, TypeError):
        return defaults

    def _get(suffix: str):
        for k, v in blob.items():
            if k.lower().endswith(suffix.lower()):
                return v
        return None

    return {
        "tempo":        _safe_float(_get("tempo"),        120.0),
        "key":          int(_safe_float(_get("key"),      0)),
        "mode":         int(_safe_float(_get("mode"),     1)),
        "energy":       _safe_float(_get("energy"),       0.5),
        "valence":      _safe_float(_get("valence"),      0.5),
        "danceability": _safe_float(_get("danceability"), 0.5),
        "loudness":     _safe_float(_get("loudness"),     -10.0),
    }


def _extract_librosa_features(features_json: str | None) -> dict:
    """
    Pull summary features from the librosa JSON blob stored by ingest.py.
    We use mean chroma (harmonic content proxy) and mean RMS energy.
    """
    defaults = {"chroma_mean": 0.5, "rms_mean": 0.5}
    if not features_json:
        return defaults
    try:
        blob = json.loads(features_json)
    except (json.JSONDecodeError, TypeError):
        return defaults

    def _find_mean(prefix: str) -> float | None:
        for k, v in blob.items():
            if prefix.lower() in k.lower() and "mean" in k.lower():
                return _safe_float(v, None)
        return None

    chroma = _find_mean("chroma")
    rms    = _find_mean("rmse") or _find_mean("rms")
    return {
        "chroma_mean": chroma if chroma is not None else 0.5,
        "rms_mean":    rms    if rms    is not None else 0.5,
    }


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _camelot_key(key: int, mode: int) -> str:
    encoded = (key % 12) * 2 + (0 if mode == 1 else 1)
    return CAMELOT.get(encoded, "1A")


def harmonic_compatibility(key1: int, mode1: int, key2: int, mode2: int) -> float:
    c1 = _camelot_key(key1, mode1)
    c2 = _camelot_key(key2, mode2)
    return 1.0 if c2 in CAMELOT_COMPATIBLE.get(c1, []) else 0.0


def bpm_smoothness(bpm1: float, bpm2: float) -> float:
    """1.0 at delta<=5 BPM, 0.0 at delta>=30 BPM, linear in between."""
    delta = abs(bpm1 - bpm2)
    return float(np.clip(1.0 - (delta - 5.0) / 25.0, 0.0, 1.0))


def transition_reward(transition: int, bpm_delta: float, energy_delta: float) -> float:
    """Bonus for choosing a contextually appropriate transition type."""
    if transition == 2:    # beatmatch — best when BPM is close
        return float(np.clip(1.0 - bpm_delta / 20.0, 0.0, 1.0))
    elif transition == 1:  # fade — best for moderate energy shift
        return float(np.clip(1.0 - abs(energy_delta - 0.2) / 0.3, 0.0, 1.0))
    else:                  # cut — best for large energy shift
        return float(np.clip(abs(energy_delta) / 0.5, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Database loader
# ---------------------------------------------------------------------------

def load_tracks_from_db(
    db_path: str | Path,
    subset_name: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    Query the FMA SQLite database built by your partner's ingest.py,
    join fma_tracks with fma_track_features, and extract audio features
    from the echonest and librosa JSON blobs.

    Args:
        db_path:     Path to the .db file produced by initialize_database()
        subset_name: Optional FMA subset filter ('small', 'medium', etc.)
        limit:       Optional cap on number of tracks loaded

    Returns:
        List of track dicts with fields:
        fma_track_id, title, artist_name, genre_top, audio_path,
        tempo, key, mode, energy, valence, danceability, loudness,
        chroma_mean, rms_mean
    """
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            t.fma_track_id,
            t.title,
            t.artist_name,
            t.genre_top,
            t.audio_path,
            f.echonest_json,
            f.features_json
        FROM fma_tracks t
        LEFT JOIN fma_track_features f ON t.fma_track_id = f.fma_track_id
        WHERE (f.echonest_json IS NOT NULL OR f.features_json IS NOT NULL)
    """
    params: list = []

    if subset_name:
        query += " AND t.subset_name = ?"
        params.append(subset_name)

    if limit:
        query += f" LIMIT {int(limit)}"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    tracks = []
    for row in rows:
        echonest = _extract_echonest_features(row["echonest_json"])
        librosa  = _extract_librosa_features(row["features_json"])
        tracks.append({
            "fma_track_id":  row["fma_track_id"],
            "title":         row["title"] or f"Track {row['fma_track_id']}",
            "artist_name":   row["artist_name"] or "Unknown",
            "genre_top":     row["genre_top"] or "Unknown",
            "audio_path":    row["audio_path"],
            **echonest,
            **librosa,
        })

    return tracks


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class DJEnv(gym.Env):
    """
    AI DJ environment backed by the FMA SQLite database.

    Observation (9 features, normalised to [0, 1]):
        tempo, key, mode, energy, valence, danceability,
        loudness, chroma_mean, rms_mean

    Action:
        MultiDiscrete([n_tracks, 3])
        - Index of next track to play
        - Transition type: 0=cut, 1=fade, 2=beatmatch

    Proxy reward (replaced by your RLHF preference model in Week 3):
        harmonic_score   * 0.4
        bpm_score        * 0.3
        transition_score * 0.2
        energy_flow      * 0.1
        repeat_penalty   for choosing the same track again
    """

    metadata = {"render_modes": ["human"]}

    FEATURE_KEYS = [
        "tempo", "key", "mode", "energy", "valence",
        "danceability", "loudness", "chroma_mean", "rms_mean",
    ]

    # Known min/max ranges for normalisation
    FEATURE_RANGES: dict[str, tuple[float, float]] = {
        "tempo":        (40.0,  220.0),
        "key":          (0.0,   11.0),
        "mode":         (0.0,   1.0),
        "energy":       (0.0,   1.0),
        "valence":      (0.0,   1.0),
        "danceability": (0.0,   1.0),
        "loudness":     (-60.0, 0.0),
        "chroma_mean":  (0.0,   1.0),
        "rms_mean":     (0.0,   1.0),
    }

    def __init__(
        self,
        db_path: str | Path,
        subset_name: str | None = None,
        limit: int | None = None,
        episode_length: int = 10,
        repeat_track_penalty: float = 0.25,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.episode_length = episode_length
        self.repeat_track_penalty = repeat_track_penalty
        self.render_mode    = render_mode

        self.tracks = load_tracks_from_db(db_path, subset_name=subset_name, limit=limit)
        if not self.tracks:
            raise ValueError(
                f"No tracks with feature data found in {db_path}. "
                "Make sure your partner's ingest script has been run first."
            )

        self.n_tracks = len(self.tracks)
        self.features = self._build_feature_matrix()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(len(self.FEATURE_KEYS),),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.array([self.n_tracks, len(TRANSITION_TYPES)], dtype=np.int64)
        )

        self._current_idx: int = 0
        self._step_count:  int = 0
        self._history:     list[int] = []

    def _build_feature_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.n_tracks, len(self.FEATURE_KEYS)), dtype=np.float32)
        for i, track in enumerate(self.tracks):
            for j, key in enumerate(self.FEATURE_KEYS):
                lo, hi = self.FEATURE_RANGES[key]
                raw = float(track.get(key, (lo + hi) / 2))
                matrix[i, j] = float(np.clip((raw - lo) / (hi - lo + 1e-8), 0.0, 1.0))
        return matrix

    def _obs(self) -> np.ndarray:
        return self.features[self._current_idx].copy()

    def _info(self) -> dict:
        t = self.tracks[self._current_idx]
        return {
            "step":         self._step_count,
            "current_idx":  self._current_idx,
            "fma_track_id": t["fma_track_id"],
            "title":        t["title"],
            "artist":       t["artist_name"],
            "genre":        t["genre_top"],
            "history_ids":  [self.tracks[i]["fma_track_id"] for i in self._history],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_idx = int(self.np_random.integers(0, self.n_tracks))
        self._step_count  = 0
        self._history     = [self._current_idx]
        return self._obs(), self._info()

    def _decode_action(self, action) -> tuple[int, int]:
        if isinstance(action, np.ndarray):
            values = action.astype(int).reshape(-1).tolist()
        elif isinstance(action, (list, tuple)):
            values = [int(v) for v in action]
        else:
            raise ValueError(
                "Action must be a 2-item sequence: [next_track_idx, transition_idx]."
            )

        if len(values) != 2:
            raise ValueError(
                "Action must contain exactly two integers: "
                "[next_track_idx, transition_idx]."
            )

        next_idx = int(np.clip(values[0], 0, self.n_tracks - 1))
        transition_idx = int(np.clip(values[1], 0, len(TRANSITION_TYPES) - 1))
        return next_idx, transition_idx

    def step(self, action):
        next_idx, transition_idx = self._decode_action(action)

        curr = self.tracks[self._current_idx]
        nxt  = self.tracks[next_idx]

        h_score      = harmonic_compatibility(curr["key"], curr["mode"], nxt["key"], nxt["mode"])
        b_score      = bpm_smoothness(curr["tempo"], nxt["tempo"])
        energy_delta = nxt["energy"] - curr["energy"]
        bpm_delta    = abs(nxt["tempo"] - curr["tempo"])
        t_score      = transition_reward(transition_idx, bpm_delta, energy_delta)
        energy_flow  = float(np.clip(energy_delta * 2.0, -1.0, 1.0)) * 0.5 + 0.5

        base_reward = h_score * 0.4 + b_score * 0.3 + t_score * 0.2 + energy_flow * 0.1
        repeat_penalty = self.repeat_track_penalty if next_idx == self._current_idx else 0.0
        reward = float(np.clip(base_reward - repeat_penalty, -1.0, 1.0))

        self._current_idx = next_idx
        self._step_count += 1
        self._history.append(next_idx)

        terminated = self._step_count >= self.episode_length
        info = self._info()
        info.update({
            "harmonic_score":   h_score,
            "bpm_score":        b_score,
            "transition_score": t_score,
            "energy_flow":      energy_flow,
            "transition_type":  TRANSITION_TYPES[transition_idx],
            "base_reward":      base_reward,
            "repeat_penalty":   repeat_penalty,
            "repeated_track":   next_idx == self._history[-2],
            "reward":           reward,
        })

        if self.render_mode == "human":
            self.render()

        return self._obs(), float(reward), terminated, False, info

    def render(self):
        t = self.tracks[self._current_idx]
        print(
            f"Step {self._step_count:02d} | "
            f"{t['title']} — {t['artist_name']} | "
            f"BPM: {t['tempo']:.1f} | "
            f"Energy: {t['energy']:.2f} | "
            f"Genre: {t['genre_top']}"
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db",       default="fma.db")
    parser.add_argument("--subset",   default=None, help="e.g. 'small'")
    parser.add_argument("--limit",    type=int, default=500)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps",    type=int, default=10)
    parser.add_argument("--repeat-penalty", type=float, default=0.25)
    args = parser.parse_args()

    env = DJEnv(
        db_path=args.db,
        subset_name=args.subset,
        limit=args.limit,
        episode_length=args.steps,
        repeat_track_penalty=args.repeat_penalty,
        render_mode="human",
    )

    print(f"Loaded {env.n_tracks} tracks.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}\n")

    for ep in range(args.episodes):
        obs, info = env.reset()
        ep_reward, done = 0.0, False
        print(f"=== Episode {ep + 1} | Start: {info['title']} ===")
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            ep_reward += reward
            done = terminated or truncated
        print(f"Episode reward: {ep_reward:.3f}\n")

    env.close()
