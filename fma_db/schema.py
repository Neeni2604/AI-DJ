from __future__ import annotations


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_source_id INTEGER NOT NULL,
        metadata_dir TEXT NOT NULL,
        audio_dir TEXT,
        subset_name TEXT,
        started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at TEXT,
        status TEXT NOT NULL,
        track_count INTEGER NOT NULL DEFAULT 0,
        genre_count INTEGER NOT NULL DEFAULT 0,
        notes TEXT,
        FOREIGN KEY (data_source_id) REFERENCES data_sources(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fma_genres (
        genre_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        parent_id INTEGER,
        top_level_page TEXT,
        handle TEXT,
        color TEXT,
        raw_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fma_tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fma_track_id INTEGER NOT NULL UNIQUE,
        subset_name TEXT,
        title TEXT,
        url TEXT,
        date_created TEXT,
        date_recorded TEXT,
        duration_seconds REAL,
        genre_top TEXT,
        license TEXT,
        genres_json TEXT,
        genres_all_json TEXT,
        tags_json TEXT,
        track_interest INTEGER,
        track_listens INTEGER,
        track_favorites INTEGER,
        track_number INTEGER,
        album_title TEXT,
        album_type TEXT,
        album_information TEXT,
        artist_name TEXT,
        artist_location TEXT,
        artist_bio TEXT,
        audio_path TEXT,
        raw_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fma_track_genres (
        fma_track_id INTEGER NOT NULL,
        genre_id INTEGER NOT NULL,
        relation_type TEXT NOT NULL DEFAULT 'direct',
        PRIMARY KEY (fma_track_id, genre_id, relation_type),
        FOREIGN KEY (fma_track_id) REFERENCES fma_tracks(fma_track_id),
        FOREIGN KEY (genre_id) REFERENCES fma_genres(genre_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fma_track_features (
        fma_track_id INTEGER PRIMARY KEY,
        features_json TEXT,
        echonest_json TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (fma_track_id) REFERENCES fma_tracks(fma_track_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fma_tracks_subset ON fma_tracks(subset_name)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fma_tracks_genre_top ON fma_tracks(genre_top)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fma_tracks_audio_path ON fma_tracks(audio_path)
    """,
]