from __future__ import annotations

import ast
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from fma_db.db import connect, initialize_database


def _flatten_headers(header_rows: list[list[str]]) -> list[str]:
    width = max(len(row) for row in header_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in header_rows]
    columns: list[str] = []
    for column_index in range(width):
        parts: list[str] = []
        for row in normalized_rows:
            cell = row[column_index].strip()
            if cell and cell not in parts:
                parts.append(cell)
        columns.append("__".join(parts) if parts else f"column_{column_index}")
    return columns


def _read_multiheader_csv(csv_path: Path, header_rows: int) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = [next(reader) for _ in range(header_rows)]
        columns = _flatten_headers(header)
        rows: list[dict[str, str]] = []
        for row in reader:
            padded_row = row + [""] * (len(columns) - len(row))
            rows.append(dict(zip(columns, padded_row)))
    return rows


def _read_simple_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _clean_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _parse_literal_list(value: str | None) -> list[Any]:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return []
    try:
        parsed = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_int(value: str | None) -> int | None:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _audio_path(audio_dir: Path | None, track_id: int) -> str | None:
    if audio_dir is None:
        return None
    rel_path = Path(f"{track_id:06d}"[:3]) / f"{track_id:06d}.mp3"
    candidate = audio_dir / rel_path
    return str(candidate) if candidate.exists() else None


def _fma_source_id(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT id FROM data_sources WHERE name = 'fma'"
    ).fetchone()
    if row is None:
        raise RuntimeError("Missing FMA data source row")
    return int(row["id"])


def _create_import_run(
    connection: sqlite3.Connection,
    *,
    metadata_dir: Path,
    audio_dir: Path | None,
    subset_name: str | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO import_runs(data_source_id, metadata_dir, audio_dir, subset_name, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            _fma_source_id(connection),
            str(metadata_dir),
            str(audio_dir) if audio_dir else None,
            subset_name,
            "running",
        ),
    )
    return int(cursor.lastrowid)


def _finish_import_run(
    connection: sqlite3.Connection,
    import_run_id: int,
    *,
    status: str,
    track_count: int,
    genre_count: int,
    notes: str | None = None,
) -> None:
    connection.execute(
        """
        UPDATE import_runs
        SET finished_at = CURRENT_TIMESTAMP,
            status = ?,
            track_count = ?,
            genre_count = ?,
            notes = ?
        WHERE id = ?
        """,
        (status, track_count, genre_count, notes, import_run_id),
    )


def _import_genres(connection: sqlite3.Connection, genres_path: Path) -> int:
    rows = _read_simple_csv(genres_path)
    imported = 0
    for row in rows:
        genre_id = _parse_int(row.get("genre_id") or row.get("#genre_id"))
        title = _clean_scalar(row.get("title"))
        if genre_id is None or title is None:
            continue
        connection.execute(
            """
            INSERT INTO fma_genres(
                genre_id, title, parent_id, top_level_page, handle, color, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(genre_id) DO UPDATE SET
                title = excluded.title,
                parent_id = excluded.parent_id,
                top_level_page = excluded.top_level_page,
                handle = excluded.handle,
                color = excluded.color,
                raw_json = excluded.raw_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                genre_id,
                title,
                _parse_int(row.get("parent")),
                _clean_scalar(row.get("top_level_page")),
                _clean_scalar(row.get("handle")),
                _clean_scalar(row.get("color")),
                json.dumps(row),
            ),
        )
        imported += 1
    return imported


def _import_tracks(
    connection: sqlite3.Connection,
    tracks_path: Path,
    *,
    audio_dir: Path | None,
    subset_name: str | None,
    limit: int | None,
) -> int:
    rows = _read_multiheader_csv(tracks_path, header_rows=2)
    imported = 0
    for row in rows:
        track_id = _parse_int(
            row.get("track__id")
            or row.get("track_id")
            or row.get("Unnamed: 0_level_0__Unnamed: 0_level_1")
        )
        if track_id is None:
            continue

        genres = _parse_literal_list(row.get("track__genres"))
        genres_all = _parse_literal_list(row.get("track__genres_all"))
        tags = _parse_literal_list(row.get("track__tags"))

        connection.execute(
            """
            INSERT INTO fma_tracks(
                fma_track_id, subset_name, title, url, date_created, date_recorded,
                duration_seconds, genre_top, license, genres_json, genres_all_json,
                tags_json, track_interest, track_listens, track_favorites, track_number,
                album_title, album_type, album_information, artist_name, artist_location,
                artist_bio, audio_path, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(fma_track_id) DO UPDATE SET
                subset_name = excluded.subset_name,
                title = excluded.title,
                url = excluded.url,
                date_created = excluded.date_created,
                date_recorded = excluded.date_recorded,
                duration_seconds = excluded.duration_seconds,
                genre_top = excluded.genre_top,
                license = excluded.license,
                genres_json = excluded.genres_json,
                genres_all_json = excluded.genres_all_json,
                tags_json = excluded.tags_json,
                track_interest = excluded.track_interest,
                track_listens = excluded.track_listens,
                track_favorites = excluded.track_favorites,
                track_number = excluded.track_number,
                album_title = excluded.album_title,
                album_type = excluded.album_type,
                album_information = excluded.album_information,
                artist_name = excluded.artist_name,
                artist_location = excluded.artist_location,
                artist_bio = excluded.artist_bio,
                audio_path = excluded.audio_path,
                raw_json = excluded.raw_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                track_id,
                subset_name,
                _clean_scalar(row.get("track__title")),
                _clean_scalar(row.get("track__url")),
                _clean_scalar(row.get("track__date_created")),
                _clean_scalar(row.get("track__date_recorded")),
                _parse_float(row.get("track__duration")),
                _clean_scalar(row.get("track__genre_top")),
                _clean_scalar(row.get("track__license")),
                json.dumps(genres),
                json.dumps(genres_all),
                json.dumps(tags),
                _parse_int(row.get("track__interest")),
                _parse_int(row.get("track__listens")),
                _parse_int(row.get("track__favorites")),
                _parse_int(row.get("track__number")),
                _clean_scalar(row.get("album__title")),
                _clean_scalar(row.get("album__type")),
                _clean_scalar(row.get("album__information")),
                _clean_scalar(row.get("artist__name")),
                _clean_scalar(row.get("artist__location")),
                _clean_scalar(row.get("artist__bio")),
                _audio_path(audio_dir, track_id),
                json.dumps(row),
            ),
        )

        connection.execute(
            "DELETE FROM fma_track_genres WHERE fma_track_id = ?",
            (track_id,),
        )
        for genre_id in genres:
            if isinstance(genre_id, int):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fma_track_genres(fma_track_id, genre_id, relation_type)
                    VALUES (?, ?, 'direct')
                    """,
                    (track_id, genre_id),
                )
        for genre_id in genres_all:
            if isinstance(genre_id, int):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fma_track_genres(fma_track_id, genre_id, relation_type)
                    VALUES (?, ?, 'all')
                    """,
                    (track_id, genre_id),
                )

        imported += 1
        if limit is not None and imported >= limit:
            break
    return imported


def _feature_payload(rows: list[dict[str, str]], track_id_key: str) -> dict[int, dict[str, Any]]:
    payload: dict[int, dict[str, Any]] = {}
    for row in rows:
        track_id = _parse_int(row.get(track_id_key))
        if track_id is None:
            first_key = next(iter(row.keys()), None)
            track_id = _parse_int(row.get(first_key) if first_key else None)
        if track_id is None:
            continue
        payload[track_id] = row
    return payload


def _import_features(
    connection: sqlite3.Connection,
    features_path: Path | None,
    echonest_path: Path | None,
) -> None:
    features_rows = (
        _feature_payload(_read_multiheader_csv(features_path, header_rows=3), "track_id")
        if features_path and features_path.exists()
        else {}
    )
    echonest_rows = (
        _feature_payload(_read_multiheader_csv(echonest_path, header_rows=3), "track_id")
        if echonest_path and echonest_path.exists()
        else {}
    )

    all_track_ids = set(features_rows) | set(echonest_rows)
    for track_id in all_track_ids:
        connection.execute(
            """
            INSERT INTO fma_track_features(fma_track_id, features_json, echonest_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(fma_track_id) DO UPDATE SET
                features_json = excluded.features_json,
                echonest_json = excluded.echonest_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                track_id,
                json.dumps(features_rows.get(track_id)) if track_id in features_rows else None,
                json.dumps(echonest_rows.get(track_id)) if track_id in echonest_rows else None,
            ),
        )


def import_fma_dataset(
    db_path: str | Path,
    *,
    metadata_dir: str | Path,
    audio_dir: str | Path | None = None,
    subset_name: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:

    """Import FMA metadata and optional audio paths into SQLite."""

    initialize_database(db_path)

    metadata_path = Path(metadata_dir)
    audio_path: Path | None = Path(audio_dir) if audio_dir is not None else None
    tracks_csv: Path = metadata_path / "tracks.csv"
    genres_csv: Path = metadata_path / "genres.csv"
    features_csv: Path = metadata_path / "features.csv"
    echonest_csv: Path = metadata_path / "echonest.csv"

    if not tracks_csv.exists():
        raise FileNotFoundError(f"Missing tracks.csv in {metadata_path}")
    if not genres_csv.exists():
        raise FileNotFoundError(f"Missing genres.csv in {metadata_path}")

    with connect(db_path) as connection:
        with connection:
            import_run_id = _create_import_run(
                connection,
                metadata_dir=metadata_path,
                audio_dir=audio_path,
                subset_name=subset_name,
            )
            try:
                genre_count = _import_genres(connection, genres_csv)
                track_count = _import_tracks(
                    connection,
                    tracks_csv,
                    audio_dir=audio_path,
                    subset_name=subset_name,
                    limit=limit,
                )
                _import_features(connection, features_csv, echonest_csv)
                _finish_import_run(
                    connection,
                    import_run_id,
                    status="succeeded",
                    track_count=track_count,
                    genre_count=genre_count,
                )
                return {"tracks": track_count, "genres": genre_count}
            except Exception as exc:
                _finish_import_run(
                    connection,
                    import_run_id,
                    status="failed",
                    track_count=0,
                    genre_count=0,
                    notes=str(exc),
                )
                raise
