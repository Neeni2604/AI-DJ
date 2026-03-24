from __future__ import annotations

import sqlite3
from pathlib import Path

from fma_db.schema import SCHEMA_STATEMENTS


SCHEMA_VERSION = "0001_initial"


def connect(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a SQLite connection with foreign keys enabled.
    """
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_database(db_path: str | Path) -> Path:
    """
    Create the SQLite database and apply the FMA schema.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with connect(path) as connection:
        with connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO data_sources(name, description)
                VALUES (?, ?)
                """,
                (
                    "fma",
                    "Free Music Archive local metadata and audio dataset",
                ),
            )

    return path


def list_tables(db_path: str | Path) -> list[str]:
    """
    Return user-defined table names in the database.
    """
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return [row["name"] for row in rows]
