"""Gestión de conexiones y sesiones para el motor embebido SQLite."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_DATABASE_PATH = "/app/data/database.db"


def get_database_path() -> Path:
    """Obtiene la ruta configurada para el archivo SQLite y asegura su directorio padre."""
    raw_path = os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH)
    db_path = Path(raw_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Abre una conexión SQLite con integridad referencial (FK) y modo WAL activos."""
    target_path = db_path or get_database_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # SQLite desactiva las llaves foráneas por defecto en cada nueva conexión;
    # es obligatorio activarlas explícitamente para cumplir la relación 1:N.
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def db_session(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager transaccional que confirma cambios o ejecuta rollback ante errores."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
