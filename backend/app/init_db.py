"""Script idempotente de inicialización y migración del esquema relacional SQLite.

Implementa estrictamente el modelo DBML documentado en 'Propuesta Inicial.md':
- Tabla Usuarios (1)
- Tabla Biometria (N) vinculada mediante Biometria.usuario_id -> Usuarios.id
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.database import db_session, get_database_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("init_db")

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS Usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombres VARCHAR NOT NULL,
    edad INTEGER,
    carrera VARCHAR,
    gustos_musicales VARCHAR,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Biometria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario_id INTEGER NOT NULL,
    vector_embedding TEXT NOT NULL,
    FOREIGN KEY (usuario_id) REFERENCES Usuarios(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_biometria_usuario_id
    ON Biometria(usuario_id);
"""


def init_database(db_path: Path | None = None) -> Path:
    """Inicializa el archivo SQLite y aplica el esquema relacional de forma idempotente."""
    target_path = db_path or get_database_path()
    logger.info("Inicializando base de datos SQLite en: %s", target_path)

    with db_session(target_path) as conn:
        # executescript realiza un COMMIT implícito previo; reactivamos foreign_keys tras el DDL
        conn.executescript(SCHEMA_DDL)
        conn.execute("PRAGMA foreign_keys = ON;")

        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
            ).fetchall()
        ]
        fk_enabled = conn.execute("PRAGMA foreign_keys;").fetchone()[0]

    logger.info(
        "Esquema verificado correctamente. Tablas activas: %s | PRAGMA foreign_keys=%s",
        tables,
        fk_enabled,
    )
    return target_path


if __name__ == "__main__":
    init_database()
