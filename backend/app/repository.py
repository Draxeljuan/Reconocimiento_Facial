"""Capa de acceso a datos (RepositorioBaseDatos) sobre SQLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.database import db_session
from app.schemas import UsuarioCreateData

EMBEDDING_DIM = 128


class RepositorioBaseDatos:
    """Gestiona la persistencia relacional de Usuarios (1) y Biometria (N)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def guardar_nuevo_usuario(
        self,
        perfil: UsuarioCreateData,
        vectores: list[np.ndarray | list[float]],
    ) -> dict[str, Any]:
        """Persiste de forma atómica un perfil de usuario y sus N embeddings de 128 dimensiones."""
        if not vectores:
            raise ValueError("Se requiere al menos un vector biométrico para registrar al usuario.")

        vectores_serializados: list[str] = []
        for vec in vectores:
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
            if arr.shape[0] != EMBEDDING_DIM:
                raise ValueError(
                    f"Dimensión de embedding inválida: se esperaban {EMBEDDING_DIM}, recibido {arr.shape[0]}."
                )
            vectores_serializados.append(json.dumps(arr.tolist()))

        with db_session(self._db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO Usuarios (nombres, edad, carrera, gustos_musicales)
                VALUES (?, ?, ?, ?);
                """,
                (
                    perfil.nombres.strip(),
                    perfil.edad,
                    perfil.carrera.strip() if perfil.carrera else None,
                    perfil.gustos_musicales.strip() if perfil.gustos_musicales else None,
                ),
            )
            usuario_id = int(cursor.lastrowid)

            conn.executemany(
                """
                INSERT INTO Biometria (usuario_id, vector_embedding)
                VALUES (?, ?);
                """,
                [(usuario_id, vec_json) for vec_json in vectores_serializados],
            )

            row = conn.execute(
                """
                SELECT id, nombres, edad, carrera, gustos_musicales, fecha_registro
                FROM Usuarios
                WHERE id = ?;
                """,
                (usuario_id,),
            ).fetchone()

        return dict(row)

    def obtener_todos_embeddings(self) -> tuple[list[int], np.ndarray]:
        """Recupera todos los vectores biométricos como una matriz NumPy (M, 128) para cálculo vectorizado."""
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT usuario_id, vector_embedding
                FROM Biometria
                ORDER BY id ASC;
                """
            ).fetchall()

        if not rows:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float64)

        usuario_ids: list[int] = []
        vectores: list[list[float]] = []
        for row in rows:
            vec = json.loads(row["vector_embedding"])
            if len(vec) == EMBEDDING_DIM:
                usuario_ids.append(int(row["usuario_id"]))
                vectores.append(vec)

        if not vectores:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float64)

        matriz = np.asarray(vectores, dtype=np.float64)
        return usuario_ids, matriz

    def obtener_perfil_usuario(self, usuario_id: int) -> dict[str, Any] | None:
        """Consulta los datos biográficos de un usuario por su identificador primario."""
        with db_session(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT id, nombres, edad, carrera, gustos_musicales, fecha_registro
                FROM Usuarios
                WHERE id = ?;
                """,
                (usuario_id,),
            ).fetchone()

        return dict(row) if row is not None else None

    def listar_usuarios(self) -> list[dict[str, Any]]:
        """Retorna el listado de perfiles registrados (sin exponer vectores biométricos)."""
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, nombres, edad, carrera, gustos_musicales, fecha_registro
                FROM Usuarios
                ORDER BY id DESC;
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def contar_registros(self) -> dict[str, int]:
        """Obtiene conteos totales de Usuarios y vectores en Biometria."""
        with db_session(self._db_path) as conn:
            total_usuarios = int(conn.execute("SELECT COUNT(*) FROM Usuarios;").fetchone()[0])
            total_vectores = int(conn.execute("SELECT COUNT(*) FROM Biometria;").fetchone()[0])

        return {
            "usuarios_registrados": total_usuarios,
            "vectores_biometricos": total_vectores,
        }
