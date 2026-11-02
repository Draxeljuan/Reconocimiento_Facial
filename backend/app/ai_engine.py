"""Motor de Inteligencia Artificial para detección facial y extracción de embeddings 128D.

Cumple estrictamente RNF03 (Privacidad por diseño): todo procesamiento ocurre de forma
transitoria en memoria RAM mediante buffers NumPy/OpenCV, sin persistencia en disco.
"""

from __future__ import annotations

import os

import cv2
import face_recognition
import numpy as np


class ImagenInvalidaError(ValueError):
    """Excepción lanzada cuando los bytes recibidos no corresponden a una imagen decodificable."""


class MotorReconocimientoIA:
    """Encapsula la detección de rostros, proyección ResNet-128D y distancia euclidiana vectorizada."""

    def __init__(
        self,
        umbral_tolerancia: float | None = None,
        max_dimension: int = 1280,
    ) -> None:
        env_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
        self.umbral_tolerancia: float = (
            umbral_tolerancia if umbral_tolerancia is not None else env_threshold
        )
        self.max_dimension: int = max_dimension

    def _decodificar_imagen_en_ram(self, imagen_bytes: bytes) -> np.ndarray:
        """Decodifica bytes crudos en un arreglo RGB contiguo en RAM sin tocar el disco."""
        if not imagen_bytes:
            raise ImagenInvalidaError("El contenido de la imagen está vacío.")

        buffer_np = np.frombuffer(imagen_bytes, dtype=np.uint8)
        bgr_img = cv2.imdecode(buffer_np, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ImagenInvalidaError("No fue posible decodificar el archivo como una imagen válida.")

        alto, ancho = bgr_img.shape[:2]
        lado_mayor = max(alto, ancho)
        if lado_mayor > self.max_dimension:
            escala = self.max_dimension / float(lado_mayor)
            nuevo_ancho = max(1, int(round(ancho * escala)))
            nuevo_alto = max(1, int(round(alto * escala)))
            bgr_img = cv2.resize(bgr_img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

        # dlib requiere un arreglo uint8 RGB contiguo en memoria C
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb_img, dtype=np.uint8)

    @staticmethod
    def _ordenar_por_area_desc(
        ubicaciones: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Ordena las cajas delimitadoras (top, right, bottom, left) priorizando el rostro principal."""
        return sorted(
            ubicaciones,
            key=lambda box: max(0, box[2] - box[0]) * max(0, box[1] - box[3]),
            reverse=True,
        )

    def detectar_rostros(self, imagen_bytes: bytes) -> list[tuple[int, int, int, int]]:
        """Detecta rostros en la imagen y devuelve sus BoundingBoxes (top, right, bottom, left)."""
        rgb_img = self._decodificar_imagen_en_ram(imagen_bytes)
        ubicaciones = face_recognition.face_locations(rgb_img, model="hog")
        return self._ordenar_por_area_desc(ubicaciones)

    def extraer_embeddings(self, imagen_bytes: bytes) -> list[np.ndarray]:
        """Extrae vectores numéricos de 128 dimensiones para los rostros detectados en RAM."""
        rgb_img = self._decodificar_imagen_en_ram(imagen_bytes)
        ubicaciones = self._ordenar_por_area_desc(
            face_recognition.face_locations(rgb_img, model="hog")
        )
        if not ubicaciones:
            return []

        encodings = face_recognition.face_encodings(
            rgb_img,
            known_face_locations=ubicaciones,
            num_jitters=1,
        )
        return [np.asarray(enc, dtype=np.float64) for enc in encodings]

    @staticmethod
    def calcular_distancia(matriz_conocidos: np.ndarray, vector_consulta: np.ndarray) -> np.ndarray:
        """Calcula de forma vectorizada la distancia euclidiana L2 frente a M vectores conocidos."""
        if matriz_conocidos.size == 0:
            return np.empty((0,), dtype=np.float64)
        consulta = np.asarray(vector_consulta, dtype=np.float64).reshape(1, -1)
        return np.linalg.norm(matriz_conocidos - consulta, axis=1)

    def buscar_mejor_coincidencia(
        self,
        vector_consulta: np.ndarray,
        usuario_ids: list[int],
        matriz_conocidos: np.ndarray,
    ) -> tuple[int | None, float | None]:
        """Evalúa la distancia mínima contra los vectores registrados aplicando umbral <= 0.6."""
        if matriz_conocidos.size == 0 or not usuario_ids:
            return None, None

        distancias = self.calcular_distancia(matriz_conocidos, vector_consulta)
        indice_min = int(np.argmin(distancias))
        distancia_min = float(distancias[indice_min])

        if distancia_min <= self.umbral_tolerancia:
            return usuario_ids[indice_min], distancia_min

        return None, distancia_min
