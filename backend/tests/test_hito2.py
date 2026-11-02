"""Pruebas integrales del Hito 2: Motor IA (dlib/ResNet-128D), Repositorio 1:N y Endpoints FastAPI."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import dlib
import face_recognition
import numpy as np
from fastapi.testclient import TestClient

from app.ai_engine import MotorReconocimientoIA
from app.init_db import init_database
from app.main import app, get_motor_ia, get_repositorio
from app.repository import RepositorioBaseDatos


def _generar_jpeg_en_ram(ancho: int = 320, alto: int = 320, semilla: int = 42) -> bytes:
    """Genera una imagen JPEG sintética en memoria RAM para pruebas sin tocar disco."""
    rng = np.random.default_rng(semilla)
    img = rng.integers(40, 210, size=(alto, ancho, 3), dtype=np.uint8)
    exito, encoded = cv2.imencode(".jpg", img)
    assert exito, "Fallo al codificar JPEG en RAM"
    return encoded.tobytes()


class TestMotorIAyEndpointsHito2(unittest.TestCase):
    """Valida compilación C++, inferencia ResNet-128D, cálculo vectorizado y contratos HTTP."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_database.db"
        init_database(self.db_path)

        self.repo = RepositorioBaseDatos(db_path=self.db_path)
        self.motor = MotorReconocimientoIA(umbral_tolerancia=0.6)

        app.dependency_overrides[get_repositorio] = lambda: self.repo
        app.dependency_overrides[get_motor_ia] = lambda: self.motor
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self._tmpdir.cleanup()

    def test_01_bindings_cpp_dlib_resnet128d_y_ram_decode(self) -> None:
        """Verifica que dlib, OpenCV y el modelo ResNet de 128 dimensiones operan en RAM."""
        self.assertTrue(hasattr(dlib, "cnn_face_detection_model_v1"))

        # Verificar reescalado automático en RAM (> 1280px -> 1280px)
        jpeg_grande = _generar_jpeg_en_ram(ancho=1920, alto=1080)
        rgb = self.motor._decodificar_imagen_en_ram(jpeg_grande)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(rgb.flags["C_CONTIGUOUS"])
        self.assertEqual(max(rgb.shape[:2]), 1280)

        # Ejecutar el descriptor ResNet real de face_recognition forzando una caja conocida
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=[(100, 300, 300, 100)],
            num_jitters=1,
        )
        self.assertEqual(len(encodings), 1)
        self.assertEqual(encodings[0].shape, (128,))

    def test_02_distancia_euclidiana_vectorizada_y_umbral_06(self) -> None:
        """Comprueba la exactitud del motor vectorizado frente al umbral <= 0.6."""
        base = np.zeros(128, dtype=np.float64)
        vec_cercano = base.copy()
        vec_cercano[0] = 0.45  # Distancia L2 exacta = 0.45 (<= 0.6 -> Coincidencia)

        vec_lejano = base.copy()
        vec_lejano[0] = 0.75  # Distancia L2 exacta = 0.75 (> 0.6 -> Rechazo)

        matriz = np.vstack([vec_lejano, vec_cercano])
        uid, dist = self.motor.buscar_mejor_coincidencia(base, [10, 20], matriz)
        self.assertEqual(uid, 20)
        self.assertAlmostEqual(dist, 0.45, places=6)

        # Solo vectores fuera de umbral (> 0.6)
        uid_no, dist_no = self.motor.buscar_mejor_coincidencia(base, [10], np.vstack([vec_lejano]))
        self.assertIsNone(uid_no)
        self.assertAlmostEqual(dist_no, 0.75, places=6)

    def test_03_flujo_completo_enrolamiento_1_a_n_e_inferencia_http(self) -> None:
        """Valida POST /registro (1:N), POST /inferencia (200 y 404), latencia <3s y cero disco."""
        vec_usuario_foto1 = np.zeros(128, dtype=np.float64)
        vec_usuario_foto1[0] = 0.10

        vec_usuario_foto2 = np.zeros(128, dtype=np.float64)
        vec_usuario_foto2[0] = 0.20

        img_bytes_1 = _generar_jpeg_en_ram(semilla=1)
        img_bytes_2 = _generar_jpeg_en_ram(semilla=2)

        # 1. Enrolamiento con 2 fotografías para el mismo usuario (relación 1:N)
        with patch.object(
            self.motor,
            "extraer_embeddings",
            side_effect=[[vec_usuario_foto1], [vec_usuario_foto2]],
        ):
            resp_reg = self.client.post(
                "/registro",
                data={
                    "nombres": "Ana María Gómez",
                    "edad": "21",
                    "carrera": "Ingeniería Electrónica",
                    "gustos_musicales": "Synthwave, Rock Alternativo",
                },
                files=[
                    ("imagenes", ("captura1.jpg", img_bytes_1, "image/jpeg")),
                    ("imagenes", ("captura2.jpg", img_bytes_2, "image/jpeg")),
                ],
            )

        self.assertEqual(resp_reg.status_code, 201, resp_reg.text)
        body_reg = resp_reg.json()
        self.assertEqual(body_reg["vectores_registrados"], 2)
        self.assertEqual(body_reg["usuario"]["nombres"], "Ana María Gómez")

        # Verificar en health que hay 1 usuario y 2 vectores biométricos (relación 1:N)
        resp_health = self.client.get("/health")
        self.assertEqual(resp_health.status_code, 200)
        self.assertEqual(resp_health.json()["usuarios_registrados"], 1)
        self.assertEqual(resp_health.json()["vectores_biometricos"], 2)

        # 2. Inferencia exitosa (distancia = 0.35 <= 0.6) -> HTTP 200 y latencia < 3s (RNF01)
        vec_consulta_conocido = np.zeros(128, dtype=np.float64)
        vec_consulta_conocido[0] = 0.55  # Distancia a foto2 (0.20) = 0.35 <= 0.6

        t0 = time.perf_counter()
        with patch.object(self.motor, "extraer_embeddings", return_value=[vec_consulta_conocido]):
            resp_inf_ok = self.client.post(
                "/inferencia",
                files={"imagen": ("frame.jpg", img_bytes_1, "image/jpeg")},
            )
        latencia = time.perf_counter() - t0
        self.assertLess(latencia, 3.0, "La latencia excedió RNF01 (3s)")
        self.assertEqual(resp_inf_ok.status_code, 200, resp_inf_ok.text)
        perfil = resp_inf_ok.json()
        self.assertEqual(perfil["nombres"], "Ana María Gómez")
        self.assertEqual(perfil["carrera"], "Ingeniería Electrónica")
        self.assertAlmostEqual(perfil["distancia"], 0.35, places=4)

        # 3. Inferencia de desconocido (distancia = 0.85 > 0.6) -> HTTP 404
        vec_desconocido = np.zeros(128, dtype=np.float64)
        vec_desconocido[0] = 1.05  # Distancia mínima = 0.85 > 0.6
        with patch.object(self.motor, "extraer_embeddings", return_value=[vec_desconocido]):
            resp_inf_404 = self.client.post(
                "/inferencia",
                files={"imagen": ("desconocido.jpg", img_bytes_1, "image/jpeg")},
            )
        self.assertEqual(resp_inf_404.status_code, 404)
        self.assertEqual(resp_inf_404.json()["error"], "Rostro no reconocido")

        # 4. Inferencia real sobre imagen sin rostro (pipeline HOG completo) -> HTTP 404
        resp_sin_rostro = self.client.post(
            "/inferencia",
            files={"imagen": ("ruido.jpg", img_bytes_1, "image/jpeg")},
        )
        self.assertEqual(resp_sin_rostro.status_code, 404)
        self.assertEqual(resp_sin_rostro.json()["error"], "Rostro no reconocido")

        # 5. Imagen corrupta -> HTTP 400
        resp_corrupta = self.client.post(
            "/inferencia",
            files={"imagen": ("bad.jpg", b"no-es-una-imagen", "image/jpeg")},
        )
        self.assertEqual(resp_corrupta.status_code, 400)

        # 6. Verificar RNF03: ningún archivo .jpg/.png persistido en el directorio temporal
        archivos_img = list(Path(self._tmpdir.name).glob("*.jpg")) + list(
            Path(self._tmpdir.name).glob("*.png")
        )
        self.assertEqual(archivos_img, [], "Violación RNF03: se escribieron imágenes en disco")


if __name__ == "__main__":
    unittest.main(verbosity=2)
