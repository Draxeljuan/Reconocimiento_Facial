"""Esquemas Pydantic de validación y respuesta para la API de Reconocimiento Facial."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UsuarioCreateData(BaseModel):
    """Datos alfanuméricos de entrada para registrar un nuevo perfil."""

    nombres: str = Field(..., min_length=1, max_length=150, description="Nombres completos del usuario")
    edad: int | None = Field(default=None, ge=0, le=130, description="Edad en años")
    carrera: str | None = Field(default=None, max_length=150, description="Programa académico o carrera")
    gustos_musicales: str | None = Field(default=None, max_length=255, description="Géneros o artistas preferidos")


class UsuarioPerfilResponse(BaseModel):
    """Datos públicos del perfil almacenado en la tabla Usuarios."""

    id: int
    nombres: str
    edad: int | None = None
    carrera: str | None = None
    gustos_musicales: str | None = None
    fecha_registro: str | None = None


class InferenciaResponse(UsuarioPerfilResponse):
    """Respuesta exitosa del endpoint POST /inferencia tras superar el umbral biométrico."""

    distancia: float = Field(..., description="Distancia euclidiana mínima frente a la base biométrica")
    confianza: float = Field(..., description="Índice normalizado de similitud estimado (0.0 a 1.0)")


class RegistroUsuarioResponse(BaseModel):
    """Respuesta exitosa del endpoint de enrolamiento biométrico."""

    mensaje: str
    usuario: UsuarioPerfilResponse
    vectores_registrados: int
    imagenes_procesadas: int
    imagenes_descartadas: int


class HealthResponse(BaseModel):
    """Estado operativo del servicio backend y métricas de persistencia."""

    status: str
    umbral_tolerancia: float
    usuarios_registrados: int
    vectores_biometricos: int


class ErrorResponse(BaseModel):
    """Estructura estándar de error JSON (ej. 404 Rostro no reconocido)."""

    error: str
    detalle: str | None = None
