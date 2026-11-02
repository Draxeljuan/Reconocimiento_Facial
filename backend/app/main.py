"""Controlador principal FastAPI para el Sistema de Reconocimiento Facial y Recuperación de Datos."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, File, Form, status, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.ai_engine import ImagenInvalidaError, MotorReconocimientoIA
from app.init_db import init_database
from app.repository import RepositorioBaseDatos
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    InferenciaResponse,
    RegistroUsuarioResponse,
    UsuarioCreateData,
    UsuarioPerfilResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Inicializa y verifica el esquema SQLite al arrancar el contenedor."""
    init_database()
    yield


app = FastAPI(
    title="API de Reconocimiento Facial y Recuperación de Datos",
    description=(
        "Microservicio de biometría facial 128D (dlib/ResNet) y consulta de perfiles en LAN. "
        "Procesamiento estrictamente transitorio en RAM (RNF03)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_motor_ia = MotorReconocimientoIA()
_repositorio = RepositorioBaseDatos()


def get_motor_ia() -> MotorReconocimientoIA:
    """Proveedor de dependencia del motor de inferencia IA."""
    return _motor_ia


def get_repositorio() -> RepositorioBaseDatos:
    """Proveedor de dependencia del repositorio SQLite."""
    return _repositorio


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Estado de salud del servicio y métricas biométricas",
)
def health_check(
    motor: Annotated[MotorReconocimientoIA, Depends(get_motor_ia)],
    repo: Annotated[RepositorioBaseDatos, Depends(get_repositorio)],
) -> HealthResponse:
    conteos = repo.contar_registros()
    return HealthResponse(
        status="ok",
        umbral_tolerancia=motor.umbral_tolerancia,
        usuarios_registrados=conteos["usuarios_registrados"],
        vectores_biometricos=conteos["vectores_biometricos"],
    )


@app.get(
    "/usuarios",
    response_model=list[UsuarioPerfilResponse],
    tags=["Usuarios"],
    summary="Listar perfiles de usuarios enrolados (sin exponer biometría)",
)
def listar_usuarios(
    repo: Annotated[RepositorioBaseDatos, Depends(get_repositorio)],
) -> list[UsuarioPerfilResponse]:
    registros = repo.listar_usuarios()
    return [UsuarioPerfilResponse(**reg) for reg in registros]


def _procesar_enrolamiento_en_ram(
    perfil: UsuarioCreateData,
    payloads_imagenes: list[bytes],
    motor: MotorReconocimientoIA,
    repo: RepositorioBaseDatos,
) -> tuple[dict, int, int]:
    """Extrae embeddings 128D en RAM de cada imagen base y persiste la relación 1:N."""
    vectores_extraidos = []
    descartadas = 0

    for img_bytes in payloads_imagenes:
        try:
            embeddings = motor.extraer_embeddings(img_bytes)
        except ImagenInvalidaError:
            descartadas += 1
            continue

        if embeddings:
            # Tomamos el rostro principal (mayor área en el encuadre) de cada fotografía base
            vectores_extraidos.append(embeddings[0])
        else:
            descartadas += 1

    if not vectores_extraidos:
        return {}, 0, descartadas

    usuario_guardado = repo.guardar_nuevo_usuario(perfil, vectores_extraidos)
    return usuario_guardado, len(vectores_extraidos), descartadas


@app.post(
    "/registro",
    response_model=RegistroUsuarioResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Ningún rostro válido detectado en las imágenes"},
    },
    tags=["Biometría"],
    summary="Enrolar un nuevo usuario con una o múltiples fotografías base (relación 1 a N)",
)
@app.post(
    "/usuarios",
    response_model=RegistroUsuarioResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def registrar_usuario(
    nombres: Annotated[str, Form(description="Nombres completos del usuario")],
    imagenes: Annotated[list[UploadFile], File(description="Una o más imágenes faciales base")],
    motor: Annotated[MotorReconocimientoIA, Depends(get_motor_ia)],
    repo: Annotated[RepositorioBaseDatos, Depends(get_repositorio)],
    edad: Annotated[int | None, Form(description="Edad en años")] = None,
    carrera: Annotated[str | None, Form(description="Carrera o programa académico")] = None,
    gustos_musicales: Annotated[str | None, Form(description="Gustos musicales")] = None,
) -> RegistroUsuarioResponse | JSONResponse:
    nombres_limpio = nombres.strip()
    if not nombres_limpio:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "El campo 'nombres' no puede estar vacío."},
        )

    if not imagenes:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Debe adjuntar al menos una imagen facial para el registro."},
        )

    perfil_input = UsuarioCreateData(
        nombres=nombres_limpio,
        edad=edad,
        carrera=carrera,
        gustos_musicales=gustos_musicales,
    )

    # Lectura estrictamente en memoria RAM y cierre inmediato de descriptores
    payloads_ram: list[bytes] = []
    try:
        for upload in imagenes:
            contenido = await upload.read()
            await upload.close()
            payloads_ram.append(contenido)

        total_recibidas = len(payloads_ram)
        usuario_dict, registrados, descartadas = await run_in_threadpool(
            _procesar_enrolamiento_en_ram,
            perfil_input,
            payloads_ram,
            motor,
            repo,
        )
    finally:
        # Liberación explícita de los buffers de imagen en RAM (RNF03)
        payloads_ram.clear()

    if registrados == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "No se detectó ningún rostro válido en las imágenes proporcionadas.",
                "detalle": f"Imágenes analizadas: {total_recibidas}, descartadas: {descartadas}.",
            },
        )

    logger.info(
        "Nuevo usuario registrado id=%s ('%s') con %d vector(es) biométrico(s).",
        usuario_dict["id"],
        usuario_dict["nombres"],
        registrados,
    )
    return RegistroUsuarioResponse(
        mensaje="Usuario registrado exitosamente",
        usuario=UsuarioPerfilResponse(**usuario_dict),
        vectores_registrados=registrados,
        imagenes_procesadas=total_recibidas,
        imagenes_descartadas=descartadas,
    )


def _ejecutar_inferencia_en_ram(
    fotograma_bytes: bytes,
    motor: MotorReconocimientoIA,
    repo: RepositorioBaseDatos,
) -> tuple[dict | None, float | None, str | None]:
    """Ejecuta extracción 128D y distancia euclidiana vectorizada en RAM."""
    embeddings = motor.extraer_embeddings(fotograma_bytes)
    if not embeddings:
        return None, None, "No se detectó ningún rostro en el fotograma capturado."

    vector_consulta = embeddings[0]
    usuario_ids, matriz_conocidos = repo.obtener_todos_embeddings()

    coincidencia_id, distancia_min = motor.buscar_mejor_coincidencia(
        vector_consulta=vector_consulta,
        usuario_ids=usuario_ids,
        matriz_conocidos=matriz_conocidos,
    )
    if coincidencia_id is None or distancia_min is None:
        return None, distancia_min, "Ningún perfil registrado supera el umbral de confianza."

    perfil = repo.obtener_perfil_usuario(coincidencia_id)
    if perfil is None:
        return None, distancia_min, "Perfil asociado no encontrado."

    return perfil, distancia_min, None


@app.post(
    "/inferencia",
    response_model=InferenciaResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Archivo de imagen inválido o corrupto"},
        404: {"model": ErrorResponse, "description": "Rostro no reconocido o fuera del umbral <= 0.6"},
    },
    tags=["Biometría"],
    summary="Autenticar rostro mediante fotograma transitorio y recuperar datos de perfil",
)
async def inferencia_facial(
    imagen: Annotated[UploadFile, File(description="Fotograma capturado desde WebRTC (JPEG/PNG/WebP)")],
    motor: Annotated[MotorReconocimientoIA, Depends(get_motor_ia)],
    repo: Annotated[RepositorioBaseDatos, Depends(get_repositorio)],
) -> InferenciaResponse | JSONResponse:
    fotograma_bytes: bytes | None = None
    try:
        fotograma_bytes = await imagen.read()
        await imagen.close()

        perfil, distancia_min, motivo = await run_in_threadpool(
            _ejecutar_inferencia_en_ram,
            fotograma_bytes,
            motor,
            repo,
        )
    except ImagenInvalidaError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Imagen inválida", "detalle": str(exc)},
        )
    finally:
        # Garantía de no persistencia y descarte inmediato de memoria (RNF03)
        del fotograma_bytes

    if perfil is None or distancia_min is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Rostro no reconocido",
                "detalle": motivo,
            },
        )

    confianza = round(max(0.0, min(1.0, 1.0 - distancia_min)), 4)
    return InferenciaResponse(
        **perfil,
        distancia=round(distancia_min, 4),
        confianza=confianza,
    )
