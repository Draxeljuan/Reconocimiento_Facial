# ==============================================================================
# Etapa 1: Builder (Compilación C++ de dlib y construcción del entorno virtual)
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=4

# Capa 1: Dependencias del sistema y toolchain C++ (altamente cacheable)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual aislado para transportarlo limpio a la etapa runtime
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Capa 2: Compilación e instalación aislada del motor IA pesado (dlib + OpenCV + face_recognition)
# Se ejecuta ANTES de copiar requirements.txt para que ajustes en dependencias web o código
# jamás invaliden la caché de compilación de dlib.
RUN pip install --no-cache-dir --upgrade pip "setuptools>=69.0.0,<81.0.0" "wheel>=0.43.0" && \
    pip install --no-cache-dir \
        "numpy==1.26.4" \
        "dlib==19.24.6" \
        "face-recognition-models==0.3.0" \
        "face_recognition==1.3.0" \
        "opencv-python-headless==4.10.0.84"

# Capa 3: Instalación del resto de dependencias declaradas en requirements.txt (FastAPI, Uvicorn, etc.)
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Etapa 2: Runtime (Imagen ligera de producción sin compiladores C++)
# ==============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DATABASE_PATH=/app/data/database.db \
    FACE_MATCH_THRESHOLD=0.6

# Instalar únicamente librerías dinámicas compartidas requeridas en tiempo de ejecución
RUN apt-get update && \
    GLIB_PKG=$(apt-cache show libglib2.0-0t64 >/dev/null 2>&1 && echo libglib2.0-0t64 || echo libglib2.0-0) && \
    apt-get install -y --no-install-recommends \
        libopenblas0 \
        liblapack3 \
        libgomp1 \
        $GLIB_PKG \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar el entorno virtual ya compilado desde la etapa builder
COPY --from=builder /opt/venv /opt/venv

# Copiar código fuente y suite de pruebas del servicio backend
COPY ./app /app/app
COPY ./tests /app/tests

# Crear directorio de persistencia y usuario sin privilegios (UID/GID 1000)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/sh appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
