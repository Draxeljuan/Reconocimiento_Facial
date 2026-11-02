# Sistema de Reconocimiento Facial y Recuperación de Datos (LAN)

Plataforma contenerizada basada en microservicios para autenticación biométrica facial en tiempo real (embeddings ResNet de 128 dimensiones) y recuperación de perfiles biográficos en red local (LAN) sobre **HTTPS + WebRTC**.

## Arquitectura de Servicios

| Servicio | Contenedor | Tecnología | Rol |
|---|---|---|---|
| **Frontend + Proxy** | `rf_frontend` | Nginx 1.27 + OpenSSL + HTML5/CSS3/Vanilla JS | Sirve la SPA WebRTC sobre HTTPS (`443`), redirige HTTP (`80` -> `443`) y actúa como Reverse-Proxy hacia `/api/*`. |
| **Backend IA** | `rf_backend` | Python 3.11 + FastAPI + OpenCV + `dlib`/`face_recognition` | Extracción de embeddings 128D estrictamente en RAM (**RNF03**) y cálculo vectorizado de distancia euclidiana (`<= 0.6`). |
| **Persistencia** | Volumen `./data` | SQLite (`database.db` en modo WAL + FK `1:N`) | Tablas `Usuarios` (1) y `Biometria` (N) persistidas en `/app/data/database.db`. |
| **Seguridad LAN** | Volumen `./certs` | Certificados X.509 RSA-2048 con SAN | Generados automáticamente al iniciar `rf_frontend` (`server.crt` y `server.key`) para habilitar `getUserMedia` en móviles (**RNF02**). |

## Despliegue y Ejecución (100% Docker)

1. **(Opcional) Configurar IP LAN en `.env`:**
   Si deseas que el certificado autofirmado incluya explícitamente la IP local de tu equipo anfitrión en el campo *Subject Alternative Name (SAN)*, asigna `LAN_IP=192.168.X.X` en `.env` (si ya existen certificados previos en `certs/`, elimínalos para regenerarlos).

2. **Construir e iniciar todos los servicios:**
   ```bash
   docker compose up -d --build
   ```

3. **Acceso desde PC o Dispositivos Móviles en la LAN:**
   - En el mismo equipo: `https://localhost`
   - Desde smartphones u otros equipos en la misma red Wi-Fi/LAN: `https://<IP_LOCAL_DEL_HOST>`
   - Al tratarse de un certificado autofirmado local, acepta la advertencia inicial del navegador (*"Configuración avanzada -> Continuar"*). Una vez aceptado, tanto la cámara WebRTC como todas las peticiones hacia `/api/inferencia` y `/api/registro` operarán cifradas bajo el mismo origen.

## Ejecución de Pruebas Integrales (en Contenedor)

```bash
docker compose exec backend python -m unittest discover -s tests -v
```

## Endpoints Principales (vía Reverse-Proxy HTTPS)

- `GET /api/health` — Estado operativo, umbral configurado y conteo de perfiles/vectores.
- `GET /api/usuarios` — Listado de perfiles registrados (sin exponer vectores biométricos).
- `POST /api/registro` — Enrolamiento `multipart/form-data` (`nombres`, `edad`, `carrera`, `gustos_musicales`, `imagenes[]`).
- `POST /api/inferencia` — Autenticación biométrica transitoria en RAM (`imagen`). Retorna `HTTP 200` con el perfil si `distancia <= 0.6`, o `HTTP 404` (`{"error": "Rostro no reconocido"}`).
