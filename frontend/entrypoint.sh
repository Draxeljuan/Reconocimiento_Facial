#!/bin/sh
set -eu

CERT_DIR="/etc/nginx/certs"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"

mkdir -p "${CERT_DIR}"

if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
    echo "[entrypoint] Generando certificado SSL/TLS autofirmado para despliegue LAN..."

    SAN_LIST="DNS:localhost,DNS:frontend,IP:127.0.0.1"
    if [ -n "${LAN_IP:-}" ]; then
        SAN_LIST="${SAN_LIST},IP:${LAN_IP}"
    fi

    openssl req -x509 -nodes -newkey rsa:2048 \
        -days 365 \
        -keyout "${KEY_FILE}" \
        -out "${CERT_FILE}" \
        -subj "/C=CO/ST=Local/L=LAN/O=ReconocimientoFacial/OU=DevOps/CN=${LAN_IP:-localhost}" \
        -addext "subjectAltName=${SAN_LIST}" \
        -addext "keyUsage=digitalSignature,keyEncipherment" \
        -addext "extendedKeyUsage=serverAuth"

    chmod 644 "${CERT_FILE}"
    chmod 640 "${KEY_FILE}"
    echo "[entrypoint] Certificado generado exitosamente con SAN: ${SAN_LIST}"
else
    echo "[entrypoint] Certificados SSL/TLS existentes detectados en ${CERT_DIR}."
fi

exec nginx -g "daemon off;"
