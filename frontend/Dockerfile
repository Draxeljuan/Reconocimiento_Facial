FROM nginx:1.27-alpine

# Instalar OpenSSL únicamente dentro del contenedor para autogenerar certificados TLS LAN
RUN apk add --no-cache openssl

# Configuración de Nginx (HTTPS + Reverse-Proxy hacia FastAPI)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Archivos estáticos de la SPA (HTML5, CSS3, Vanilla JS + WebRTC)
COPY public/ /usr/share/nginx/html/

# Script de arranque para aprovisionamiento automático de certificados autofirmados
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh && mkdir -p /etc/nginx/certs

EXPOSE 80 443

ENTRYPOINT ["/entrypoint.sh"]
