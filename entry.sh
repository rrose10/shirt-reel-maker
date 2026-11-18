#!/bin/sh
# Runs inside the nginx container: makes the data folders, creates a one-time
# self-signed https certificate (needed for the fast video mode), then starts nginx.
set -e
mkdir -p /certs /data/videos /data/music /data/presets /data/.tmp
if [ ! -f /certs/cert.pem ] || [ ! -f /certs/key.pem ]; then
  apk add --no-cache openssl >/dev/null
  SAN="DNS:localhost"
  for ip in $CERT_IPS; do SAN="$SAN,IP:$ip"; done
  openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout /certs/key.pem -out /certs/cert.pem \
    -subj "/CN=ecotech-reels" -addext "subjectAltName=$SAN"
fi
chmod -R a+rwX /data 2>/dev/null || true
exec nginx -g "daemon off;"
