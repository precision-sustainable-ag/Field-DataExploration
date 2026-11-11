#!/usr/bin/env bash
# Create a self-signed HTTPS certificate for the dashboard (no admin rights needed).
# Browsers will warn once that it isn't from a known authority; the connection is still encrypted.
#
#   scripts/make_cert.sh [server IP] [host name]
#
# Defaults: the server's first IP address and its short host name. Valid for about two years.
set -euo pipefail

IP="${1:-$(hostname -I | awk '{print $1}')}"
NAME="${2:-$(hostname)}"
DIR="${FIELD_DASHBOARD_CERT_DIR:-$HOME/.config/field-dashboard}"

mkdir -p "$DIR"
chmod 700 "$DIR"
openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout "$DIR/key.pem" -out "$DIR/cert.pem" \
    -subj "/CN=$NAME" -addext "subjectAltName=IP:$IP,DNS:$NAME" 2>/dev/null
chmod 600 "$DIR/key.pem"

echo "Certificate for https://$IP:8501 (and https://$NAME:8501) written to $DIR"
openssl x509 -in "$DIR/cert.pem" -noout -enddate | sed 's/notAfter=/Expires: /'
