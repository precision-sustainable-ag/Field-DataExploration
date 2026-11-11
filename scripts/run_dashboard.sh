#!/usr/bin/env bash
# Start, stop or check the Field-AgIR dashboard on this server.
#
#   scripts/run_dashboard.sh start | stop | restart | status
#
# Serves HTTPS on all interfaces (port 8501 by default) using the certificate from
# scripts/make_cert.sh, so anyone on the NCSU VPN can open https://<server IP>:8501.
# Settings come from scripts/dashboard.env (copy dashboard.env.example).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f scripts/dashboard.env ] && set -a && . scripts/dashboard.env && set +a

PORT="${PORT:-8501}"
CERT_DIR="${FIELD_DASHBOARD_CERT_DIR:-$HOME/.config/field-dashboard}"
STATE_DIR="${FIELD_DASHBOARD_STATE_DIR:-$HOME/.local/state/field-dashboard}"
PID_FILE="$STATE_DIR/dashboard.pid"
LOG_FILE="$STATE_DIR/dashboard.log"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"  # cron has a minimal PATH
export FIELD_DASHBOARD_DATA="${FIELD_DASHBOARD_DATA:-$HOME/field-dashboard-data}"
mkdir -p "$STATE_DIR"

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

start() {
    if running; then echo "Already running (pid $(cat "$PID_FILE"))."; return 0; fi
    if [ ! -f "$CERT_DIR/cert.pem" ] || [ ! -f "$CERT_DIR/key.pem" ]; then
        echo "No certificate in $CERT_DIR. Run scripts/make_cert.sh first." >&2
        exit 1
    fi
    nohup "$UV" run streamlit run dashboard/app.py \
        --server.address 0.0.0.0 \
        --server.port "$PORT" \
        --server.headless true \
        --server.sslCertFile "$CERT_DIR/cert.pem" \
        --server.sslKeyFile "$CERT_DIR/key.pem" \
        --server.fileWatcherType none \
        --browser.gatherUsageStats false \
        >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    for _ in $(seq 1 30); do
        if curl -skf "https://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
            echo "Dashboard up at https://$(hostname -I | awk '{print $1}'):$PORT (pid $(cat "$PID_FILE")). Log: $LOG_FILE"
            return 0
        fi
        sleep 2
    done
    echo "Started, but it isn't answering yet. Check $LOG_FILE" >&2
    exit 1
}

stop() {
    if running; then
        kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE"
        echo "Stopped."
    else
        rm -f "$PID_FILE"
        echo "Not running."
    fi
}

case "${1:-status}" in
    start) start ;;
    stop) stop ;;
    restart) stop; sleep 2; start ;;
    status)
        if running && curl -skf "https://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
            echo "Running and healthy (pid $(cat "$PID_FILE")), https://$(hostname -I | awk '{print $1}'):$PORT"
        elif running; then
            echo "Process is up (pid $(cat "$PID_FILE")) but not answering. See $LOG_FILE"; exit 1
        else
            echo "Not running."; exit 1
        fi ;;
    *) echo "Usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac
