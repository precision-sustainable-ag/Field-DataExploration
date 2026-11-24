#!/usr/bin/env bash
# Runs on SUNNY (where long-term storage is mounted): build the dashboard snapshot and
# copy it to the dashboard server over SSH.
#
#   scripts/push_snapshot.sh
#
# Settings (environment or scripts/dashboard.env):
#   DASHBOARD_HOST   user@server to copy to          (default: $USER@10.76.169.110)
#   DASHBOARD_DIR    folder on that server           (default: field-dashboard-data, in the home folder)
#   SNAPSHOT_DIR     where to build it on SUNNY      (default: dashboard/snapshot)
#
# Every run leaves status.json next to the data: ok, or which step failed and why. If the build
# fails, that status is still copied over, so the dashboard shows a "latest update failed" banner.
#
# Needs an SSH key from SUNNY to the server (ssh-keygen, then ssh-copy-id). For cron,
# e.g. every morning at 8:00 (SUNNY's local time, Eastern; the weekly database rebuild finishes around 6:30):
#   0 8 * * * $HOME/envs/Field-DataExploration/scripts/push_snapshot.sh >> $HOME/push_snapshot.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f scripts/dashboard.env ] && set -a && . scripts/dashboard.env && set +a

DASHBOARD_HOST="${DASHBOARD_HOST:-$USER@10.76.169.110}"
DASHBOARD_DIR="${DASHBOARD_DIR:-field-dashboard-data}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT/dashboard/snapshot}"
FIELD_DB="${FIELD_DB:-/mnt/research-projects/r/raatwell/longterm_images3/field-db/field_exploration.db}"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
STATUS="$SNAPSHOT_DIR/status.json"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
mkdir -p "$SNAPSHOT_DIR"

write_status() {  # ok(true|false) step message
    local msg
    msg=$(printf '%s' "$3" | tr '\n\r\t' '   ' | sed 's/\\/\\\\/g; s/"/\\"/g' | cut -c1-600)
    printf '{"ok": %s, "at": "%s", "step": "%s", "message": "%s"}\n' \
        "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" "$msg" >"$STATUS"
}

send_failure() {  # step message: record the failure and send just status.json, so the dashboard shows it
    write_status false "$1" "$2"
    echo "$(date -Is) FAILED while $1: $2"
    rsync -a -e "$SSH" "$STATUS" "$DASHBOARD_HOST:$DASHBOARD_DIR/status.json" \
        || echo "$(date -Is) could not send the status either (the dashboard will warn once the data is 2 days old)"
    exit 1
}

# The research drive is mounted with Kerberos (sec=krb5p): every read needs a valid ticket, which
# comes from logging in to SUNNY and lasts about 14 hours. Check first, so the failure says why.
if ! err=$(head -c 1 "$FIELD_DB" 2>&1 >/dev/null); then
    case "$err" in
        *"Permission denied"*)
            send_failure "reading the database on SUNNY" "Permission denied. SUNNY's login to the research drive (Kerberos ticket) has expired. Fix: ssh to SUNNY (logging in gives a new ticket), then run scripts/push_snapshot.sh." ;;
        *)
            send_failure "reading the database on SUNNY" "Can't read $FIELD_DB: $err" ;;
    esac
fi

echo "$(date -Is) building snapshot in $SNAPSHOT_DIR"
if ! "$UV" run python dashboard/snapshot.py --out-dir "$SNAPSHOT_DIR" 2>&1 | tee "$LOG"; then
    last=$(grep -v '^[[:space:]]*$' "$LOG" | tail -n 1)  # the error itself, not the whole traceback
    case "$last" in
        *PermissionError*|*"Permission denied"*)
            send_failure "building the snapshot on SUNNY" "Permission denied partway through. SUNNY's login to the research drive (Kerberos ticket) probably expired. Fix: ssh to SUNNY, then run scripts/push_snapshot.sh. ($last)" ;;
        *)
            send_failure "building the snapshot on SUNNY" "$last (full output: ~/push_snapshot.log on SUNNY)" ;;
    esac
fi

write_status true "done" ""
echo "$(date -Is) copying to $DASHBOARD_HOST:$DASHBOARD_DIR"
# --delay-updates swaps everything in at the end, so the app never sees a half-copied snapshot.
if ! rsync -a --delete --delay-updates -e "$SSH" "$SNAPSHOT_DIR/" "$DASHBOARD_HOST:$DASHBOARD_DIR/" 2>&1 | tee "$LOG"; then
    echo "$(date -Is) FAILED to copy to $DASHBOARD_HOST (the dashboard will warn once the data is 2 days old)"
    exit 1
fi
echo "$(date -Is) done"
