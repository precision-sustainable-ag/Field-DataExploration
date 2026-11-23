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
# e.g. every morning at 06:15:
#   15 6 * * * $HOME/envs/Field-DataExploration/scripts/push_snapshot.sh >> $HOME/push_snapshot.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f scripts/dashboard.env ] && set -a && . scripts/dashboard.env && set +a

DASHBOARD_HOST="${DASHBOARD_HOST:-$USER@10.76.169.110}"
DASHBOARD_DIR="${DASHBOARD_DIR:-field-dashboard-data}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT/dashboard/snapshot}"
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

echo "$(date -Is) building snapshot in $SNAPSHOT_DIR"
if ! "$UV" run python dashboard/snapshot.py --out-dir "$SNAPSHOT_DIR" 2>&1 | tee "$LOG"; then
    write_status false "building the snapshot on SUNNY" "$(tail -n 5 "$LOG")"
    echo "$(date -Is) FAILED while building; sending the failure status so the dashboard shows it"
    rsync -a -e "$SSH" "$STATUS" "$DASHBOARD_HOST:$DASHBOARD_DIR/status.json" \
        || echo "$(date -Is) could not send the status either (the dashboard will warn once the data is 2 days old)"
    exit 1
fi

write_status true "done" ""
echo "$(date -Is) copying to $DASHBOARD_HOST:$DASHBOARD_DIR"
# --delay-updates swaps everything in at the end, so the app never sees a half-copied snapshot.
if ! rsync -a --delete --delay-updates -e "$SSH" "$SNAPSHOT_DIR/" "$DASHBOARD_HOST:$DASHBOARD_DIR/" 2>&1 | tee "$LOG"; then
    echo "$(date -Is) FAILED to copy to $DASHBOARD_HOST (the dashboard will warn once the data is 2 days old)"
    exit 1
fi
echo "$(date -Is) done"
