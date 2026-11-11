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
# Needs an SSH key from SUNNY to the server (ssh-keygen, then ssh-copy-id). For cron,
# e.g. every morning at 06:15:
#   15 6 * * * $HOME/envs/Field-DataExploration/scripts/push_snapshot.sh >> $HOME/push_snapshot.log 2>&1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f scripts/dashboard.env ] && set -a && . scripts/dashboard.env && set +a

DASHBOARD_HOST="${DASHBOARD_HOST:-$USER@10.76.169.110}"
DASHBOARD_DIR="${DASHBOARD_DIR:-field-dashboard-data}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT/dashboard/snapshot}"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

echo "$(date -Is) building snapshot in $SNAPSHOT_DIR"
"$UV" run python dashboard/snapshot.py --out-dir "$SNAPSHOT_DIR"

echo "$(date -Is) copying to $DASHBOARD_HOST:$DASHBOARD_DIR"
# --delay-updates swaps everything in at the end, so the app never sees a half-copied snapshot.
rsync -a --delete --delay-updates -e "ssh -o BatchMode=yes" "$SNAPSHOT_DIR/" "$DASHBOARD_HOST:$DASHBOARD_DIR/"
echo "$(date -Is) done"
