#!/usr/bin/env bash
# For cron: restart the dashboard if it isn't answering. Install with `crontab -e`:
#
#   @reboot      $HOME/envs/Field-DataExploration/scripts/check_dashboard.sh
#   */10 * * * * $HOME/envs/Field-DataExploration/scripts/check_dashboard.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${FIELD_DASHBOARD_STATE_DIR:-$HOME/.local/state/field-dashboard}"
mkdir -p "$STATE_DIR"

if ! "$ROOT/scripts/run_dashboard.sh" status >/dev/null 2>&1; then
    echo "$(date -Is) not healthy, restarting" >>"$STATE_DIR/check.log"
    "$ROOT/scripts/run_dashboard.sh" restart >>"$STATE_DIR/check.log" 2>&1
fi
