#!/usr/bin/env bash
# Weekly cron entry point: refreshes the DB from all 4 sources, builds the
# report/plots/inspection images, zips them, and posts the summary to Slack.
#
# cron runs with a minimal environment (no PATH from .bashrc/.profile), so
# this script sets up its own PATH rather than relying on `uv`/`globus`
# already being reachable.
#
# Install with (adjust day/time as desired):
#   (crontab -l 2>/dev/null; echo "0 6 * * 1 $(pwd)/scripts/run_weekly_report.sh") | crontab -

set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CRON_LOG_DIR="${PROJECT_ROOT}/logging/cron"
mkdir -p "${CRON_LOG_DIR}"
CRON_LOG_FILE="${CRON_LOG_DIR}/run_weekly_report_$(date +%Y-%m-%d_%H-%M-%S).log"

{
    echo "Starting weekly field report: $(date -Is)"

    uv run python main.py \
        "+pipeline=[wir_table_generator,wir_blob_data_generator,merge_samples,append_datetime_db,report_db,plot_by_season_db,scan_file_locations,image_inspection_db,weekly_export,notify_slack]"

    echo "Finished weekly field report: $(date -Is)"
} >>"${CRON_LOG_FILE}" 2>&1
