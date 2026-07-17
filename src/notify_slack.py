import logging
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from omegaconf import DictConfig
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from db.connection import get_connection
from db.reporting import (
    load_partial_batch_summary_dataframe,
    load_planned_batches_summary_dataframe,
    load_processed_by_species_dataframe,
    load_processing_status_dataframe,
)
from utils.utils import read_yaml

log = logging.getLogger(__name__)

"""
    Posts the weekly report summary to Slack: uploads/processing/batch
    tables, two plots, and a Globus deep link into this run's report folder.
    Run last, after report_db/plot_by_season_db/image_inspection_db/
    weekly_export have populated and zipped cfg.paths.reportdir_timestamp.
"""

# (filename, cfg.paths key for the directory it lives in)
PLOTS_TO_ATTACH = [
    ("image_jpgs_vs_raws_by_state_current_season.png", "plots_current_season"),
    ("image_vs_raws_by_state.png", "plots_all_years"),
]


# Slack section blocks cap text at 3000 chars total, including the
# */```/``` wrapper build_blocks() adds around every table (worst case ~70
# chars) - 2900 leaves headroom for that without truncating tables that
# comfortably fit.
MAX_TABLE_CHARS = 2900


def truncate_table(table: str) -> str:
    """Cuts a rendered table down to whole lines under MAX_TABLE_CHARS, with
    a note on how many rows got dropped - a table that grows past Slack's
    block size limit should still post (truncated) rather than crash the
    whole weekly run."""
    if len(table) <= MAX_TABLE_CHARS:
        return table

    lines = table.split("\n")
    kept, total_len = [], 0
    for line in lines:
        if total_len + len(line) + 1 > MAX_TABLE_CHARS:
            break
        kept.append(line)
        total_len += len(line) + 1

    dropped = len(lines) - len(kept)
    return "\n".join(kept) + f"\n... ({dropped} more rows truncated)"


def resolve_channel_id(client: WebClient, channel_id: str) -> str:
    """files_upload_v2's completeUploadExternal requires an actual
    conversation id (C.../G.../D.../Z...) - chat.postMessage tolerates a bare
    user id (U...) and auto-opens the DM, but file upload doesn't, so resolve
    it to the DM's channel id up front and use that everywhere."""
    if not channel_id.startswith("U"):
        return channel_id
    response = client.conversations_open(users=[channel_id])
    return response["channel"]["id"]


def build_globus_link(cfg: DictConfig) -> str | None:
    collection_id = cfg.globus.get("reviewer_collection_id")
    base_path = cfg.globus.get("reviewer_collection_base_path")
    if not collection_id or not base_path:
        log.warning("globus.reviewer_collection_id/reviewer_collection_base_path not configured, skipping link")
        return None

    origin_path = f"{base_path.rstrip('/')}/{cfg.job.job_now_date}/"
    return f"https://app.globus.org/file-manager?origin_id={collection_id}&origin_path={quote(origin_path, safe='/')}"


def load_uploads_table(report_dir: Path, num_past_days_for_report: int) -> str | None:
    csv_path = report_dir / f"uploads_last_{num_past_days_for_report}_days_by_state.csv"
    if not csv_path.exists():
        log.warning(f"{csv_path} not found, skipping uploads table")
        return None
    df = pd.read_csv(csv_path)
    return df.to_string(index=False)


def load_matched_status_dataframe(conn) -> pd.DataFrame:
    """load_processing_status_dataframe(), limited to HasMatchingJpgAndRaw=True
    uploads (a clean raw+jpg pair in blob) - shared by the processing-status
    and blob-not-in-nfs tables so both use one query/filter and stay
    consistent with create_batches_db's own HasMatchingJpgAndRaw==True gate
    on what's eligible to copy to NFS in the first place."""
    df = load_processing_status_dataframe(conn)
    return df[df["HasMatchingJpgAndRaw"] == 1]


def load_processing_status_table(df: pd.DataFrame) -> str | None:
    """Images with a raw on NFS that still need RawTherapee processing vs.
    images already processed (developed-images/ present), by state. Both
    columns are NFS-only (raw_in_nfs/processed_jpg_in_nfs from
    file_locations) - an image still only in Azure Blob, not yet copied to
    NFS, counts toward neither (see load_blob_not_in_nfs_table)."""
    if df.empty:
        return None
    summary = (
        df.groupby("UsState")[["NeedsProcessing", "ProcessedJpgInNfs"]]
        .sum()
        .reset_index()
        .sort_values("UsState")
        .rename(columns={"UsState": "State", "ProcessedJpgInNfs": "AlreadyProcessed"})
    )
    return summary.to_string(index=False)


def load_blob_not_in_nfs_table(df: pd.DataFrame) -> str | None:
    """Images uploaded to Azure Blob with a matched raw+jpg pair, but whose
    raw hasn't been copied to NFS yet - the gap the processing-status
    table's NFS-only columns don't surface (see its docstring)."""
    pending = df[(df["RawInBlob"] == 1) & (df["RawInNfs"] == 0)]
    if pending.empty:
        return None
    summary = (
        pending.groupby("UsState")
        .size()
        .reset_index(name="InBlobNotInNfs")
        .sort_values("UsState")
        .rename(columns={"UsState": "State"})
    )
    return summary.to_string(index=False)


def load_partial_batches_table(conn) -> str | None:
    """Batches with both a raw count and a processed count on NFS, but not
    matching yet - in progress, not done."""
    df = load_partial_batch_summary_dataframe(conn)
    if df.empty:
        return None
    return df.to_string(index=False)


def load_planned_batches_table(conn) -> str | None:
    """Batches create_batches_db has planned/targeted but that don't exist
    on NFS yet."""
    df = load_planned_batches_summary_dataframe(conn)
    if df.empty:
        return None
    return df.to_string(index=False)


def load_processed_species_table(conn) -> tuple[str | None, int]:
    """Processed-image counts by species, plus the overall total (sum across
    species) for the headline number."""
    df = load_processed_by_species_dataframe(conn)
    if df.empty:
        return None, 0
    return df.to_string(index=False), int(df["ProcessedCount"].sum())


def build_blocks(
    cfg: DictConfig,
    uploads_table: str | None,
    status_table: str | None,
    blob_not_in_nfs_table: str | None,
    partial_batches_table: str | None,
    planned_batches_table: str | None,
    species_table: str | None,
    total_processed: int,
    globus_link: str | None,
) -> list:
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"Field Data Weekly Report - {cfg.job.job_now_date}"}},
    ]

    if uploads_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Uploads (last {cfg.inspection.num_past_days_for_report} days) by state/plant type*\n```{truncate_table(uploads_table)}```"}})
    
    if species_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Total images processed: {total_processed}*\n```{truncate_table(species_table)}```"}})

    if status_table or blob_not_in_nfs_table or partial_batches_table or planned_batches_table:
        blocks.append({"type": "divider"})
        blocks.append({"type": "header", "text": {"type": "plain_text", "text": "Processing / Batch Summary"}})
    if status_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Images needing processing vs. already processed (by state)*\n_Filtered to HasMatchingJpgAndRaw=True. NFS-only: both columns are based on raw/processed-JPG presence on NFS, not Azure Blob - images still only in Blob (not yet copied to NFS) aren't counted in either column._\n```{truncate_table(status_table)}```"}})
    if blob_not_in_nfs_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Images in Blob not yet copied to NFS (by state)*\n_Filtered to HasMatchingJpgAndRaw=True, matching create_batches_db's own eligibility filter._\n```{truncate_table(blob_not_in_nfs_table)}```"}})
    if partial_batches_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Batches partially processed (raw count vs. processed count)*\n```{truncate_table(partial_batches_table)}```"}})
    if planned_batches_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Batches planned but not yet created on NFS*\n```{truncate_table(planned_batches_table)}```"}})

    if globus_link:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"<{globus_link}|Open this week's report folder in Globus>"}})
    else:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "_Globus link not configured (conf/config.yaml: globus.reviewer_collection_id/reviewer_collection_base_path)_"}})

    return blocks


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    keys = read_yaml(cfg.pipeline_keys)
    slack_keys = keys.get("slack") or {}
    bot_token = slack_keys.get("bot_token")
    channel_id = slack_keys.get("channel_id")
    if not bot_token or not channel_id:
        raise ValueError("slack.bot_token and slack.channel_id (both in keys/authorized_keys.yaml) are required")

    report_dir = Path(cfg.paths.reportdir_timestamp)
    conn = get_connection(cfg.paths.db_path)
    try:
        uploads_table = load_uploads_table(report_dir, cfg.inspection.num_past_days_for_report)
        matched_status_df = load_matched_status_dataframe(conn)
        status_table = load_processing_status_table(matched_status_df)
        blob_not_in_nfs_table = load_blob_not_in_nfs_table(matched_status_df)
        partial_batches_table = load_partial_batches_table(conn)
        planned_batches_table = load_planned_batches_table(conn)
        species_table, total_processed = load_processed_species_table(conn)
    finally:
        conn.close()

    globus_link = build_globus_link(cfg)
    blocks = build_blocks(
        cfg, uploads_table, status_table, blob_not_in_nfs_table, partial_batches_table,
        planned_batches_table, species_table, total_processed, globus_link,
    )

    client = WebClient(token=bot_token)
    try:
        channel_id = resolve_channel_id(client, channel_id)
        result = client.chat_postMessage(channel=channel_id, blocks=blocks, text=f"Field Data Weekly Report - {cfg.job.job_now_date}")

        for filename, plot_dir_key in PLOTS_TO_ATTACH:
            plot_path = Path(cfg.paths[plot_dir_key]) / filename
            if plot_path.exists():
                client.files_upload_v2(channel=channel_id, file=str(plot_path), thread_ts=result["ts"], title=filename)
            else:
                log.warning(f"{plot_path} not found, skipping plot upload")
    except SlackApiError as e:
        log.exception(f"Slack API call failed: {e.response['error']}")
        raise

    log.info(f"{cfg.general.task} completed.")
