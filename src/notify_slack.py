import logging
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from omegaconf import DictConfig
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from db.connection import get_connection
from db.reporting import (
    load_location_code_corrections_dataframe,
    load_newly_processed_dataframe,
    load_not_started_subbatch_summary_dataframe,
    load_partial_subbatch_summary_dataframe,
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


def resolve_channel_id(client: WebClient, channel_id: str | list[str]) -> str:
    """files_upload_v2's completeUploadExternal requires an actual
    conversation id (C.../G.../D.../Z...) - chat.postMessage tolerates a bare
    user id (U...) and auto-opens the DM, but file upload doesn't, so resolve
    it to the conversation's channel id up front and use that everywhere.
    channel_id can be a single user id (bot<->that-user DM, the existing
    case) or a list of user ids - conversations.open then opens/resumes a
    private multi-person DM containing the bot plus everyone listed (Slack
    bots can only post into conversations they're a member of, so a group DM
    with the bot included is the closest equivalent to "post into a DM
    between two other people")."""
    if isinstance(channel_id, str):
        if not channel_id.startswith("U"):
            return channel_id
        user_ids = [channel_id]
    else:
        user_ids = list(channel_id)
    response = client.conversations_open(users=user_ids)
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


def load_planned_batches_table(conn) -> str | None:
    """Batches create_batches_db has planned/targeted but that don't exist
    on NFS yet."""
    df = load_planned_batches_summary_dataframe(conn)
    if df.empty:
        return None
    return df.to_string(index=False)


def load_partial_subbatches_table(conn) -> str | None:
    """Sub-batches (the raws/<sub_batch_index>/ folder within a batch) with
    some but not all raws processed yet - in progress, not done."""
    df = load_partial_subbatch_summary_dataframe(conn)
    if df.empty:
        return None
    return df.to_string(index=False)


def count_not_started_subbatches(conn) -> int:
    """Count of sub-batches with raws on NFS but zero processed JPGs - not
    started. The full backlog routinely has more rows than fit in a Slack
    table under MAX_TABLE_CHARS, so it's written to CSV by scan_file_locations
    (db.reporting.export_not_started_subbatches_csv) and attached as a file
    instead - this just backs the headline count."""
    return len(load_not_started_subbatch_summary_dataframe(conn))


def load_processed_species_table(conn) -> tuple[str | None, int]:
    """Processed-image counts by species, plus the overall total (sum across
    species) for the headline number."""
    df = load_processed_by_species_dataframe(conn)
    if df.empty:
        return None, 0
    return df.to_string(index=False), int(df["ProcessedCount"].sum())


def load_newly_processed_table(conn, num_past_days_for_report: int) -> tuple[str | None, int]:
    """New processed_jpg files that landed on NFS in the last
    num_past_days_for_report days (by state), plus the total for the
    headline number - see load_newly_processed_dataframe."""
    df = load_newly_processed_dataframe(conn, num_past_days_for_report)
    if df.empty:
        return None, 0
    return df.to_string(index=False), int(df["ProcessedCount"].sum())


def load_location_code_corrections_table(conn, num_past_days_for_report: int) -> tuple[str | None, int]:
    """Location-code auto-corrections in the last N days, plus the total
    count for the headline number - see load_location_code_corrections_dataframe."""
    df = load_location_code_corrections_dataframe(conn, num_past_days_for_report)
    if df.empty:
        return None, 0
    return df.to_string(index=False), int(df["Count"].sum())


def build_header_blocks(cfg: DictConfig, uploads_table: str | None) -> list:
    """Blocks for the top-level chat message: just the report header plus the
    first (uploads) table. Everything else is posted as a threaded reply -
    see build_reply_blocks."""
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"Field Data Weekly Report - {cfg.job.job_now_date}"}},
    ]

    if uploads_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Uploads (last {cfg.inspection.num_past_days_for_report} days) by state/plant type*\n```{truncate_table(uploads_table)}```"}})

    return blocks


def build_reply_blocks(
    cfg: DictConfig,
    status_table: str | None,
    blob_not_in_nfs_table: str | None,
    planned_batches_table: str | None,
    partial_subbatches_table: str | None,
    species_table: str | None,
    total_processed: int,
    newly_processed_table: str | None,
    total_newly_processed: int,
    location_corrections_table: str | None,
    total_location_corrections: int,
    globus_link: str | None,
) -> list:
    """Blocks for every table/link after the first (uploads) one, posted as a
    threaded reply under the header message - see build_header_blocks."""
    blocks = []

    if species_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Total images processed: {total_processed}*\n```{truncate_table(species_table)}```"}})

    if status_table or blob_not_in_nfs_table or planned_batches_table or partial_subbatches_table:
        blocks.append({"type": "divider"})
        blocks.append({"type": "header", "text": {"type": "plain_text", "text": "Processing / Batch Summary"}})
    if status_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Images needing processing vs. already processed (by state)*\n_Filtered to HasMatchingJpgAndRaw=True. NFS-only: both columns are based on raw/processed-JPG presence on NFS, not Azure Blob - images still only in Blob (not yet copied to NFS) aren't counted in either column._\n```{truncate_table(status_table)}```"}})
    if blob_not_in_nfs_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Images in Blob not yet copied to NFS (by state)*\n_Filtered to HasMatchingJpgAndRaw=True, matching create_batches_db's own eligibility filter._\n```{truncate_table(blob_not_in_nfs_table)}```"}})
    if planned_batches_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Batches planned but not yet created on NFS*\n```{truncate_table(planned_batches_table)}```"}})
    if partial_subbatches_table:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Sub-batches partially processed (raw count vs. processed count)*\n_Grouped by the raws/<sub_batch_index>/ folder within each batch, not the whole batch - see db.reporting.load_partial_subbatch_summary_dataframe. Full per-image detail is in images_needing_processing.csv in this run's report folder._\n```{truncate_table(partial_subbatches_table)}```"}})

    newly_processed_text = (
        f"*Newly processed in NFS (last {cfg.inspection.num_past_days_for_report} days): {total_newly_processed}*\n"
        f"_Processed-jpg files under developed-images/ first found by a scan in this window - "
        f"see db.reporting.load_newly_processed_dataframe._\n"
    )
    if newly_processed_table:
        newly_processed_text += f"```{truncate_table(newly_processed_table)}```"
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": newly_processed_text}})

    if location_corrections_table:
        blocks.append({"type": "divider"})
        blocks.append({"type": "header", "text": {"type": "plain_text", "text": "Data Quality"}})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": (
            f"*Location codes auto-corrected (last {cfg.inspection.num_past_days_for_report} days): {total_location_corrections}*\n"
            f"_A sample's/batch's location_code didn't match a real state - likely mistyped in the field app - so it was "
            f"inferred from that image's/batch's own filename prefix instead (e.g. 'OHA00623' -> OH). "
            f"See db.locations.infer_location_codes_from_images._\n"
            f"```{truncate_table(location_corrections_table)}```"
        )}})

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
        planned_batches_table = load_planned_batches_table(conn)
        partial_subbatches_table = load_partial_subbatches_table(conn)
        not_started_count = count_not_started_subbatches(conn)
        species_table, total_processed = load_processed_species_table(conn)
        newly_processed_table, total_newly_processed = load_newly_processed_table(
            conn, cfg.inspection.num_past_days_for_report
        )
        location_corrections_table, total_location_corrections = load_location_code_corrections_table(
            conn, cfg.inspection.num_past_days_for_report
        )
    finally:
        conn.close()

    globus_link = build_globus_link(cfg)
    header_blocks = build_header_blocks(cfg, uploads_table)
    reply_blocks = build_reply_blocks(
        cfg, status_table, blob_not_in_nfs_table,
        planned_batches_table, partial_subbatches_table,
        species_table, total_processed,
        newly_processed_table, total_newly_processed,
        location_corrections_table, total_location_corrections, globus_link,
    )

    client = WebClient(token=bot_token)
    try:
        channel_id = resolve_channel_id(client, channel_id)
        result = client.chat_postMessage(channel=channel_id, blocks=header_blocks, text=f"Field Data Weekly Report - {cfg.job.job_now_date}")

        if reply_blocks:
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=result["ts"],
                blocks=reply_blocks,
                text=f"Field Data Weekly Report - {cfg.job.job_now_date} (details)",
            )

        for filename, plot_dir_key in PLOTS_TO_ATTACH:
            plot_path = Path(cfg.paths[plot_dir_key]) / filename
            if plot_path.exists():
                client.files_upload_v2(channel=channel_id, file=str(plot_path), thread_ts=result["ts"], title=filename)
            else:
                log.warning(f"{plot_path} not found, skipping plot upload")

        if not_started_count:
            not_started_csv_path = report_dir / "not_started_subbatches.csv"
            if not_started_csv_path.exists():
                client.files_upload_v2(
                    channel=channel_id, file=str(not_started_csv_path), thread_ts=result["ts"],
                    title="not_started_subbatches.csv",
                )
            else:
                log.warning(f"{not_started_csv_path} not found (run scan_file_locations first), skipping upload")
    except SlackApiError as e:
        log.exception(f"Slack API call failed: {e.response['error']}")
        raise

    log.info(f"{cfg.general.task} completed.")
