#!/usr/bin/env python3
# fmt: off
# isort: off
import logging

from omegaconf import DictConfig

from db.connection import get_connection
from db.locations import (
    apply_batch_location_corrections,
    apply_location_corrections,
    infer_batch_location_codes_from_images,
    infer_location_codes_from_images,
)

log = logging.getLogger(__name__)

"""
    Repairs samples.location_code and batches.location_code where either is a
    known placeholder/typo (cfg.invalid_location_codes, e.g. 'DV' entered
    instead of a real state), by inferring the true code from the image
    filename prefix - see db.locations.infer_location_codes_from_images /
    infer_batch_location_codes_from_images. Re-runnable: only touches rows
    still on an invalid code, so it's safe to run again after new data comes
    in with the same typo.

    Only updates DB columns - never renames batch_label or any physical
    NFS/JUNO folder. A batch whose location_code this corrects may already
    have a real on-disk folder under its old (wrong) batch_label; that
    rename, if wanted, is a separate manual/deliberate step.

    Not part of the automatic pipeline - run manually:
        python main.py general.task=fix_location_codes +pipeline=[fix_location_codes]
"""


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    codes_to_fix = set(cfg.get("invalid_location_codes", []))
    if not codes_to_fix:
        log.warning("cfg.invalid_location_codes is empty, nothing to do")
        return

    valid_codes = set(cfg.state_list) - codes_to_fix
    conn = get_connection(cfg.paths.db_path)
    try:
        corrections, unresolved = infer_location_codes_from_images(conn, codes_to_fix, valid_codes)
        num_corrected = apply_location_corrections(conn, corrections)

        batch_corrections, unresolved_batches = infer_batch_location_codes_from_images(conn, codes_to_fix, valid_codes)
        num_batches_corrected = apply_batch_location_corrections(conn, batch_corrections)

        conn.commit()
    finally:
        conn.close()

    log.info(
        f"Corrected location_code for {num_corrected} samples and "
        f"{num_batches_corrected} batches ({sorted(codes_to_fix)} -> inferred "
        f"from image filename prefix)"
    )
    if unresolved:
        log.warning(
            f"{len(unresolved)} samples still have an invalid location_code "
            f"(no unambiguous image-filename prefix found), left unchanged: "
            f"{unresolved[:10]}{'...' if len(unresolved) > 10 else ''}"
        )
    if unresolved_batches:
        log.warning(
            f"{len(unresolved_batches)} batches still have an invalid location_code, "
            f"left unchanged (batch ids): {unresolved_batches[:10]}"
            f"{'...' if len(unresolved_batches) > 10 else ''}"
        )
    log.info(f"{cfg.general.task} completed.")
