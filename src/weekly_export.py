import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from omegaconf import DictConfig

log = logging.getLogger(__name__)

"""
    Bundles the current run's report folder (cfg.paths.reportdir_timestamp -
    plots, CSVs, inspection images from report_db/plot_by_season_db/
    image_inspection_db) into a single zip alongside the loose files, so a
    reviewer can grab the zip or browse the folder directly over Globus.
    Run after every stage that writes into reportdir_timestamp.
"""


def build_manifest(report_dir: Path, files: list[Path]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_dir": str(report_dir),
        "file_count": len(files),
        "files": sorted(str(f.relative_to(report_dir)) for f in files),
    }


def zip_report_dir(report_dir: Path, zip_path: Path) -> Path:
    files = [f for f in report_dir.rglob("*") if f.is_file() and f != zip_path]

    manifest_path = report_dir / "manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(report_dir, files), indent=2))
    files.append(manifest_path)

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.relative_to(report_dir))

    log.info(f"Zipped {len(files)} files into {zip_path}")
    return zip_path


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    report_dir = Path(cfg.paths.reportdir_timestamp)
    if not report_dir.exists():
        raise FileNotFoundError(
            f"{report_dir} does not exist - run report_db/plot_by_season_db/image_inspection_db first"
        )

    zip_path = report_dir / f"field_report_{cfg.job.job_now_date}.zip"
    zip_report_dir(report_dir, zip_path)

    log.info(f"{cfg.general.task} completed.")
