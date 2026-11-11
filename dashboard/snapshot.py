"""Build the dashboard snapshot: everything the Streamlit app shows, computed where
the data lives (SUNNY, where long-term storage is mounted) and written to one folder
that can be copied to the dashboard server.

Run on SUNNY from the repo root (read-only against the live database):

    uv run python dashboard/snapshot.py --out-dir <folder>

Writes to the folder (default dashboard/snapshot/):
    data.json             counts, storage summary, history and settings for the app
    cutouts.parquet       every cutout: species, location, season, batch and file path (Browse list)
    thumbs/, full/        cutout thumbnails and full-resolution copies (WebP) for Browse
    to_be_processed.csv   every image in the processing backlog
    history.csv           one row of totals per database refresh (feeds the trend charts)

Options: --db (default: long-term storage, or $FIELD_DB), --storage (field-batches,
default: next to field-db), --target, --thumbs-per-group, --no-full-size,
--skip-storage-scan.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import storage  # noqa: E402

HERE = Path(__file__).resolve().parent
DIMENSIONS = ["plant_type", "location_code", "season", "species"]
SCHEMA_VERSION = 1


def season_order(values) -> list[str]:
    known = sorted(v for v in values if v != "Unknown")
    return known + (["Unknown"] if "Unknown" in values else [])


def build_cube(images: pd.DataFrame, samples: pd.DataFrame) -> dict:
    """Aggregate to one row per (plant type, location, season, species, state,
    unrecognised status). The page filters and regroups this in the browser."""
    dims = {
        "plant_type": sorted(set(images.plant_type) | set(samples.plant_type)),
        "location_code": sorted(set(images.location_code) | set(samples.location_code)),
        "season": season_order(set(images.season) | set(samples.season)),
        "species": sorted(set(images.species) | set(samples.species)),
        "state": list(data.STATE_CATEGORY),
        "status": sorted(set(images.unrecognised_status)),
    }
    index = {name: {v: i for i, v in enumerate(values)} for name, values in dims.items()}

    grouped = (
        images.groupby(DIMENSIONS + ["state", "unrecognised_status"], observed=True)
        .agg(images=("base_name", "size"), cutouts=("n_cutouts", "sum"))
        .reset_index()
    )
    rows = [
        [index["plant_type"][r.plant_type], index["location_code"][r.location_code],
         index["season"][r.season], index["species"][r.species], index["state"][r.state],
         index["status"][r.unrecognised_status], int(r.images), int(r.cutouts)]
        for r in grouped.itertuples()
    ]
    sample_rows = [  # [plant type, location, species, samples, season]
        [index["plant_type"][r.plant_type], index["location_code"][r.location_code],
         index["species"][r.species], int(r.samples), index["season"][r.season]]
        for r in samples.groupby(["plant_type", "location_code", "species", "season"]).size()
        .rename("samples").reset_index().itertuples()
    ]
    return {"dims": dims, "rows": rows, "samples": sample_rows, "index": index}


def default_target(images: pd.DataFrame) -> int:
    """Median cutouts per species (species with any cutouts), rounded.
    A placeholder until the team agrees on a real target."""
    per_group = images[images.state == "Cutout"].groupby("species").n_cutouts.sum()
    if per_group.empty:
        return 100
    median = float(per_group.median())
    magnitude = 10 ** max(0, len(str(int(median))) - 2)
    return max(10, int(round(median / magnitude) * magnitude))


def save_images(source: Path, thumb: Path, full: Path | None, size: int = 320) -> tuple[int, int]:
    """Save a thumbnail (and optionally the full-resolution image) as WebP; return the original size."""
    from PIL import Image

    with Image.open(source) as img:
        width, height = img.size
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        if full is not None:
            img.save(full, format="WEBP", quality=90, method=4)
        img.thumbnail((size, size))
        square = Image.new("RGBA", (size, size), (0, 0, 0, 0))  # same shape for every thumbnail, so the grid lines up
        square.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        square.save(thumb, format="WEBP", quality=75)
    return width, height


def build_thumbnails(images: pd.DataFrame, root: Path, per_group: int, seed: int, index: dict,
                     out_dir: Path, full_size: bool) -> tuple[list, int, int]:
    """Up to per_group random cutouts per species x location, so Browse still has pictures
    after filtering by location. Writes thumbs/<base_name>.webp and full/<base_name>.webp."""
    thumb_dir, full_dir = out_dir / "thumbs", out_dir / "full"
    for d in (thumb_dir, full_dir):
        d.mkdir(exist_ok=True)
    thumbs, failed, full_bytes, keep = [], 0, 0, set()
    if per_group > 0:
        cutouts = images[(images.state == "Cutout") & images.cutout_path.notna()]
        picks = cutouts.sample(frac=1, random_state=seed).groupby(["species", "location_code"]).head(per_group)
        for r in picks.itertuples():
            name = f"{r.base_name}.webp"
            try:
                width, height = save_images(root / r.cutout_path, thumb_dir / name, full_dir / name if full_size else None)
            except Exception:  # missing, unreadable or corrupt file: skip it, count it
                failed += 1
                continue
            keep.add(name)
            if full_size:
                full_bytes += (full_dir / name).stat().st_size
            thumbs.append({
                "species": index["species"][r.species], "location": index["location_code"][r.location_code],
                "plant_type": index["plant_type"][r.plant_type], "season": index["season"][r.season],
                "name": r.base_name, "batch": r.batch_label if isinstance(r.batch_label, str) else "",
                "path": r.cutout_path, "w": width, "h": height,
                "thumb": f"thumbs/{name}", "full": f"full/{name}" if full_size else "",
            })
    for d in (thumb_dir, full_dir):  # drop files from earlier builds that weren't picked this time
        for old in d.iterdir():
            if old.is_file() and old.name not in keep:
                old.unlink()
    return thumbs, failed, full_bytes


def build_storage(scan: dict, images: pd.DataFrame, index: dict) -> dict:
    """Aggregate the disk scan to (plant type, location, product) so the page can filter it.
    Each file is matched to its image by name (<base_name>.ARW, <base_name>.jpg,
    <base_name>_<n>.png); files that don't match take the plant type and location
    most images in their batch have."""
    frame = scan["frame"]
    if frame.empty:
        return {"scanned": True, "rows": [], "products": [], "roles": {}, "extensions": {}, "unknown": {},
                **_scan_meta(scan, 0, 0)}
    by_name = images.drop_duplicates("base_name").set_index("base_name")[["plant_type", "location_code"]]
    stem = frame["stem"]
    exact = stem.map(by_name.plant_type)
    trimmed = stem.str.replace(r"_\d+$", "", regex=True)  # cutouts: <base_name>_<index>
    use_trimmed = exact.isna() & trimmed.isin(by_name.index)
    key = stem.where(~use_trimmed, trimmed)
    frame = frame.assign(plant_type=key.map(by_name.plant_type), location_code=key.map(by_name.location_code))
    matched_bytes = int(frame.loc[frame.plant_type.notna(), "bytes"].sum())
    per_batch = (
        images.dropna(subset=["batch_label"]).groupby("batch_label")[["plant_type", "location_code"]]
        .agg(lambda col: col.value_counts().index[0])
    )
    fallback = frame[["batch"]].join(per_batch, on="batch")
    frame["plant_type"] = frame.plant_type.fillna(fallback.plant_type)
    frame["location_code"] = frame.location_code.fillna(fallback.location_code)
    unmatched = int(frame.loc[frame.plant_type.isna(), "batch"].nunique())
    frame["pt"] = frame.plant_type.map(index["plant_type"]).fillna(-1).astype(int)
    frame["loc"] = frame.location_code.map(index["location_code"]).fillna(-1).astype(int)
    totals = frame.groupby("product")["bytes"].sum().sort_values(ascending=False)
    products = list(totals.index)
    pidx = {p: i for i, p in enumerate(products)}
    grouped = frame.groupby(["pt", "loc", "product"]).agg(files=("bytes", "size"), bytes=("bytes", "sum")).reset_index()
    rows = [[int(r.pt), int(r.loc), pidx[r.product], int(r.files), int(r.bytes)] for r in grouped.itertuples()]
    by_ext = frame.groupby(["product", "ext"]).agg(files=("bytes", "size"), bytes=("bytes", "sum")).reset_index()
    extensions = {
        p: [[e.ext, int(e.files), int(e.bytes)] for e in g.sort_values("bytes", ascending=False).head(4).itertuples()]
        for p, g in by_ext.groupby("product")
    }
    batch_share = frame.groupby("product")["batch"].nunique() / max(1, scan["batches"])
    roles = {p: storage.role(p, set(by_ext.loc[by_ext["product"] == p, "ext"]), float(batch_share[p])) for p in products}
    share = matched_bytes / max(1, int(frame["bytes"].sum()))
    return {"scanned": True, "rows": rows, "products": products, "roles": roles, "extensions": extensions,
            "unknown": unknown_paths(frame, [p for p in products if roles[p] == "unknown"]),
            **_scan_meta(scan, unmatched, share)}


def unknown_paths(frame: pd.DataFrame, products: list[str], limit: int = 500) -> dict:
    """Where the unrecognised data sits, relative to field-batches: each loose file, or each
    batch's copy of an unknown folder. Largest first, at most `limit` per product."""
    out = {}
    for product in products:
        part = frame[frame["product"] == product]
        if product == storage.TOP_LEVEL:
            items = part.assign(path=part["batch"] + "/" + part["name"].fillna(part["stem"] + part["ext"]), files=1)
            items = items[["path", "files", "bytes"]]
            kind = "files"
        else:
            items = part.groupby("batch").agg(files=("bytes", "size"), bytes=("bytes", "sum")).reset_index()
            items["path"] = items["batch"] + "/" + product + "/"
            items = items[["path", "files", "bytes"]]
            kind = "folders"
        items = items.sort_values(["bytes", "path"], ascending=[False, True])
        out[product] = {"kind": kind, "total": len(items),
                        "items": [[r.path, int(r.files), int(r.bytes)] for r in items.head(limit).itertuples()]}
    return out


def _scan_meta(scan: dict, unmatched: int, matched_share: float) -> dict:
    return {"root": scan.get("root", ""), "batches": scan["batches"], "errors": scan["errors"],
            "seconds": scan["seconds"], "unmatched_batches": unmatched, "matched_share": round(matched_share, 4)}


def update_history(path: Path, images: pd.DataFrame, refreshed_at: str, generated_at: str,
                   storage_bytes: int | None) -> list[dict]:
    counts = images.category.value_counts()
    row = {
        "refreshed_at": refreshed_at or generated_at,
        "generated_at": generated_at,
        "images": len(images),
        **{c: int(counts.get(c, 0)) for c in data.CATEGORIES},
        "cutout_images": int((images.state == "Cutout").sum()),
        "cutouts": int(images.n_cutouts.sum()),
        "storage_bytes": storage_bytes,
    }
    history = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    history = history[history.refreshed_at != row["refreshed_at"]]  # rebuilds replace that refresh's row
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True).sort_values("refreshed_at")
    history.to_csv(path, index=False)
    return json.loads(history.to_json(orient="records"))  # NaN -> null for builds without a storage scan


def write_json_atomically(path: Path, payload: dict) -> None:
    """Write next to the target, then rename, so the app never reads a half-written file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), default=int), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default=None, help="SQLite database (default: long-term storage, or $FIELD_DB)")
    parser.add_argument("--storage", help="field-batches folder that cutout_path is relative to (default: next to field-db)")
    parser.add_argument("--out-dir", default=str(HERE / "snapshot"))
    parser.add_argument("--target", type=int, help="cutouts wanted per species (Priorities tab)")
    parser.add_argument("--thumbs-per-group", type=int, default=3,
                        help="random cutouts per species x location on the Browse tab (0 = none)")
    parser.add_argument("--no-full-size", action="store_true", help="thumbnails only, no full-resolution copies")
    parser.add_argument("--skip-storage-scan", action="store_true", help="don't measure field-batches on disk")
    parser.add_argument("--storage-workers", type=int, default=16, help="parallel folder walkers for the disk scan")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.db = args.db or data.default_db_path()
    if not Path(args.db).is_file():
        sys.exit(f"Database not found: {args.db}. Pass --db <path>.")
    if not args.storage:
        sibling = Path(args.db).resolve().parent.parent / "field-batches"
        args.storage = str(sibling) if sibling.is_dir() else ""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    loaded = data.load(args.db)
    images, samples = loaded["images"], loaded["samples_without_images"]
    for check in loaded["checks"]:
        if not check["ok"]:
            print(f"WARNING: {check['check']}: expected {check['expected']:,}, got {check['actual']:,}", file=sys.stderr)

    cube = build_cube(images, samples)
    index = cube.pop("index")
    thumbs, failed_thumbs, full_bytes = build_thumbnails(
        images, Path(args.storage), args.thumbs_per_group if args.storage else 0, args.seed, index,
        out_dir, not args.no_full_size)

    storage_root = Path(args.storage) if args.storage else None
    if args.skip_storage_scan or not (storage_root and storage_root.is_dir()):
        disk = {"scanned": False, "reason": "skipped (--skip-storage-scan)" if args.skip_storage_scan
                else "no field-batches folder found (pass --storage)"}
        storage_bytes = None
    else:
        print(f"Measuring {storage_root} ({args.storage_workers} parallel walkers)...", flush=True)
        scan = storage.scan(storage_root, workers=args.storage_workers)
        scan["root"] = str(storage_root)
        disk = build_storage(scan, images, index)
        storage_bytes = int(scan["frame"]["bytes"].sum())
    history = update_history(out_dir / "history.csv", images, loaded["refreshed_at"], generated_at, storage_bytes)

    cutout_list = loaded["cutout_list"].rename(columns={"location_code": "location", "batch_label": "batch"})
    tmp = out_dir / "cutouts.parquet.tmp"
    cutout_list.to_parquet(tmp, index=False)
    os.replace(tmp, out_dir / "cutouts.parquet")

    todo = images[images.category == "To be processed"]
    todo[["base_name", "master_ref_id", "state", "location_code", "plant_type", "species", "season", "batch_label"]] \
        .sort_values(["state", "location_code", "species", "base_name"]) \
        .to_csv(out_dir / "to_be_processed.csv", index=False)

    payload = {
        "schema": SCHEMA_VERSION,
        **cube,
        "categories": data.CATEGORIES,
        "state_category": data.STATE_CATEGORY,
        "plant_statuses": data.PLANT_STATUSES,
        "no_plant_statuses": data.NO_PLANT_STATUSES,
        "aliases": data.SPECIES_ALIASES,
        "cover_crop_type": data.COVER_CROP_TYPE,
        "refreshed_at": loaded["refreshed_at"],
        "generated_at": generated_at,
        "checks": loaded["checks"],
        "target": args.target or default_target(images),
        "target_is_placeholder": args.target is None,
        "history": history,
        "thumbs": thumbs,
        "thumbs_failed": failed_thumbs,
        "thumbs_per_group": args.thumbs_per_group,
        "storage": disk,
        "field_batches": args.storage,
    }
    write_json_atomically(out_dir / "data.json", payload)

    counts = images.category.value_counts()
    print(f"Wrote snapshot to {out_dir}")
    print(f"  {len(images):,} images: " + ", ".join(f"{c} {counts.get(c, 0):,}" for c in data.CATEGORIES))
    print(f"  {len(samples):,} samples without images; {len(todo):,} backlog rows in to_be_processed.csv")
    print(f"  {len(cutout_list):,} cutouts listed for Browse; {len(thumbs):,} sample thumbnails" + (f", {failed_thumbs:,} unreadable" if failed_thumbs else "")
          + (f"; {full_bytes / 1e6:,.0f} MB full size" if thumbs and not args.no_full_size else ""))
    if disk["scanned"]:
        print(f"  Storage: {storage_bytes / 1e12:,.2f} TB in {disk['batches']:,} batch folders, scanned in {disk['seconds']:,} s"
              + (f" ({disk['errors']:,} unreadable entries)" if disk["errors"] else ""))
    else:
        print(f"  Storage: {disk['reason']}")
    if failed_thumbs and not thumbs:
        print("  No cutout could be read: pass --storage <the field-batches folder>.")
    odd = images.unrecognised_status[images.unrecognised_status != ""].value_counts()
    if not odd.empty:
        print("  NOTE: not recognised, counted as backlog: " + ", ".join(f"{k} {v:,}" for k, v in odd.items()))


if __name__ == "__main__":
    main()
