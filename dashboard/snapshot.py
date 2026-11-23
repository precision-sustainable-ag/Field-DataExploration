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
import time
from concurrent.futures import ProcessPoolExecutor
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


def _make_pictures(job: tuple) -> tuple[int, int] | None:
    """Worker: one cutout's thumbnail (and full-size copy). Top level so worker processes can run it."""
    source, thumb, full = job
    try:
        return save_images(Path(source), Path(thumb), Path(full) if full else None)
    except Exception:  # missing, unreadable or corrupt file
        return None


def build_thumbnails(images: pd.DataFrame, root: Path, per_group: int, seed: int, index: dict,
                     out_dir: Path, full_size: bool, workers: int) -> tuple[list, int, int]:
    """Up to per_group cutouts per species x location x season for the Browse sample pictures.

    Picks are stable: cutouts are ranked by a hash of their path, so the same ones are chosen
    day after day and pictures made by earlier runs are reused. Only new picks are converted,
    in parallel. Writes thumbs/<base_name>.webp and full/<base_name>.webp."""
    thumb_dir, full_dir = out_dir / "thumbs", out_dir / "full"
    for d in (thumb_dir, full_dir):
        d.mkdir(exist_ok=True)
    sizes_file = thumb_dir / "sizes.json"  # original width x height of each picture, so reuse needs no file reads
    try:
        known_sizes = json.loads(sizes_file.read_text())
    except (OSError, ValueError):
        known_sizes = {}
    thumbs, failed, keep, new_sizes = [], 0, set(), {}
    if per_group > 0:
        cutouts = images[(images.state == "Cutout") & images.cutout_path.notna()]
        rank = pd.util.hash_pandas_object(cutouts.cutout_path.astype(str) + f"|{seed}", index=False).values
        picks = cutouts.assign(_rank=rank).sort_values("_rank") \
            .groupby(["species", "location_code", "season"], observed=True).head(per_group)
        jobs, reused = {}, 0
        for r in picks.itertuples():
            name = f"{r.base_name}.webp"
            have = (thumb_dir / name).exists() and (not full_size or (full_dir / name).exists()) and name in known_sizes
            if have:
                new_sizes[name] = known_sizes[name]
                reused += 1
            else:
                jobs[name] = (str(root / r.cutout_path), str(thumb_dir / name), str(full_dir / name) if full_size else None)
        print(f"      {len(picks):,} sample pictures: {reused:,} reused from earlier runs, {len(jobs):,} to make"
              + (f" with {workers} workers" if jobs else ""), flush=True)
        start, done = time.time(), 0
        names = list(jobs)
        results = map(_make_pictures, jobs.values()) if workers <= 1 else None
        pool = None
        if results is None:
            pool = ProcessPoolExecutor(max_workers=workers)
            results = pool.map(_make_pictures, jobs.values(), chunksize=4)
        try:
            for name, dims in zip(names, results):
                done += 1
                if dims is None:
                    failed += 1
                else:
                    new_sizes[name] = list(dims)
                if done % max(1, len(names) // 10) == 0 or done == len(names):
                    print(f"      made {done:,}/{len(names):,} pictures ({time.time() - start:,.0f} s)", flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
        for r in picks.itertuples():
            name = f"{r.base_name}.webp"
            if name not in new_sizes:
                continue
            keep.add(name)
            w, h = new_sizes[name]
            thumbs.append({
                "species": index["species"][r.species], "location": index["location_code"][r.location_code],
                "plant_type": index["plant_type"][r.plant_type], "season": index["season"][r.season],
                "name": r.base_name, "batch": r.batch_label if isinstance(r.batch_label, str) else "",
                "path": r.cutout_path, "w": w, "h": h,
                "thumb": f"thumbs/{name}", "full": f"full/{name}" if full_size else "",
            })
    for d in (thumb_dir, full_dir):  # drop pictures that are no longer picked
        for old in d.iterdir():
            if old.is_file() and old.suffix == ".webp" and old.name not in keep:
                old.unlink()
    sizes_file.write_text(json.dumps(new_sizes))
    full_bytes = sum((full_dir / n).stat().st_size for n in keep if (full_dir / n).exists()) if full_size else 0
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
        # Per-state counts (backlog steps, no-plant images); older rows lack them and read as empty.
        **{state: int((images.state == state).sum()) for state in data.STATE_CATEGORY},
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
                        help="sample pictures per species x location x season on the Browse tab (0 = none)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2),
                        help="processes converting sample pictures (default: up to 8)")
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
    started = time.time()

    def step(n: int, what: str) -> None:
        print(f"[{n}/6] {what} ({time.time() - started:,.0f} s so far)", flush=True)

    step(1, f"Reading the database {args.db}")
    loaded = data.load(args.db)
    images, samples = loaded["images"], loaded["samples_without_images"]
    for check in loaded["checks"]:
        if not check["ok"]:
            print(f"WARNING: {check['check']}: expected {check['expected']:,}, got {check['actual']:,}", file=sys.stderr)

    step(2, "Summarising counts")
    cube = build_cube(images, samples)
    index = cube.pop("index")
    step(3, "Sample pictures for Browse")
    thumbs, failed_thumbs, full_bytes = build_thumbnails(
        images, Path(args.storage), args.thumbs_per_group if args.storage else 0, args.seed, index,
        out_dir, not args.no_full_size, args.workers)

    storage_root = Path(args.storage) if args.storage else None
    if args.skip_storage_scan or not (storage_root and storage_root.is_dir()):
        step(4, "Storage: skipped")
        disk = {"scanned": False, "reason": "skipped (--skip-storage-scan)" if args.skip_storage_scan
                else "no field-batches folder found (pass --storage)"}
        storage_bytes = None
    else:
        step(4, f"Measuring storage in {storage_root} ({args.storage_workers} parallel walkers)")
        scan = storage.scan(storage_root, workers=args.storage_workers)
        scan["root"] = str(storage_root)
        disk = build_storage(scan, images, index)
        storage_bytes = int(scan["frame"]["bytes"].sum())
    history = update_history(out_dir / "history.csv", images, loaded["refreshed_at"], generated_at, storage_bytes)

    step(5, "Cutout list and backlog CSV")
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
        "thumbs_grouping": "species × location × season",
        "storage": disk,
        "field_batches": args.storage,
    }
    step(6, "Writing data.json")
    write_json_atomically(out_dir / "data.json", payload)

    counts = images.category.value_counts()
    print(f"Wrote snapshot to {out_dir} in {time.time() - started:,.0f} s")
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
