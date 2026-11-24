"""Measure what the field data takes up on disk, for the report's Storage tab.

Walks every batch folder under field-batches/ (one thread per batch, since NFS
is latency-bound) and records each file's size and data product, which is the
first subfolder inside a batch (raws/, cutouts/, ...). Nothing is read or
written, only stat()ed.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
JPG_EXTENSIONS = {".jpg", ".jpeg"}
TOP_LEVEL = "unknown"  # files sitting directly in a batch folder, not in any product subfolder
DEVELOPED_NAMES = ("develop", "jpg", "jpeg", "image", "processed")


def _walk(batch_dir: Path) -> tuple[list, int]:
    """[(product, extension, file stem, bytes, name)] for every file in one batch folder, plus an
    error count. name (the exact file name) is kept only for loose files, to list their paths."""
    files, errors = [], 0
    stack = [(batch_dir, None)]
    while stack:
        folder, product = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append((entry.path, product or entry.name))
                        elif entry.is_file(follow_symlinks=False):
                            stem, ext = os.path.splitext(entry.name)
                            files.append((product or TOP_LEVEL, ext.lower() or "(none)", stem,
                                          entry.stat(follow_symlinks=False).st_size,
                                          None if product else entry.name))
                    except OSError:
                        errors += 1
        except OSError:
            errors += 1
    return files, errors


def scan(root: Path, workers: int = 16) -> dict:
    """One row per file under root (the field-batches folder): batch, product, ext, stem, bytes."""
    start = time.time()
    batches = sorted(p for p in root.iterdir() if p.is_dir())
    parts, errors = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch, (files, errs) in zip(batches, pool.map(_walk, batches)):
            errors += errs
            if files:
                part = pd.DataFrame(files, columns=["product", "ext", "stem", "bytes", "name"])
                part.insert(0, "batch", batch.name)
                parts.append(part)
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["batch", "product", "ext", "stem", "bytes", "name"])
    return {"frame": frame, "batches": len(batches), "errors": errors, "seconds": round(time.time() - start, 1)}


def role(product: str, extensions: set[str], batch_share: float) -> str:
    """Guess what a product folder holds, so the report can project future storage.
    batch_share is the fraction of batch folders that have this subfolder: a JPG-only
    folder counts as developed images only if it's named like it or is in most batches
    (so a one-off folder holding a single JPG stays unknown)."""
    name = product.lower()
    if product == TOP_LEVEL:
        return "unknown"
    if "cutout" in name:
        return "cutouts"
    if "mask" in name:
        return "masks"
    if extensions & RAW_EXTENSIONS:
        return "raws"
    if any(word in name for word in DEVELOPED_NAMES) or (extensions and extensions <= JPG_EXTENSIONS and batch_share >= 0.5):
        return "developed"
    return "unknown"
