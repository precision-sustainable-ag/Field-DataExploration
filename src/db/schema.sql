CREATE TABLE IF NOT EXISTS locations (
    code TEXT PRIMARY KEY,
    display_name TEXT,
    parent_code TEXT REFERENCES locations(code)
);

CREATE TABLE IF NOT EXISTS samples (
    master_ref_id TEXT PRIMARY KEY,
    location_code TEXT REFERENCES locations(code),
    plant_type TEXT,
    species TEXT,
    height TEXT,
    size_class TEXT,
    growth_stage TEXT,
    cotton_variety TEXT,
    crop_or_fallow TEXT,
    crop_type_secondary TEXT,
    cover_crop_family TEXT,
    flower_fruit_or_seeds TEXT,
    cloud_cover TEXT,
    ground_residue TEXT,
    ground_cover TEXT,
    username TEXT,
    wirmastermeta_rowkey TEXT,
    wirmastermeta_timestamp TEXT,
    wircovercropsmeta_rowkey TEXT,
    wircovercropsmeta_timestamp TEXT,
    wircropsmeta_rowkey TEXT,
    wircropsmeta_timestamp TEXT,
    wirweedsmeta_rowkey TEXT,
    wirweedsmeta_timestamp TEXT,
    wirsoilsmeta_rowkey TEXT,
    wirsoilsmeta_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_code TEXT REFERENCES locations(code),
    batch_label TEXT NOT NULL,
    batch_date TEXT,
    UNIQUE(location_code, batch_date, batch_label)
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    blob_name TEXT UNIQUE NOT NULL,
    master_ref_id TEXT REFERENCES samples(master_ref_id),
    batch_id INTEGER REFERENCES batches(id),
    container TEXT,
    base_name TEXT,
    extension TEXT,
    size_mib REAL,
    upload_datetime_utc TEXT,
    exif_datetime TEXT,
    image_url TEXT,
    image_index TEXT,
    has_matching_jpg_and_raw BOOLEAN,
    sub_batch_index TEXT,
    stem TEXT,
    wirimagerefs_rowkey TEXT,
    wirimagerefs_timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_images_master_ref_id ON images(master_ref_id);
CREATE INDEX IF NOT EXISTS idx_images_batch_id ON images(batch_id);
CREATE INDEX IF NOT EXISTS idx_samples_location_code ON samples(location_code);
CREATE INDEX IF NOT EXISTS idx_batches_location_code ON batches(location_code);

CREATE TABLE IF NOT EXISTS raw_sample_attributes (
    source TEXT NOT NULL,
    master_ref_id TEXT NOT NULL,
    partition_key TEXT,
    row_key TEXT,
    source_timestamp TEXT,
    data TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (source, master_ref_id)
);

-- Physical presence of a file at a location that isn't already tracked by
-- `images` (which only models Azure Blob). Populated by the NFS and Globus/JUNO
-- scanners; blob presence is derived from `images` directly (container +
-- extension) instead of being duplicated here.
CREATE TABLE IF NOT EXISTS file_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_name TEXT NOT NULL,
    extension TEXT NOT NULL,        -- 'arw' | 'jpg'
    artifact_kind TEXT NOT NULL,    -- 'raw' | 'processed_jpg'
    storage_location TEXT NOT NULL, -- 'nfs' | 'juno'
    path TEXT NOT NULL,
    batch_label TEXT,
    master_ref_id TEXT REFERENCES samples(master_ref_id),
    size_bytes INTEGER,
    mtime_utc TEXT,
    scanned_at TEXT NOT NULL,
    UNIQUE(storage_location, path)
);

CREATE INDEX IF NOT EXISTS idx_file_locations_base_name ON file_locations(base_name);
CREATE INDEX IF NOT EXISTS idx_file_locations_batch_label ON file_locations(batch_label);

-- Materialized summary joining samples/images/file_locations: one row per
-- known image base_name, with sample metadata and a presence flag per
-- location/artifact kind. Rebuilt in full by db.reporting.refresh_file_status()
-- (a derived/summary table, not upserted incrementally) - query this directly
-- instead of re-running the join by hand.
CREATE TABLE IF NOT EXISTS file_status (
    base_name TEXT PRIMARY KEY,
    master_ref_id TEXT REFERENCES samples(master_ref_id),
    location_code TEXT,
    plant_type TEXT,
    species TEXT,
    crop_or_fallow TEXT,
    cover_crop_family TEXT,
    growth_stage TEXT,
    username TEXT,
    has_matching_jpg_and_raw BOOLEAN,
    raw_in_blob BOOLEAN,
    preview_jpg_in_blob BOOLEAN,
    raw_in_nfs BOOLEAN,
    processed_jpg_in_nfs BOOLEAN,
    raw_in_juno BOOLEAN,
    processed_jpg_in_juno BOOLEAN,
    needs_processing BOOLEAN,
    ready_for_juno_upload BOOLEAN,
    batch_id INTEGER REFERENCES batches(id),  -- only set when raw_in_nfs=1
    batch_label TEXT,                         -- batches.batch_label, only set when raw_in_nfs=1
    file_path TEXT,                           -- raw's NFS path, only set when raw_in_nfs=1
    refreshed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_status_location_code ON file_status(location_code);
CREATE INDEX IF NOT EXISTS idx_file_status_needs_processing ON file_status(needs_processing);
CREATE INDEX IF NOT EXISTS idx_file_status_ready_for_juno_upload ON file_status(ready_for_juno_upload);
CREATE INDEX IF NOT EXISTS idx_file_status_batch_id ON file_status(batch_id);

-- Raws that create_batches_db has grouped into a batch and assigned a target
-- path under batches_root, but that aren't archived anywhere yet (not on NFS
-- or JUNO per file_status) - the "about to be created" plan. Refreshed as a
-- full replace by create_batches_db.persist_planned_batches() each run,
-- independent of whether the download step actually executes afterward.
CREATE TABLE IF NOT EXISTS planned_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_name TEXT NOT NULL,
    raw_blob_name TEXT NOT NULL UNIQUE,
    batch_label TEXT NOT NULL,
    location_code TEXT,
    batch_date TEXT,
    sub_batch_index TEXT,
    target_path TEXT NOT NULL,
    master_ref_id TEXT REFERENCES samples(master_ref_id),
    planned_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_planned_batches_batch_label ON planned_batches(batch_label);

