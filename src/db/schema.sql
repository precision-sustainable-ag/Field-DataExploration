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

