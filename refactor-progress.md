# Refactor Progress

Tracks what's been implemented against `refactor-plan.md`'s build order (section 5),
and what's left. Sections 1–3 of the build order are done; this file is the detailed
record of what changed and the plan for section 4 onward.

All new code lives alongside the existing pipeline — nothing described below has
modified or removed any existing script's behavior. `wir_table_generator.py` and
`wir_blob_data_generator.py` are the only existing files whose *internals* changed
(section 2), and they still produce byte-for-byte the same CSVs as before. Everything
else (`process_blob_analysis.py`, `process_tables_analysis.py`, `create_batches.py`,
`report.py`, `plot_by_season.py`, `append_datetime.py`, `image_inspection.py`) is
untouched and still runs exactly as it did before this refactor started.

---

## Section 1 — Schema + migration (done)

**Goal (plan section 5, step 1):** stand up the SQLite schema for
Image/Sample/Location/Batch, and write a one-time script to load the existing CSVs
into it.

**What was built:**

- `src/db/schema.sql` — DDL for four tables:
  - `locations(code PK, display_name, parent_code FK → locations.code)`
  - `samples(master_ref_id PK, location_code FK, plant_type, species, height,
    size_class, growth_stage, cotton_variety, crop_or_fallow, crop_type_secondary,
    cover_crop_family, flower_fruit_or_seeds, cloud_cover, ground_residue,
    ground_cover, username, + one rowkey/timestamp lineage pair per source table:
    wirmastermeta, wircovercropsmeta, wircropsmeta, wirweedsmeta, wirsoilsmeta)`
  - `batches(id PK, location_code FK, batch_label, batch_date, UNIQUE(location_code,
    batch_date, batch_label))`
  - `images(id PK, blob_name UNIQUE, master_ref_id FK, batch_id FK, container,
    base_name, extension, size_mib, upload_datetime_utc, exif_datetime, image_url,
    image_index, has_matching_jpg_and_raw, sub_batch_index, stem,
    wirimagerefs_rowkey, wirimagerefs_timestamp)`
- `src/db/connection.py` — `get_connection(db_path)`, applies `schema.sql` via
  `CREATE TABLE IF NOT EXISTS`, enables `PRAGMA foreign_keys = ON`.
- `src/db/normalize.py` — one canonical `normalize_datetime(series)` (EXIF
  `YYYY:MM:DD HH:MM:SS` → `YYYY-MM-DD HH:MM:SS`), replacing the need for the three
  duplicated implementations across `utils.py`/`append_datetime.py`/`create_batches.py`
  going forward (those three still exist, untouched, for now).
- `src/migrate_to_db.py` — one-time migration, not in `cfg.pipeline`, run manually:
  `python main.py general.task=migrate_to_db +pipeline=[migrate_to_db]`. Reads
  `merged_blobs_tables_metadata_permanent.csv`, dedupes `cfg.state_list` (fixing the
  duplicate `KS` entry as a byproduct), infers `parent_code` for codes like `NC01`→`NC`,
  groups by `MasterRefID` for sample-level fields, derives `batches` from the
  `{location}_{date}` `BatchID` format, and upserts everything idempotently.
- Added `db_path` to `conf/paths/default.yaml`.

**Notable findings during migration:**
- The historical data contains `UsState=OH`, which wasn't in `cfg.state_list` — the
  migration logs a warning and adds it rather than crashing, directly demonstrating the
  fragility section 3.4 is meant to fix.
- 10 `BatchID` values are literally the string `nan_<date>` (a legacy
  `create_batches.py` bug stringifying a missing location) — stored with
  `location_code=NULL` rather than crashing or inventing a fake location.
- 226,461 CSV rows → 226,187 unique images (the ~274-row gap is duplicate blob names in
  the source CSV, consistent with this branch's own duplicate-image fix).

**Verified:** row counts match source CSV (minus explained duplicates), spot-checked
field values match, idempotent on re-run, `locations` has no duplicate codes and
correct `parent_code` values.

---

## Section 2 — Config-driven ingestion, upserting into the DB (done)

**Goal (plan sections 3.2/3.3, build order step 2):** rewrite `wir_table_generator.py`/
`wir_blob_data_generator.py` as generic, config-driven ingestion that upserts into the
DB instead of writing a new dated CSV each run; run alongside the old pipeline and diff
outputs until they match.

**What was built:**

- `conf/config.yaml` — new `sources:` list, one entry per table/container
  (`weedsimagerepo`, `wirimagerefs`, `wirmastermeta`, `wircovercropsmeta`,
  `wircropsmeta`, `wirweedsmeta`, `wirsoilsmeta`, `wirlogs`), each with `name`, `type`
  (`azure_table`/`azure_blob`), `key_ref` (which keys-file entry to use), and `entity`
  (how ingestion should route the pulled rows — see below). This is the list that
  drives both generators now; adding a new source is a new entry here, not new code.
  `keys/authorized_keys.yaml`'s existing (fragmented) schema was deliberately **not**
  restructured — lower risk, deferred.
- `src/ingestion_sources.py` — generic `AzureTableSource`/`AzureBlobSource` pull
  classes (logic extracted from the old exporter classes, not reimplemented) plus
  `resolve_table_credentials`/`resolve_blob_credentials` against the existing keys
  shape.
- `src/db/upsert.py` — three upsert functions, each entity-scoped:
  - `upsert_image_from_blob` — blob metadata → `images` (container, base_name,
    extension, size_mib, upload_datetime_utc). Owns only these columns.
  - `upsert_image_from_imageref` — `wirimagerefs` rows → `images` (master_ref_id,
    image_url, wirimagerefs_rowkey/timestamp), deriving `blob_name` from
    `basename(ImageURL)`. Owns only these columns — the two upserts can land in either
    order without clobbering each other.
  - `upsert_sample_attributes` — the five sample-attribute tables → staged as raw JSON
    in a new `raw_sample_attributes(source, master_ref_id, partition_key, row_key,
    source_timestamp, data, ingested_at)` table. **Deliberately not** written directly
    into `samples`, since their fields need coalesce-with-priority logic (section 3.5)
    — writing them straight into `samples` one table at a time would have silently
    overwritten good data with whichever table ran last.
- Rewired `src/wir_table_generator.py`/`src/wir_blob_data_generator.py` to iterate
  `cfg.sources` (instead of the raw keys dict / a hardcoded container name), still write
  the exact same CSVs as before, and additionally dispatch to the right upsert per
  source's `entity`.

**Notable finding:** `wirmastermeta` has no `MasterRefID` column at all —
`process_tables_analysis.py` renames its `RowKey` to `MasterRefID` downstream
(confirmed by reading that file). Handled with a `master_ref_key: RowKey` override on
that one source's config entry, rather than a hardcoded special case in the upsert
code.

**Verified:** CSV column headers identical to pre-change output. `process_blob_analysis.py`
and `process_tables_analysis.py` run unchanged against the new CSVs. DB populated as
expected (231,735 images, 21,583 sample stubs, 43,442 staged attribute rows at the
time). Idempotent — a second run of both generators leaves every count unchanged.

---

## Section 3 — Declarative merge (done)

**Goal (plan section 3.5, build order step 3):** replace the hand-written
`process_tables_analysis.py` merge/coalesce chain with a declarative config + one
generic function; validate against the current `merged_blobs_tables_metadata.csv`.

**What was built:**

- **Schema follow-up** (a TODO surfaced mid-session): the four `wir*_rowkey`/
  `wir*_timestamp` lineage pairs for the sample-attribute tables were on `images` in
  section 1's schema — but they're sample-scoped, not image-scoped (every image sharing
  a `MasterRefID` had identical values). Moved to `samples`; added
  `wirsoilsmeta_rowkey`/`timestamp` (missing before, since that table is new). Only
  `wirimagerefs_rowkey`/`timestamp` stayed on `images` (genuinely image-scoped).
  `migrate_to_db.py` updated to match; `build_locations`/`upsert_locations` moved out of
  it into `src/db/upsert.py` so both scripts share one implementation instead of two.
- `conf/config.yaml` — `coalesce_fields`: target field → ordered list of
  `{source, field}` fallbacks, first non-null wins. Single-source fields are just
  one-entry lists (no separate mechanism needed). Priorities copied exactly from the
  legacy fillna chain: `Species` ← WeedType → CoverCropSpecies → CropName; `Height`/
  `SizeClass` ← crops → weeds; `FlowerFruitOrSeeds` ← covercrops → weeds; etc.
- `src/merge_samples.py` — new task module (not yet in `cfg.pipeline`):
  1. Ensures `locations` covers every `UsState` actually seen in staged
     `wirmastermeta` rows (same "log and add, don't crash" pattern as section 1).
  2. For every `MasterRefID` in `raw_sample_attributes`, gathers each source's JSON row
     and applies `coalesce_fields` generically — no pandas merge, no join-key
     fragility, since each source's data is already keyed by `(source, master_ref_id)`.
  3. Applies the two legacy field-specific normalization rules explicitly (SizeClass
     `Large/Medium/Small/3/2/1` → `LARGE/MEDIUM/SMALL`; Species lowercased).
  4. Carries each source's `row_key`/`source_timestamp` onto the matching
     `samples.wir<source>_rowkey`/`_timestamp` columns.
  5. Upserts into `samples`.
  6. Recomputes `HasMatchingJpgAndRaw` over all of `images` (not just merged rows).
  7. Exports a validation CSV (`merged_blobs_tables_metadata_from_db.csv`) via one
     `images ⨝ samples` query, same column set as the legacy output, for diffing.

**Validation result (legacy `process_tables_analysis.py` run side-by-side, same
session):**
- Columns: identical (34/34).
- Every directly-comparable field (`Species`, `Height`, `SizeClass`,
  `CropTypeSecondary`, `FlowerFruitOrSeeds`, `UsState`, `PlantType`) across the 229,840
  rows common to both outputs: **zero mismatches**.
- Row counts differ (231,937 new vs 230,836 legacy) — fully accounted for, and the gap
  is two legacy bugs, not a regression in the new code:
  - Legacy has 996 duplicate rows (same blob `Name` twice) — a real fan-out bug from its
    outer join when a sample has multiple weed/crop attribute rows. The new pipeline
    can't produce duplicates by construction (`images.blob_name` is `UNIQUE`,
    `raw_sample_attributes` is one row per `(source, master_ref_id)`).
  - Legacy silently drops 980 orphan blobs (routed to a side CSV, excluded from the
    merged output) and 1,117 orphan `wirimagerefs` entries (dropped entirely, appear
    nowhere) — matches `21,970 + 980 + 1,117 = 231,937` exactly. The new pipeline keeps
    both as real rows with the missing side left `NULL`.
- `HasMatchingJpgAndRaw` disagreed on 699/229,840 rows (0.3%), all traced to one cause:
  legacy computes the flag *after* dropping orphaned blobs, so an ARW with no
  `wirimagerefs` entry vanishes before the match check runs, undercounting real
  JPG/ARW pairs. The new pipeline computes it from the full blob listing.
- Idempotent on re-run (`samples` count, a spot-checked `species` value, and the
  matching-image count all unchanged).

---

## What's next: section 4 onward

Per `refactor-plan.md` section 5's build order, remaining steps map to remaining plan
sections as follows:

### Section 4 — Batching against the DB (plan section 3.4)

**This is the step that actually unlocks adding new locations cleanly** — the original
motivation for the whole refactor.

**Problems to fix (from plan section 3.4, confirmed by reading `create_batches.py`/
`report.py` this session):**
- `create_batches.py`'s batch-folder regex
  (`^[A-Z]{2}_\d{4}-\d{2}-\d{2}$|^[A-Z]{2}\d{2}_\d{4}-\d{2}-\d{2}$`) hardcodes "2 letters,
  optionally + 2 digits" — a new location naming convention breaks this silently
  (folders just get skipped, no error).
- `NC01` handling is inconsistent across the codebase: excluded entirely in
  `plot_by_season.py`'s `plot_image_vs_raws_by_species_current_season`, renamed to `NC`
  in `report.py`'s `plot_cumulative_samples_species_by_year` and
  `plot_sample_species_state_distribution`, left alone in
  `plot_unique_masterrefids_by_state_and_planttype`.
- No single function every plot can call to backfill zero-count locations — each
  plot (`report.py`, `plot_by_season.py`) independently decides whether to do this,
  and they've drifted.

**Plan for this section:**
1. Add a generic `all_known_locations(conn)` function (in `src/db/` or a new
   `src/locations.py`) that returns every `code` from the `locations` table, with
   `parent_code` available for callers that want to roll `NC01` up into `NC` — replacing
   the three inconsistent ad hoc treatments with one explicit, callable decision point.
2. Generate the batch-folder regex from the DB's actual location codes at runtime
   (`locations` table, not a hand-written pattern) — a script/module reads
   `SELECT code FROM locations` and builds the regex dynamically, so a new location code
   with any shape just works.
3. Rewrite `create_batches.py`'s batching logic against `images`/`samples`/`locations`
   in the DB instead of the CSV + `find_most_recent_csv`, following the same
   parallel-validation approach as sections 2–3 (new task module, not wired into
   `cfg.pipeline` yet, diffed against the legacy batch folder output).
4. Fix the two standalone bugs in `create_batches.py` while touching this file (plan
   section 4's bug list): `row['Name'].replace('JPG', 'ARW')` is case-sensitive
   (breaks on lowercase `.jpg`), and `os.sched_getaffinity(0)` is Linux-only (crashes on
   macOS) — plus remove the unused `add_extra_number` dead code.

**Verification approach:** same pattern as sections 1–3 — run the new DB-driven
batching alongside the legacy `create_batches.py`, diff the resulting batch
folder/file assignments, and account for any discrepancies explicitly (as sections 2–3
did) rather than assuming a mismatch is a defect.

### Section 5 — Reporting/plotting (plan sections 3.4 cont'd, 3.6)

Lowest risk, highest visibility — a good checkpoint to confirm parity with current
dashboards before retiring old code.

1. Every plot in `report.py`/`plot_by_season.py` switches to `all_known_locations()`
   (from section 4) for backfilling zero-count locations, and to the DB instead of
   `find_most_recent_csv`.
2. Deduplicate the near-identical plotting classes (plan section 3.6) — e.g.
   `plot_unique_masterrefids_by_state_and_planttype` (all years) vs.
   `plot_unique_samples_state_plant_current_season` (current season) are the same
   matplotlib/seaborn logic differing only by a date filter. Extract one parameterized
   function (`plot_unique_samples(df, groupby_cols, palette, title, save_path)`), call it
   twice instead of maintaining two copies.
3. Fix the standalone bug in `image_inspection.py` while touching this area: `except
   Warning as e` doesn't catch real exceptions (`KeyError`/`IndexError`/
   `FileNotFoundError`) — should be `except Exception`. Also remove the unused `cv2`
   import.

### Section 6 — Retire old code

Only after sections 4–5's outputs are cross-checked against the legacy CSV-chain
pipeline (same diff-and-account approach used throughout). At that point:
- `wir_table_generator.py`/`wir_blob_data_generator.py` drop their CSV-writing side
  (CSV becomes an on-demand export *from* the DB, not the pipeline's internal state —
  plan section 3.2/6).
- `process_blob_analysis.py`/`process_tables_analysis.py` are replaced by
  `merge_samples.py` in `cfg.pipeline`.
- `create_batches.py` is replaced by its DB-driven rewrite from section 4.
- `find_most_recent_csv`/`find_most_recent_data_csv` in `utils/utils.py` become dead
  code and can be deleted.

### Cross-cutting items not yet scheduled to a specific section

These are called out in plan sections 3.7/3.8 and the standalone bug list (plan
section 4), not yet tackled:

- **Pipeline orchestration hardening (3.7):** `main.py` still calls `sys.exit(1)` on
  the first exception in `cfg.pipeline`, killing the whole run with no per-task status
  recorded. Needs either configurable skip/continue-vs-abort behavior, or adopting a
  lightweight orchestrator (Prefect/Dagster). Not addressed by sections 1–3 since those
  added new *standalone* task modules run manually, not through the pipeline loop yet.
- **Environment-specific hardcoding (3.8):** `conf/paths/default.yaml`'s
  `longterm_storage: /mnt/research-projects/r/raatwell/longterm_images3/field-batches`
  is still one person's path baked into version control. Needs to move to an env var or
  a gitignored `conf/paths/local.yaml` overlay.
- **Keys schema unification (3.3):** `keys/authorized_keys.yaml`'s two schemas (shared
  `account_url` + per-container overrides for blobs, vs. per-table `url` for tables) are
  still fragmented — sections 2–3 deliberately routed around this rather than rewriting
  the live secrets file. Worth doing once `create_batches.py` (the other consumer of
  the blob-key schema) is rewritten in section 4, so both consumers can move to one
  schema together.
- **Remaining standalone bugs (plan section 4)** not yet touched: the
  `append_datetime.py` `ref_df.shape[0]` crash when `ref_df is None`, and the
  `utils.utils.read_yaml`/`read_csv_as_df` pattern of catching any `Exception` and
  re-raising a generic `FileNotFoundError` (hides the real error). Neither blocks
  sections 4–6 and can be picked up whenever convenient, independent of this build
  order.
