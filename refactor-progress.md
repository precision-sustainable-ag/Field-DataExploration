# Refactor Progress

Tracks what's been implemented against `refactor-plan.md`'s build order (section 5).
All 6 build-order steps are done — `cfg.pipeline` now runs the DB-driven path live
(`merge_samples`, `append_datetime_db`, `report_db`; `create_batches_db` is built and
validated but deliberately left commented out, see section 6). This file is the
detailed record of what changed at each step and what's left as follow-up.

`wir_table_generator.py`/`wir_blob_data_generator.py` had their CSV-writing side
removed in section 6 — CSV is no longer the pipeline's internal state, only the DB is.
`image_inspection.py` (section 5) got one standalone bug fixed in place (a
non-catching `except Warning` clause) — doesn't change behavior on non-buggy inputs.
The old CSV-chain files (`process_blob_analysis.py`, `process_tables_analysis.py`,
`report.py`, `plot_by_season.py`, `create_batches.py`, `append_datetime.py`) were
initially kept on disk unreferenced (section 6's original choice), then actually
deleted once asked for explicitly — see "Deleting the old CSV-chain scripts" below.
The `create_batches.py` bug fixes from section 4 (case-sensitive JPG→ARW rename,
Linux-only `os.sched_getaffinity`) now live in `create_batches_db.py`, which absorbed
that file's logic before it was deleted.

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

## Section 4 — Batching against the DB (done)

**Goal (plan section 3.4, build order step 4):** rewrite `create_batches.py`'s batching
logic against `images`/`samples`/`locations` in the DB instead of the CSV +
`find_most_recent_csv`. This is the step that actually unlocks adding new locations
cleanly — the original motivation for the whole refactor.

**What was built:**

- `src/db/locations.py` — new module, three functions:
  - `all_known_locations(conn)` — every `(code, display_name, parent_code)` row from
    `locations`, as a list of `Location` namedtuples. Single source of truth for "what
    locations exist," replacing ad hoc reads of `cfg.state_list`.
  - `roll_up_to_parent(locations, code)` — maps a code to its `parent_code` if it has
    one (e.g. `NC01` → `NC`), else returns it unchanged. One explicit decision point,
    ready for section 5 to replace the three inconsistent `NC01` treatments in
    `report.py`/`plot_by_season.py` (excluded entirely in one plot, renamed to `NC` in
    two others, left alone elsewhere) — not yet wired in, since that's section 5's scope.
  - `batch_folder_regex(locations)` — builds the batch-folder regex
    (`^(?:TX02|TX01|NC01|...)_\d{4}-\d{2}-\d{2}$`) from the DB's actual location codes,
    instead of a hardcoded `^[A-Z]{2}_\d{4}-\d{2}-\d{2}$|^[A-Z]{2}\d{2}_\d{4}-\d{2}-\d{2}$`
    pattern that assumes every code is 2 letters or 2 letters + 2 digits. A new location
    code of any shape works as soon as it's a row in `locations`. (This pattern
    currently lives in `report.py`'s `PreprocessingCheck.analyze_directory` and
    `blob2nfs.py`'s `is_batch_folder` — switching those callers over is section 5/later,
    since `report.py`'s DB migration is explicitly scoped there.)
- `src/create_batches_db.py` — new task module, not yet in `cfg.pipeline`:
  - `DbBatchProcessor(CreateBatchProcessor)` — subclasses the existing processor and
    overrides only `read_and_convert_datetime` to source the batch DataFrame from one
    `images ⨝ samples` query instead of `find_most_recent_csv` + CSV read. Every other
    step (`split_datetime`, `preprocess_df`, `adjust_groups`, `filter_batched_data`, the
    azcopy move methods) is inherited unchanged, since they only ever operated on
    `self.df` by column name. The DB's `exif_datetime` is already normalized at ingest
    (`db/normalize.py`), so unlike the CSV path there's no `':'→'-'` regex pass needed.
  - `warn_on_unknown_batch_labels()` — new method, uses `batch_folder_regex` to flag any
    synthesized batch label whose location prefix isn't a known location code, replacing
    the silent-skip failure mode called out in the plan (a bad location code previously
    just vanished from batch folders with no error).
  - `main(cfg)` mirrors `create_batches.py`'s `main()`, swapping in `DbBatchProcessor`;
    run manually: `python main.py general.task=create_batches_db
    +pipeline=[create_batches_db]`.
- `src/create_batches.py` — fixed the two standalone bugs from plan section 4 while
  touching this file, plus removed dead code:
  - `row['Name'].replace('JPG', 'ARW')` was a case-sensitive substring replace; since
    most `.JPG`s in the real data are uppercase but ~97/123,934 are lowercase `.jpg`,
    those silently kept their original (wrong) extension in the synthesized RAW path.
    Replaced with a new `jpg_name_to_arw()` helper using
    `re.sub(r'\.jpg$', '.ARW', name, flags=re.IGNORECASE)` — always produces the
    (verified) canonical uppercase `.ARW`, regardless of the JPG's case.
  - `os.sched_getaffinity(0)` is Linux-only and crashes on macOS; guarded with
    `hasattr(os, "sched_getaffinity")`, falling back to `os.cpu_count()`, and floored at
    1 worker so a <3-core machine doesn't get `ThreadPoolExecutor(max_workers=0)`.
  - Removed the unused `add_extra_number` method (confirmed no references anywhere in
    the codebase before deleting).

**Verified (locally, without live Azure — no credentials needed since none of this
touches blob storage):**
- Ran the legacy CSV-driven `CreateBatchProcessor` and the new `DbBatchProcessor` side
  by side against today's data (`CreateBatchProcessor` naturally fell back to the older
  `merged_blobs_tables_metadata_permanent.csv`, since today's fresh
  `merged_blobs_tables_metadata.csv` lacks `CameraInfo_DateTime` — `append_datetime` isn't
  in `cfg.pipeline` this session — which is itself the existing fallback behavior, not
  something this change touched).
- Post `preprocess_df`/`adjust_groups`: legacy produced 101,333 batchable images, DB
  produced 106,559. **Zero** images present in legacy but missing from the DB version —
  the new pipeline is a strict superset here.
- All 5,226 "DB-only" images traced to two already-documented staleness bugs in the old
  permanent CSV snapshot the legacy path fell back to, not to anything new in this
  change: 3,971 have a stale per-row `HasMatchingJpgAndRaw=False` (the same class of
  bug section 3 documented at 699/229,840 rows — the DB recomputes this flag from the
  full blob listing, so it doesn't inherit the staleness); 1,254 have a stale/missing
  `UsState` in that older CSV snapshot that the DB's fresher merge has since filled in
  (1 row unaccounted for, negligible).
- Of the 101,333 batch-folder assignments in common, only 15 differ (all in two
  state/date groups: `TX_2025-04-24`, `NC_2022-07-18`) — and in both cases the cause is
  mechanical: the DB version correctly includes an earlier-timestamped image that
  legacy's stale `HasMatchingJpgAndRaw` flag was wrongly excluding, which shifts the
  3-hour-bucket ordinal (`SubBatchIndex`) for the group by one. Not a defect — the new
  numbering reflects a more complete, more correct input set.
- `warn_on_unknown_batch_labels()` logged zero warnings against the real DB — every
  synthesized batch label matched a known location code, confirming the dynamic regex
  works without false positives on real data.
- `batch_folder_regex` spot-checked against `all_known_locations()` output: correctly
  matches `TX_2024-07-07`, `TX01_2024-07-07`, `NC01_2023-05-01`, `DV_2024-07-07` and
  rejects an unknown code (`ZZ_2024-01-01`).

Not run against live Azure in this session (no credentials in this environment) — the
`FieldBatchLister`/azcopy move methods are unchanged from `create_batches.py` and were
not touched by this section, so they carry no new risk beyond what already runs in
production today.

**Known gap, deferred to section 6:** `DbBatchProcessor` computes batch assignments
in memory only, same as the legacy processor — it doesn't write anything back to
`images.batch_id` or insert new `batches` rows. See section 6 below for the plan to
close that once this becomes the live batching path.

---

## Section 5 — Reporting/plotting (done)

**Goal (plan sections 3.4 cont'd, 3.6, build order step 5):** every plot in
`report.py`/`plot_by_season.py` switches to `all_known_locations()`/`roll_up_to_parent()`
(section 4) and the DB instead of `find_most_recent_csv`; deduplicate the near-identical
plotting classes; fix the standalone `image_inspection.py` bug. Lowest risk, highest
visibility — a checkpoint to confirm parity with current dashboards before retiring old
code. Followed the same pattern as section 4: new modules, not wired into `cfg.pipeline`
yet, validated against the legacy output.

**What was built:**

- `src/db/reporting.py` — `load_report_dataframe(conn)`, one `images ⨝ samples` query
  covering every column `report.py`/`plot_by_season.py`/`image_inspection.py` actually
  use (confirmed by grepping all three files for column references first). Unlike the
  CSV chain, `CameraInfo_DateTime` is always populated here — it's normalized at ingest
  (`db/normalize.py`), not bolted on later by a separate `append_datetime` run.
- `src/plotting.py` — `plot_unique_samples(df, hue_col, palette, title, save_path,
  known_states, hue_order)`, the parameterized function plan section 3.6 asks for.
  Replaces `report.py`'s `plot_unique_masterrefids_by_state_and_planttype` and
  `plot_by_season.py`'s `plot_unique_samples_state_plant_current_season` — identical
  matplotlib/seaborn logic, differing only in `hue_col`/palette/title/save path. Also
  fixes an inconsistency along the way: only the current-season version backfilled
  missing states before; the shared function always does.
- `src/report_db.py` — new task module, not yet in `cfg.pipeline`:
  - `BatchReportDb` ports every `BatchReport` method (`write_missing_raws`,
    `num_uploads_selected_days_by_state`, and all seven plots) onto
    `load_report_dataframe()`. `UsState` is rolled up via `roll_up_to_parent()` **once**,
    immediately after loading — this single line replaces the three inconsistent NC01
    treatments the plan called out (left alone in
    `plot_unique_masterrefids_by_state_and_planttype`, renamed to `NC` in
    `plot_cumulative_samples_species_by_year`/`plot_sample_species_state_distribution`),
    since every downstream method just sees `NC` and never has to know `NC01` existed.
    `plot_image_vs_raws_by_species` and `plot_num_samples_usstate` gained backfill via
    `known_states` (from `all_known_locations()`) — legacy never backfilled either.
  - `PreprocessingCheckDb` subclasses `report.py`'s `PreprocessingCheck` (same pattern as
    section 4's `DbBatchProcessor`), overriding only `analyze_directory` to build its
    folder-matching pattern from `batch_folder_regex(all_known_locations(conn))` instead
    of the hardcoded `^[A-Z]{2}_\d{4}-\d{2}-\d{2}$|^[A-Z]{2}\d{2}_\d{4}-\d{2}-\d{2}$`.
  - Run manually: `python main.py general.task=report_db +pipeline=[report_db]`.
- `src/plot_by_season_db.py` — new task module, not yet in `cfg.pipeline`:
  `PlotsBySeasonDb` ports every `PlotsBySeason` method the same way. NC01 exclusion in
  `plot_image_vs_raws_by_species_current_season` (`~df['UsState'].isin(['NC01'])`, the
  third of the three inconsistent treatments) is gone — there's no more `NC01` in the
  data by the time this method runs, it's already `NC`. Its convoluted duplicate-rows
  backfill trick for missing state/extension combinations is replaced with the same
  explicit missing-row construction `report_db.py`'s `plot_image_vs_raws_by_species`
  uses. Run manually: `python main.py general.task=plot_by_season_db
  +pipeline=[plot_by_season_db]`.
- `src/image_inspection.py` — fixed directly (not part of the DB migration, this file
  isn't a "plot" the plan asked to move): `except Warning as e` didn't catch real
  exceptions (`KeyError`/`IndexError`/`FileNotFoundError`), so QA plotting didn't
  actually fail gracefully as intended — changed to `except Exception as e`. Removed the
  unused `cv2` import (confirmed no other use in the file before removing).

**Verified (locally, against the real local DB and CSVs):**
- `BatchReportDb`'s `MasterRefID` counts by `UsState`+`PlantType` and by `UsState` alone:
  compared against the legacy `BatchReport` (same permanent-CSV-fallback situation as
  section 4, since today's fresh dated CSV still lacks `CameraInfo_DateTime` this
  session) with the same `roll_up_to_parent()` applied to both sides for a fair
  comparison. Every state/plant-type combination matched exactly or the DB version was
  higher, never lower — consistent with section 4's finding that the DB recomputes
  `HasMatchingJpgAndRaw` correctly where the old CSV snapshot has stale per-row values.
  `OH` was the largest gap (104 legacy vs. 305 DB) — traced precisely: the permanent CSV
  has 4,090 `OH` rows, 305 distinct `MasterRefID`s total, but 2,010/4,090 rows have a
  stale `HasMatchingJpgAndRaw=False`, undercounting legacy's per-state total down to 104.
  305 matches the DB's recomputed count exactly.
- `PreprocessingCheckDb`'s dynamically-generated regex vs. the legacy hardcoded one,
  run against the real `longterm_storage` NFS directory (463 folders, no live Azure
  needed): **identical** — 461 folders matched by both, zero folders only-in-legacy or
  only-in-new.
- Full smoke test: every method on `BatchReportDb`, `PlotsBySeasonDb`, and
  `PreprocessingCheckDb` run end-to-end (outputs redirected to a scratch directory, nothing
  written to the real `report/` folder) — no exceptions, all 32 expected plot/CSV files
  produced. `PlotsBySeasonDb`'s current-season filter correctly produced 0 rows this
  session — confirmed this reproduces identically against the legacy `PlotsBySeason`, so
  it isn't a section 5 regression, but it isn't purely a date-environment artifact either:
  see the `images.exif_datetime` backfill below, found while digging into this — it was
  a real, independent cause of the same symptom, now closed.

---

## EXIF datetime backfill (done, gap found during section 5)

**Goal:** close the gap found while investigating section 5's empty current-season
plots — `images.exif_datetime` was only ever populated once, by section 1's historical
migration; nothing added in sections 2–5 kept it current for newly-ingested images,
since the only code that ever extracts it (`append_datetime.py`) isn't in
`cfg.pipeline` and only ever wrote its results to CSVs, never the DB.

**What was built:**

- `src/db/upsert.py` — `update_image_exif_datetime(conn, {blob_name: datetime})`, a
  small bulk `UPDATE`, added alongside the existing entity-scoped upserts.
- `src/append_datetime_db.py` — new task module, not yet in `cfg.pipeline`. Three
  passes over `images` where `exif_datetime IS NULL`, cheapest first:
  1. **Stem lookup** — an image inherits `exif_datetime` from a sibling (same
     `base_name`) JPG that already has one. Free, no network.
  2. **EXIF download** — any JPG still missing it gets downloaded via `azcopy`
     (`weedsimagerepo` credentials) and its EXIF `DateTimeOriginal` extracted, same
     mechanism `append_datetime.py` uses, normalized through `db/normalize.py`'s
     canonical `normalize_datetime()` instead of `append_datetime.py`'s own
     duplicate implementation (`normalize_datetime_column`) — one of the three
     duplicated datetime normalizers section 1 flagged, now down to two once
     `append_datetime.py` itself is retired. Optionally capped per run via
     `+exif_download_limit=N`, so a run doesn't have to download the entire backlog
     at once.
  3. **Second stem pass** — re-checks the still-missing set against datetimes
     *just* downloaded in step 2, so an ARW whose sibling JPG was also missing
     before this run still gets filled. `append_datetime.py` doesn't do this: its
     single stem-fill call runs *before* the download step, so a pair that started
     out both-missing only ever gets the JPG side backfilled there — a small
     correctness improvement over legacy, not just a port.
  - Run manually: `python main.py general.task=append_datetime_db
    +pipeline=[append_datetime_db]`, optionally with `+exif_download_limit=N`.

**Verified:** ran against a scratch copy of the real DB (never the live one — confirmed
untouched after, still 12,533 `NULL` rows) with `exif_download_limit=5`:
- 12,533 images missing `exif_datetime` at the start.
- Stem lookup alone filled **9,328** — the large majority, for free.
- 5 JPGs downloaded live via `azcopy` and had their EXIF extracted successfully (0
  failures) — proves the live path works end-to-end, not just the stem-fill shortcut.
- Second stem pass picked up 2 more (ARW siblings of those 5 JPGs).
- Total: 9,335/12,533 filled in one capped run; 3,198 remain (963 of those are JPGs
  eligible for download, the rest are images with no JPG sibling to inherit from and
  no `image_url` recorded to download from — same ceiling legacy would hit).
- Re-ran on the already-partially-filled scratch copy to confirm idempotency: second
  run correctly found only the reduced remaining set, filled 5 more via download, 0
  redundant work on the already-filled 9,335.

**Not yet run against the real DB.** This was validated on a disposable copy on
purpose — an unlimited real run means ~1,000+ live `azcopy` downloads, which takes
real time and bandwidth against production blob storage. Whether to run it (and
whether to cap it) is a call for whoever runs the pipeline, not something to do
silently as a side effect of building the module.

---

## Section 6 — Retire old code (done)

**Goal (build order step 6):** cross-check sections 4–5's outputs against the legacy
CSV-chain pipeline (done, in each section's own write-up), then actually cut
`cfg.pipeline` over to the DB-driven path.

**Two decisions made explicitly with the user before touching the live pipeline
config, since this is the step that changes what actually runs automatically:**
1. `create_batches_db` stays **commented out** in `cfg.pipeline`, same as
   `create_batches` was — it moves real blobs via `azcopy`, and enabling it wasn't
   asked for. `append_datetime_db` **is** enabled — its live-download step is capped
   per run via `+exif_download_limit=N` if ever needed, and the backlog is already
   mostly drained (see the EXIF backfill section above).
2. The old CSV-chain files (`report.py`, `process_blob_analysis.py`,
   `process_tables_analysis.py`, `create_batches.py`, `plot_by_season.py`,
   `append_datetime.py`) were initially kept on disk, just unreferenced by
   `cfg.pipeline` — not deleted, to preserve a rollback path. **Superseded shortly
   after** — see "Deleting the old CSV-chain scripts" below, once the DB-driven path
   had run live and the rollback path was no longer wanted.

**What was built:**

- `conf/config.yaml` — `pipeline:` now reads:
  ```yaml
  pipeline:
      - wir_table_generator
      - wir_blob_data_generator
      - merge_samples
      - append_datetime_db
      - report_db
      # - plot_by_season_db
      # - image_inspection
      # - create_batches_db
  ```
  `process_blob_analysis`/`process_tables_analysis` → `merge_samples`; `report` →
  `report_db`; `append_datetime` (previously commented out) → `append_datetime_db`,
  now enabled. `plot_by_season`/`image_inspection`/`create_batches` stay commented,
  same as before, with their `_db` names substituted where one exists.
- `src/wir_table_generator.py`/`src/wir_blob_data_generator.py` — dropped the
  CSV-writing side (`to_csv`, the `tablesdir`/`blobsdir` directory creation, the
  now-unused `pandas`/`Path` imports). Renamed `get_table_csv`/`get_blob_csv` →
  `pull_and_upsert` on both exporter classes, since that's what they actually do now.
  The DB upsert logic itself is untouched.
- **`images.batch_id` gets populated** (closes the gap from section 4's write-up):
  - `src/db/upsert.py` — moved `upsert_batches` here from `migrate_to_db.py` (same
    move `build_locations`/`upsert_locations` made in section 3, for the same reason:
    one implementation instead of two now that `create_batches_db.py` needs it too).
    Added `update_image_batch_id(conn, {blob_name: batch_id})`, a small bulk update
    alongside the existing `update_image_exif_datetime`.
  - `src/create_batches_db.py` — new `DbBatchProcessor.persist_batches()`: derives a
    `BatchID` column (`{UsState}_{date}`, same format `adjust_groups()` already builds
    into the `batches` folder-path column), calls the now-shared `upsert_batches` to
    get a `batch_label -> id` map, and sets `images.batch_id` for every row in the
    *full* computed assignment — not just whatever `filter_batched_data` later decides
    still needs an `azcopy` copy — so `images.batch_id` reflects batch membership
    regardless of whether the physical copy has happened yet. Wired into `main()`
    right after `adjust_groups()`/`warn_on_unknown_batch_labels()`, before
    `filter_batched_data()`.
  - `src/migrate_to_db.py` — now imports `upsert_batches` from `db/upsert.py` instead
    of defining its own copy.

**Verified (locally, against a scratch copy of the real DB — never the live one):**
- `persist_batches()`: ran `DbBatchProcessor` through `adjust_groups()` and
  `persist_batches()` (skipping `filter_batched_data`/the `azcopy` move, so nothing
  live-Azure happens). `batches` went 733 → 739 rows; `images.batch_id` went from
  219,404 already-set (from section 1's historical migration, which did set it from
  the old CSV's `BatchID` column — the actual gap was only for images ingested
  *since* that migration) to 219,875 (+471). The increase is smaller than the 12,533
  images missing `exif_datetime` because `preprocess_df()` requires a non-null
  `CameraInfo_DateTime` to be batchable at all — the EXIF and batch-id gaps are the
  same underlying gap wearing two hats, and closing the EXIF one (previous session)
  is what let most of these 471 become batchable in the first place.
  - Idempotency: re-ran on the same scratch copy — `739`/`219,875` both unchanged,
    zero redundant work.
  - Cross-checked every `(image, batch)` pair's location against the image's own
    `samples.location_code`: **1** mismatch out of ~471 newly-assigned
    (`DSC00302.JPG`, assigned to an `MD` batch but linked to a `TX` sample) — a
    pre-existing data inconsistency (a generic default camera filename, the same
    class of issue the duplicate-image-name fix on this branch already deals with),
    not something `persist_batches()` introduces. Not chased further given the
    negligible rate; flagged here rather than silently ignored.
- Ingestion CSV-removal: confirmed `TableExporter`/`BlobMetricExporter` construct
  with no leftover `tables_dir`/`blobs_dir` attributes, and smoke-tested the actual
  upsert call paths (`_upsert_entities`, `upsert_image_from_blob`) against a throwaway
  DB with synthetic rows — both still upsert correctly. Didn't re-run a live Azure
  pull for this, since the upsert functions themselves were already validated in
  sections 2–3 and the only change here was removing the `to_csv` calls around them.

**Originally not done, done shortly after:** the old CSV-chain files were deliberately
kept on disk per the decisions above — see "Deleting the old CSV-chain scripts" below
for when that changed.

---

## Bug fix: PlantType palette crash (found via live `python main.py` run)

Running the newly-live pipeline for real surfaced a genuine crash:
`plot_unique_masterrefids_by_state_and_planttype` raised `ValueError: The palette
dictionary is missing keys: {'SOILS'}`. `samples.plant_type` has two values the
hardcoded 3-color palette (`WEEDS`/`COVERCROPS`/`CASHCROPS`) never anticipated:
`SOILS` (11 samples) and `COTTONFLOWERS` (15). **Confirmed pre-existing, not a
regression**: `report.py` has the exact same hardcoded dict at the exact same 3 call
sites (`plot_unique_masterrefids_by_state_and_planttype`,
`plot_sample_species_distribution`, `plot_sample_species_state_distribution`) — it
would have crashed identically if ever run against data containing a SOILS/
COTTONFLOWERS sample with a matching JPG/ARW pair.

**Fix:** `src/plotting.py` gained `ensure_palette_covers(palette, values)` —
extends a fixed hue palette with fallback colors (`seaborn`'s `tab10`) for any value
actually present in the data but missing from the palette, instead of `seaborn`
hard-crashing. Used inside `plot_unique_samples()` (covers both
`report_db.py`'s and `plot_by_season_db.py`'s calls through it) and at
`report_db.py`'s two remaining direct `sns.barplot(..., palette=self.planttype_palette)`
call sites. Verified against the real DB: reproduces the exact `SOILS` warning
(now a log line, not a crash) and all three plots complete.

## Deleting the old CSV-chain scripts (done)

Section 6 originally kept the old files on disk, unreferenced, as a rollback path.
Once the DB-driven pipeline had actually run live end-to-end (including the palette
fix above), that rollback path was no longer wanted — asked for explicitly, so this
went further than the plan's original text.

**Deleted:** `process_blob_analysis.py`, `process_tables_analysis.py`, `report.py`,
`plot_by_season.py`, `create_batches.py`, `append_datetime.py`. `image_inspection.py`
was **not** deleted — it has no DB-driven replacement (section 5 only fixed its bug in
place), so deleting it would have removed a capability, not retired a superseded one.

**Two real code dependencies had to be resolved first, not just config changes:**
- `report_db.py` did `from report import PreprocessingCheck` (subclassed it as
  `PreprocessingCheckDb`, only overriding `analyze_directory`). Inlined
  `PreprocessingCheck`'s remaining methods (`_count_images`, `_get_folder_metadata`,
  `save_to_csv`, `plot_batches_per_week`) directly into `PreprocessingCheckDb`, which
  is now a standalone class.
- `create_batches_db.py` did `from create_batches import CreateBatchProcessor,
  FieldBatchLister` (subclassed `CreateBatchProcessor` as `DbBatchProcessor`, used
  `FieldBatchLister` directly). Inlined `FieldBatchLister`, the module-level
  `round_down_to_nearest_3_hours`/`jpg_name_to_arw` helpers, and every
  `CreateBatchProcessor` method `DbBatchProcessor` used (`config_keys`,
  `split_datetime`, `preprocess_df`, `adjust_groups`, `filter_batched_data`,
  `move_from_weeedsimagerepo2fieldbatches`, `process_df`, `process_df_concurrently`) —
  `DbBatchProcessor` is now standalone too. This is also where the section 4 bug
  fixes (case-sensitive JPG→ARW, Linux-only `os.sched_getaffinity`) ended up living.
- `blob2nfs.py` (pre-existing WIP script, outside the refactor's scope but a real
  consumer) did `from create_batches import FieldBatchLister` — repointed at
  `create_batches_db`. It also imported `find_most_recent_csv` from `utils/utils.py`
  without ever calling it (a pre-existing unused import) — dropped.

**`utils/utils.py` cleanup**, now that the old files are actually gone:
- `find_most_recent_csv` — genuinely dead (its only caller was `create_batches.py`) —
  deleted.
- `find_most_recent_data_csv` — **kept**, `image_inspection.py` still uses it.
- `convert_datetime`/`is_wrong_format` — found to be dead code *before* even reaching
  this cleanup (`append_datetime.py` imported `convert_datetime` but never called it)
  — deleted along with a duplicate `import re` line noticed in the same pass.

**Verified:** every remaining file compiles and imports (including `blob2nfs.py`);
grepped for any dangling `from report import` / `from create_batches import` /
`find_most_recent_csv` references — none found. Re-ran the full functional smoke
test against the real DB (read-only — no live Azure): `PreprocessingCheckDb` matched
**461** folders, identical to the pre-deletion baseline; every `BatchReportDb`,
`PlotsBySeasonDb`, and `DbBatchProcessor` method ran with no exceptions
(`DbBatchProcessor` carried 107,075 rows through `adjust_groups()`); 34 output files
generated (vs. 32 in section 5's original baseline — the 2 extra are the
SOILS/COTTONFLOWERS per-state plots that now complete instead of crashing partway
through, from the palette fix above, not a regression from this deletion).

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
  the live secrets file. `create_batches_db.py` (the DB-driven consumer of the blob-key
  schema, now the only one — `create_batches.py` is deleted) reads it the same
  unrestructured way; still worth unifying whenever convenient.
- **Remaining standalone bugs (plan section 4):** the `ref_df.shape[0]` crash when
  `ref_df is None` was specific to `append_datetime.py`'s CSV-merge logic, which no
  longer exists (file deleted, replaced by `append_datetime_db.py`'s DB-based
  approach, which has no equivalent code path to crash) — **moot, not carried
  forward**. Still open: `utils.utils.read_yaml`/`read_csv_as_df`'s pattern of
  catching any `Exception` and re-raising a generic `FileNotFoundError` (hides the
  real error). Doesn't block anything, pick up whenever convenient.
