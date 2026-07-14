# Field-DataExploration Refactor Plan

## 1. What this repo is actually for

An ETL + QA pipeline for a multi-state agricultural image dataset (weeds, crops, cover
crops). It pulls raw images and metadata out of Azure Blob/Table Storage, merges them
into a clean master table, organizes images into dated "batches" per location, and
produces health/QA reports and plots. This is the job to preserve — everything below is
about how to do that job without the current fragility.

## 2. Why a rewrite instead of patches

The current implementation works, but nearly every extension point (new location, new
metadata table, new plot) requires editing code in multiple places because there's no
shared schema or config layer — state lists, datetime parsing, and merge logic are each
implemented two or three times with small inconsistencies. Patching individual bugs
won't fix that; the shape of the code has to change.

---

## 3. What needs to be done

### 3.1 Define an explicit data model

**Why:** Right now the "schema" is implicit — it's whatever columns survive a chain of
`pd.merge` calls across `process_blob_analysis.py` and `process_tables_analysis.py`.
Nothing enforces consistency, and every new source table means editing that chain by
hand.

**How:** Define four entities up front, independent of any storage format:

- **Image** — one physical file (JPG or ARW): blob path, extension, size, blob creation
  time, EXIF capture time, container.
- **Sample** (currently `MasterRefID`) — one field observation, possibly backed by
  multiple images (JPG+RAW pair): species, plant type, growth stage, size class, ground
  cover, etc.
- **Location** — a real entity with `code`, `display_name`, and optional
  `parent_location`, instead of a hardcoded string list. This is what lets `NC01` be "a
  location whose parent is NC" instead of a special-cased string comparison.
- **Batch** — a derived grouping of samples by location + date + time window. Not
  source-of-truth data; always recomputable from Images/Samples.

Every downstream step (merges, batching, plotting) consumes these entities, not raw CSV
column names.

### 3.2 Replace the CSV-passing pipeline with a real datastore

**Why:** Every stage today reads "the most recent CSV" (found via a dated-folder regex
in `find_most_recent_csv`), mutates it, and writes a new CSV for the next stage to
rediscover. This is why datetime normalization was implemented three different ways
(`utils.convert_datetime`, `create_batches.replace_date_format`,
`append_datetime.normalize_datetime_column`) — each stage re-solved the same problem
independently instead of relying on data that was already normalized once.

**How:**
- Use SQLite (or Postgres if multiple people/processes need concurrent access) as the
  single source of truth for Image, Sample, Location, Batch.
- Ingestion steps (`wir_table_generator`, `wir_blob_data_generator`) **upsert** into it
  instead of writing a new dated CSV each run.
- Reporting and plotting query the DB directly instead of calling
  `find_most_recent_csv`.
- CSV exports become a derived output *generated from* the DB when needed (e.g. for
  sharing), not the pipeline's internal state.
- Datetime and other normalization happens once, at ingest, into the DB's canonical
  format — eliminating the three-implementation duplication entirely.

### 3.3 Make ingestion sources config-driven

**Why:** `wir_table_generator.py` expects each table to have its own `url`, while
`wir_blob_data_generator.py` and `create_batches.py` expect a shared `account_url`
under `blobs` — two different, undocumented key schemas. Adding a new table or blob
container today means writing new code, not just new config, and getting the keys file
right requires reading source code to find out which pattern applies.

**How:** One canonical schema for "a data source," declared in config:

```yaml
sources:
  - name: wirmastermeta
    type: azure_table
    key_ref: wirmastermeta
    entity: sample_master
  - name: weedsimagerepo
    type: azure_blob
    key_ref: weedsimagerepo
    entity: image
```

One generic `AzureTableSource` class and one generic `AzureBlobSource` class read this
config and know how to pull + upsert into the DB. Adding a new table or container going
forward is a new list entry, not new code. The `keys/authorized_keys.yaml.template`
gets rewritten to match this schema exactly, closing the current gap where the template
doesn't actually match what the code reads (missing `write_sas_token` for
`field-batches`, for example).

### 3.4 Centralize location handling

**Why:** This is the direct blocker for "add new locations," which prompted this whole
refactor. Today, state handling is scattered:
- `conf/config.yaml`'s `state_list` is a hardcoded list (with a duplicate `KS` entry).
- Some plots backfill missing states from `state_list`; others silently omit states with
  zero data.
- `NC01` is handled three inconsistent ways: excluded entirely in one plot, renamed to
  `NC` in two others, left alone elsewhere.
- The batch-folder regex in `report.py`
  (`^[A-Z]{2}_\d{4}-\d{2}-\d{2}$|^[A-Z]{2}\d{2}_\d{4}-\d{2}-\d{2}$`) assumes every
  location code is exactly 2 letters or 2 letters + 2 digits — a new location naming
  convention breaks this silently (folders just get skipped, no error).

**How:**
- Move locations into the Location table/config described in 3.1, with explicit
  parent/child relationships instead of string prefix matching.
- Generate the batch-folder regex from the configured location codes at runtime,
  instead of hand-writing a pattern that assumes a specific code format.
- Write one function, `all_known_locations()`, that every plot uses to backfill
  zero-count locations — instead of each plot deciding independently whether to do this.

### 3.5 Declarative merge/coalesce logic

**Why:** `process_tables_analysis.py` hand-writes the merge chain: four `pd.merge`
calls, then a long sequence of `fillna`-then-drop pairs to coalesce
`Height_01`/`Height_02` into `Height`, `SizeClass_01`/`SizeClass_02` into `SizeClass`,
etc., ending in a hardcoded `cols = [...]` list of ~30 column names. Adding a new
metadata table with an overlapping field (say, another table with its own `Height`
column) means manually inserting new merge/coalesce lines and remembering to update the
final column list.

**How:** Define coalesce rules declaratively:

```yaml
coalesce_fields:
  Species: [WeedType, CoverCropSpecies, CropType]
  Height: [Height_01, Height_02]
  SizeClass: [SizeClass_01, SizeClass_02]
```

One generic function applies these rules over however many source tables are configured
in 3.3. The final output schema is derived from the entity definition in 3.1, not a
manually maintained list.

### 3.6 Deduplicate plotting code

**Why:** `report.py` and `plot_by_season.py` contain near-duplicate plotting classes —
e.g. `plot_unique_masterrefids_by_state_and_planttype` (all years) versus
`plot_unique_samples_state_plant_current_season` (current season) — copy-pasted
matplotlib/seaborn code differing only by a date filter. They've already drifted (one
backfills missing states, the other doesn't), and every styling change has to be made
in multiple places.

**How:** Extract shared plotting logic into parameterized functions:

```python
def plot_unique_samples(df, groupby_cols, palette, title, save_path): ...
```

Call the same function twice — once with the full dataset, once with a
current-season-filtered dataset — instead of maintaining two classes.

### 3.7 Harden pipeline orchestration

**Why:** `main.py` runs every task in `cfg.pipeline` in sequence and calls
`sys.exit(1)` on the first exception, killing the entire run — there's no way to skip a
failed stage and let independent later stages (e.g. plotting) still run, and no
per-task status is recorded anywhere but the log file.

**How:** Either extend `main.py` with configurable per-task failure handling
(skip/continue vs. abort, recorded per run), or adopt a lightweight orchestrator
(Prefect/Dagster) that gives this along with retries and a run history UI, since the
pipeline is already a linear DAG of named tasks and would map onto either directly.

### 3.8 Remove environment-specific hardcoding

**Why:** `conf/paths/default.yaml` hardcodes
`/mnt/research-projects/r/raatwell/longterm_images3/field-batches` — one person's
username baked into a path in version control. This breaks for anyone else and for any
new machine, and it's exactly the kind of thing that needs to change every time the
"location" of this pipeline changes (which is presumably part of what's motivating the
refactor).

**How:** Move machine-specific paths to environment variables or a local,
gitignored override file (`conf/paths/local.yaml`), loaded on top of the checked-in
defaults.

---

## 4. Bugs to fix regardless of refactor scope

These are worth fixing immediately, independent of the rewrite timeline, since some are
currently either broken outright or silently wrong:

| Location | Issue |
|---|---|
| `keys/authorized_keys.yaml.template` | Missing `write_sas_token` for `field-batches`, which `create_batches.py` requires — following the template as written causes a `KeyError`. |
| `append_datetime.py` | `load_and_prepare_dataframe` computes `ref_df.shape[0]` unconditionally even when `ref_df` is `None` (first run, no permanent CSV yet) → `AttributeError`. |
| `image_inspection.py` | `except Warning as e` doesn't catch normal exceptions (`KeyError`, `IndexError`, `FileNotFoundError`), so QA plotting doesn't actually fail gracefully as intended. Also imports `cv2` but never uses it. |
| `create_batches.py` | `row['Name'].replace('JPG', 'ARW')` is case-sensitive; lowercase `.jpg` filenames won't be renamed correctly. |
| `create_batches.py`, `wir_blob_data_generator.py` | `os.sched_getaffinity(0)` is Linux-only; will crash on macOS. |
| `create_batches.py` | `add_extra_number` method is unused dead code. |
| `append_datetime.py` | `lts_csv` path is computed but the write to it is commented out — a half-built feature left in place. |
| `conf/config.yaml` | `state_list` contains `KS` twice. |
| `utils/utils.py` | `read_yaml`/`read_csv_as_df` catch any `Exception` and re-raise a generic `FileNotFoundError`, hiding the real error (bad YAML syntax, permissions, etc.) behind a misleading message. |

---

## 5. Build order

The goal is to have a working, comparable pipeline at every step rather than a
big-bang rewrite with a long integration risk at the end.

1. **Schema + migration** — Stand up the SQLite schema for Image/Sample/Location/Batch,
   and write a one-time script to load the existing CSVs into it, so historical data
   isn't lost.
2. **Config-driven ingestion** — Rewrite `wir_table_generator` and
   `wir_blob_data_generator` as the generic `AzureTableSource`/`AzureBlobSource`
   classes from 3.3, upserting into the DB. Run alongside the old pipeline and diff
   outputs until they match.
3. **Declarative merge** — Rewrite the `process_blob_analysis` /
   `process_tables_analysis` merge as a DB query or declarative transform per 3.5;
   validate against the current `merged_blobs_tables_metadata.csv` output.
4. **Batching against the DB** — Rewrite `create_batches.py` using the generic Location
   model from 3.4. This is the step that actually unlocks adding new locations cleanly.
5. **Reporting/plotting last** — Lowest risk, highest visibility; good checkpoint to
   confirm parity with current dashboards before retiring old code.
6. **Retire old code** — Only after outputs from steps 2–5 are cross-checked against
   the legacy CSV-chain pipeline.

---

## 6. What stays the same

Worth being explicit about scope: the *sources* (Azure Table/Blob Storage), the
*outputs* (merged metadata table, organized batch folders, QA plots/reports), and the
overall *pipeline stages* (ingest → merge → batch → report) don't need to change. This
is a refactor of how those stages are implemented and connected, not a change to what
the system does or where the data lives.
