# Design: Field data dashboard

## Purpose

A web page the team can open to see what Field-AgIR data we have, what we
don't have, and what still needs processing, broken down by plant type,
species, location and season.

It answers questions like:

- How many usable palmer amaranth images do we have from Texas?
- Which species × location combinations have few or no images?
- What can still be processed into cutouts, and where is it?
- Which images can never be used, and why?

**It is not a pipeline monitor.** It doesn't track job runs, throughput or
model performance; the weekly Slack report
([`src/notify_slack.py`](../src/notify_slack.py)) covers pipeline status. If an
idea for a page is about how the pipeline is running rather than about the
data, it's out of scope.

A working example is in [`dashboard/app.py`](../dashboard/app.py). Start from
it rather than from scratch (see [Starting point](#starting-point)).

**Simple deployment is a requirement, not a nice-to-have.** Read
[Deployment](#deployment-keep-it-simple) before adding anything that changes
how the app is installed or run.

## Background

### Terms

| Term | Meaning |
|---|---|
| Sample | One plant observed in the field, with its metadata (plant type, species, location, size, ...) entered in the field app. One row in `samples`, keyed by `master_ref_id`. Many images can share one sample. |
| Raw | The camera's original `.ARW` file. A developed image can only be made from the raw. Uploaded to Azure blob storage, then copied to NFS. |
| Preview JPG | The small JPG the camera saves next to the raw. Not good enough for training. |
| Developed image | The color-corrected JPG made from the raw, on NFS under `<batch>/developed-images/`. About 50 MB each. |
| Cutout | The plant segmented out of a developed image by Field-AnnotationPipeline, saved under `<batch>/cutouts/`. |
| Batch | Images grouped by location and date, labeled `<location_code>_<YYYY-MM-DD>`, e.g. `TX_2024-06-12`. |
| NFS | The research storage mounted on SUNNY at `/mnt/research-projects/r/raatwell/longterm_images3/`. Batches are under `field-batches/`. |

### The database

Everything comes from one SQLite file,
`/mnt/research-projects/r/raatwell/longterm_images3/field-db/field_exploration.db`
(`paths.db_path` in [`conf/paths/default.yaml`](../conf/paths/default.yaml)).
The weekly cron job ([`scripts/run_weekly_report.sh`](../scripts/run_weekly_report.sh),
Mondays at 7:00) refreshes most of it. Field-AnnotationPipeline writes the
`cutouts` table whenever it runs. Full table descriptions are in the
Field-AnnotationPipeline repo: `docs/db_overview.md` and
`docs/cutout_table_schema.md`.

The dashboard reads four tables:

| Table | Used for |
|---|---|
| `file_status` | The main table: one row per image (132,793), with location, plant type, species, batch, and 0/1 flags for where its raw and developed files are. |
| `samples` | Whether the image's metadata exists (join on `master_ref_id`), and which samples have no images. |
| `cutouts` | Whether the image has been through the cutout pipeline, whether a cutout was made, and the cutout's path. |
| `images` | The camera capture time (`exif_datetime`), used for seasons. |

## Definitions

### Have, To be processed, Don't have

An image is usable if we have both its **raw** and its **sample metadata**.
With both, it can always be processed into a cutout. Without either one,
nothing can be done with it.

| Bucket | Meaning |
|---|---|
| Have | Already run through the cutout pipeline. |
| To be processed | Has a raw and metadata, but hasn't been through the cutout pipeline yet. |
| Don't have | Missing the raw or the metadata, so it can't be processed. |

### Image states

Every image gets exactly one state. The checks run in this order, and the
first one that matches wins (`classify()` in `dashboard/app.py`):

| # | State | Bucket | Rule | Images |
|---|---|---|---|---|
| 1 | Cutout | Have | `cutouts.status` is `detected_segmented` or `segmented` | 52,899 |
| 2 | Annotated, no plant found | Have | `cutouts.status = 'no_detection'` | 8,748 |
| 3 | No raw (never uploaded) | Don't have | No raw in blob, no raw on NFS and no developed image | 21,577 |
| 4 | No metadata | Don't have | `master_ref_id` has no `samples` row | 1 |
| 5 | Developed, not annotated | To be processed | `processed_jpg_in_nfs = 1` | 33,260 |
| 6 | Raw on NFS, not developed | To be processed | `raw_in_nfs = 1` | 15,855 |
| 7 | Raw in blob only | To be processed | Anything left | 453 |
| | **Total** | | | **132,793** |

Counts are from the database as rebuilt on 2026-10-05.

- **No raw:** these raws were never uploaded and are treated as lost. Most of
  them (18,297) are weeds.
- **No metadata** shows only 1 image because the "No raw" check runs first:
  507 images have no `samples` row, and nearly all of them also have no raw.
- **Samples with no images** (402) are counted separately. They have metadata
  but no image row, so there's nothing to classify.

### Seasons

Use the same rule as the weekly report (`add_season_column` in
[`src/plot_by_season_db.py`](../src/plot_by_season_db.py)), based on the camera
capture time:

- Cover crops run October to September and get a two-year label: an image from
  November 2025 or March 2026 is in `2025/2026`.
- Everything else goes by calendar year: `2025`.
- Images with no capture time (917) are `Unknown`.

Don't invent a different rule. The dashboard's season numbers have to match
the weekly report's.

### Low counts

There are no target numbers per species or location. A group is flagged as low
when its count is below a percentage of the median across all groups. The
percentage is a slider on the page (25% by default), so people can adjust it
while looking at the data. Groups with no images at all can't show up in a
count, so they appear as blank cells on the Coverage page instead.

## Pages

A row of filters at the top (plant type, location, season) applies to every page.

| Page | Shows |
|---|---|
| Overview | Totals for Have, To be processed and Don't have; samples with no images; a bar per plant type split by bucket; a state × plant type table. |
| Coverage | Species × location grid. It can count usable images, all images or cutouts; blank cells mean none. |
| Gaps | Images that can't be processed, by location and reason; samples with no images; groups with low counts (by species, location, or both). |
| To be processed | Counts by location, plant type, species and state, with a CSV download of the image list. |
| Browse | Pick a species and show 8 random cutouts as thumbnails. |

## Data access

- **Open the database read-only:**
  `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`. The dashboard never
  writes to it; two pipelines share this file.
- **Build one table.** A single load function builds a per-image DataFrame
  (about 133,000 rows) with each image's state, bucket and season, and every
  page is a grouping of it. That way, two pages can't disagree.
- **Cache it** with `st.cache_data(ttl=3600)`. Loading takes about 5 s over
  NFS, and the first page load takes about 20 s including imports. The data
  only changes weekly, so an hour of caching is fine.
- **Watch for missing indexes.** `images.base_name` and
  `file_status.master_ref_id` have no index, which makes some joins very slow
  over NFS:
  - Joining `images` in SQL took about 40 s. Load capture times with a
    separate query and merge them in pandas instead, which takes under a second.
  - For "rows with no match", use `NOT IN (SELECT ...)` rather than
    `NOT EXISTS`. The `NOT EXISTS` version was still running after 2 minutes.
- **Load images only on request, as cached thumbnails.** Cutout PNGs are
  1.6–8 MB each and developed JPGs are about 50 MB, so only load them when
  someone clicks a button. Don't put developed images in a grid.
- **Clean the data once, in the load step,** not separately in each chart:
  - Blank strings and NULLs in location, plant type and species both become `Unknown`.
  - Three species are recorded under two names each and are merged:
    `common waterhemp` → `waterhemp`, `junglerice` → `jungle rice`, and
    `crabgrass (large or other spp.)` → `large crabgrass`.
  - `NC01` and `TX01` are sub-sites with a parent state in the `locations`
    table; `TX02` has no parent. The example shows codes as they are, while
    the weekly report merges `NC01` into `NC` (`db.locations.roll_up_to_parent`).
  - If you add `height` or `ground_cover` filters, their values use an en dash
    with spaces around it (`0 – 25`), not a hyphen.
  - `flower_fruit_or_seeds` is the text `'True'` or `'False'`, not 0/1.
- **Show how old the data is.** The example puts `file_status.refreshed_at` at
  the top of the page.

## Deployment: keep it simple

This section matters as much as the pages do. Whoever is around will be
looking after the dashboard, so anyone on the team must be able to start it,
restart it and update it in a few minutes, without special access or
knowledge.

### Rules

1. **No new infrastructure.** That means no new server, database, web server,
   reverse proxy, Docker container, or scheduled data export. The app reads
   the existing database directly.
2. **One environment.** Use the repo's existing uv environment. Add `streamlit`
   and `plotly` to `pyproject.toml` with `uv add`, so `./setup.sh` installs
   everything. Streamlit 1.65 and Plotly 7.1 install on the repo's Python 3.10
   alongside its pinned pandas 2.2.0, and the example runs on those versions.
3. **One command to start it:** `scripts/run_dashboard.sh`.
4. **One crontab line to keep it running.**
5. **Paths come from `conf/paths/default.yaml`.** Don't add config files,
   hard-coded paths or secrets. The app needs no credentials; it only reads
   files the account running it can already read.
6. **Deploying a change means `git pull` and a restart,** nothing else.
7. **If a feature would need anything ruled out in rule 1, check with Matthew
   before building it.**

### How it runs

- On SUNNY (`sunny.ece.ncsu.edu`), the server that mounts both the database and
  the image storage.
- Bound to `127.0.0.1` on port **8601**. Streamlit's default is 8501; a
  different port avoids clashing with anyone else's Streamlit app on SUNNY.
  Before using it, check that it's free with `ss -tln | grep 8601`.
- Under one person's account (Matthew's for now), started from that person's
  crontab.

**Why cron and not a systemd service:** on SUNNY, user services stop when you
log out (lingering is off, and turning it on needs an admin). Cron already runs
the weekly report, so it's a known-good mechanism.

### Start script

`scripts/run_dashboard.sh` starts the app only if it isn't already responding,
so it's safe to run at any time:

```bash
#!/usr/bin/env bash
# Starts the dashboard if it isn't already running. Safe to run repeatedly:
# cron runs it every 10 minutes, so the app comes back after a crash or reboot.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
PORT=8601
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if curl -sf "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null; then
    exit 0
fi

mkdir -p logging/dashboard
nohup uv run streamlit run dashboard/app.py \
    --server.address 127.0.0.1 --server.port "${PORT}" \
    --server.headless true --browser.gatherUsageStats false \
    >>"logging/dashboard/dashboard_$(date +%Y-%m-%d).log" 2>&1 &
```

Crontab line (add it with `crontab -e`):

```
*/10 * * * * /home/mkutuga/Field-DataExploration/scripts/run_dashboard.sh
```

To deploy a change:

```bash
git pull
pkill -f "streamlit run dashboard/app.py"
scripts/run_dashboard.sh
```

### Opening it

From the NCSU VPN, the firewall only lets SSH through to SUNNY. Ports 8000,
8080 and 8501 were all refused when tested on 2026-10-08. Viewers therefore go
through SSH, using either of these:

- **VS Code connected to SUNNY:** in the **Ports** tab, click **Forward a
  Port**, enter `8601`, and open the address it shows.
- **Any terminal:** run `ssh -N -L 8601:localhost:8601 <user>@sunny.ece.ncsu.edu`,
  then open http://localhost:8601.

Both need the NCSU VPN and an account on SUNNY. Not yet tested: whether a
shell on Lightning can reach SUNNY's port directly. Lightning users can run the
`ssh` command from their own machine instead.

Anyone with a SUNNY login can open the dashboard, including people outside the
research storage group who can't read the database or images directly. The
dashboard shows counts and cutout thumbnails.

### Checking a deploy

- `curl -s http://127.0.0.1:8601/_stcore/health` prints `ok`.
- The "last rebuilt" date at the top of the page matches the most recent
  Monday run.
- `uv run python dashboard/smoke_test.py` runs every page headlessly and
  reports no exceptions.

## Starting point

- [`dashboard/app.py`](../dashboard/app.py) is a single file of about 300
  lines. Everything under [Pages](#pages) already works.
- [`dashboard/smoke_test.py`](../dashboard/smoke_test.py) loads the app
  headlessly with Streamlit's `AppTest`, clicks through the filters and the
  Browse button, and prints any exceptions.

To run them before `streamlit` and `plotly` are in `pyproject.toml`, use these
commands from the repo root:

```bash
uv run --with streamlit --with plotly \
    streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8601
uv run --with streamlit --with plotly python dashboard/smoke_test.py
```

Keep it as one file until it's clearly too big. When you split it, move the
data loading and the rules (states, seasons, species merges) into
`dashboard/data.py` first, since that's the part other code might reuse.

## Build plan

Each step has a "done when" so you know when to move on.

1. **Make it deployable.** Do this before adding features. Add the
   dependencies with `uv add`, write `scripts/run_dashboard.sh`, add the
   crontab line, and add a short "Dashboard" section to the README with the
   start, restart and open instructions.
   *Done when:* after `pkill -f "streamlit run dashboard/app.py"`, the
   dashboard comes back by itself within 10 minutes.
2. **Check the numbers.** For each count on the Overview page, write the SQL
   query that should produce it and compare. Keep the queries in a short
   checklist (or in the smoke test) so they can be rerun.
   *Done when:* every state's count matches its query and they add up to the
   row count of `file_status`.
3. **Review the open questions below with Matthew** and apply the answers.
4. **Improve the pages,** based on what people ask for. Candidates:
   - a toggle to combine sub-sites into their state;
   - filters for sample attributes (size class, growth stage, ground cover);
   - a metadata-completeness view (for example, `height` is missing for 83%
     of cutouts).

## Open questions

1. Should "Annotated, no plant found" (8,748 images) count as Have? These
   images were processed but didn't produce a cutout.
2. Should sub-sites be combined into their state by default? If so, what
   happens to `TX02`, which has no parent in `locations`?
3. Are the three species merges correct?
4. Whose account should run the dashboard long term? It stops if that
   person's account goes away.
5. Is it acceptable that anyone with a SUNNY account can open it?

## Decisions

1. **Data, not pipeline.** The pages are organized around what data exists
   and its state, not around jobs or throughput.
2. **Streamlit on SUNNY, reading the database directly.** The database and
   images are already mounted on SUNNY. A static site couldn't show images
   without copying them somewhere, and a separate data export would be one
   more thing to run.
3. **No history table.** Trends over time were dropped along with the
   pipeline focus. The dashboard always shows the current state.
4. **Low counts are relative to the median, not to targets,** because there
   are no target numbers yet.
5. **Reuse the weekly report's season rule** so both show the same numbers.
6. **Viewers use SSH port forwarding** because the firewall refuses other
   ports from the VPN. Opening a port would need ECE IT and would add
   something to maintain.
