## 1. Purpose

A lightweight Streamlit dashboard for exploring Field-AgIR image data, identifying dataset gaps, and tracking what remains available for processing.

### Core Questions

- What usable images and cutouts exist by species, location, plant type, and season?
- Where are the gaps in dataset coverage?
- Which images are ready for processing?
- Which images are unusable, and why? (e.g. lack of matching metadata and raw images)

**Scope:** Data availability, completeness, and coverage. Pipeline monitoring, performance, and job tracking are out of scope.

## 2. Data Sources

The dashboard reads the existing SQLite database:

```text
/mnt/research-projects/r/raatwell/longterm_images3/field-db/field_exploration.db
```

| Table | Purpose |
|---|---|
| `file_status` | Image inventory, file availability, and metadata |
| `samples` | Sample metadata and samples without images |
| `cutouts` | Annotation status and segmented plants |
| `images` | Capture timestamps used for seasons |

The database is refreshed weekly, while Field-AnnotationPipeline independently updates `cutouts`.

**Database access is read-only.**

## 3. Image Classification

Each image belongs to exactly one state, assigned in priority order.

| State | Category | Images |
|---|---|---:|
| Cutout | Have | 52,899 |
| Annotated, no plant found | Have | 8,748 |
| No raw available | Don't have | 21,577 |
| Missing metadata | Don't have | 1 |
| Developed, not annotated | To be processed | 33,260 |
| Raw on NFS, not developed | To be processed | 15,855 |
| Raw in Azure only | To be processed | 453 |
| **Total** | | **132,793** |

*Baseline: October 5, 2026.*

### Classification Definitions

- **Have:** Image completed annotation processing.
- **To be processed:** Raw image and metadata exist, but processing is incomplete.
- **Don't have:** Raw image or metadata is missing, preventing processing.

An additional **402 samples** have metadata but no corresponding image records.

## 4. Dashboard Pages

| Page | Functionality |
|---|---|
| **Overview** | Dataset totals, processing states, and distribution by plant type |
| **Coverage** | Species × location matrix with image and cutout counts |
| **Gaps** | Missing data, unusable images, and underrepresented groups |
| **To be processed** | Processing backlog by species, location, and state; CSV export |
| **Browse** | Display 8 random cutout thumbnails by species |

### Filtering and Grouping

- Global filters: **Plant type, location, and season**
- Low-count threshold: Adjustable percentage of the median (default 25%)
- Empty coverage cells indicate no images
- Cover crop seasons: October–September (e.g., `2025/2026`)
- Other plant types: Calendar year
- Missing capture timestamps: `Unknown`

Season definitions must match the existing weekly report.

## 5. Technical Design

| Component | Implementation |
|---|---|
| Framework | Streamlit + Plotly |
| Environment | Existing `uv` environment |
| Data source | SQLite, read-only |
| Data processing | Shared per-image pandas DataFrame |
| Caching | `st.cache_data(ttl=3600)` |
| Visualization | Plotly charts and tables |
| Image loading | On-demand cached cutout thumbnails |

### Design Requirements

1. Build one consistent per-image DataFrame shared across pages.
2. Avoid expensive SQL joins on unindexed columns.
3. Merge image capture timestamps using pandas.
4. Normalize missing values, species names, and seasons during loading.
5. Load cutout thumbnails only when requested.
6. Display the database's last refresh timestamp.
7. Keep implementation simple and modular.

### Existing Implementation

- `dashboard/app.py` — Functional Streamlit dashboard
- `dashboard/smoke_test.py` — Headless application tests

Keep the implementation in one file unless complexity justifies moving data loading and classification into `dashboard/data.py`.

## 6. Deployment

**Priority: Minimal infrastructure and maintenance.**

### Requirements

- Host on SUNNY (`sunny.ece.ncsu.edu`)
- Bind Streamlit to `127.0.0.1:8601`
- Reuse existing database, storage, configuration, and environment
- No Docker, new database, web server, reverse proxy, or scheduled exports
- Start with `scripts/run_dashboard.sh`
- Use cron to check availability every 10 minutes
- Updates require only `git pull` and an application restart

### Access

Users connect through the NCSU VPN and SSH port forwarding:

```bash
ssh -N -L 8601:localhost:8601 <user>@sunny.ece.ncsu.edu
```

Open:

```text
http://localhost:8601
```

Alternatively, forward port `8601` using VS Code's **Ports** tab.

### Validation

```bash
# Confirm application health
curl -s http://127.0.0.1:8601/_stcore/health

# Run dashboard smoke tests
uv run python dashboard/smoke_test.py
```

## 7. Implementation Priorities

- [ ] **Deployment**
  - Add Streamlit and Plotly to project dependencies.
  - Create `scripts/run_dashboard.sh`.
  - Configure automatic startup using cron.
  - Document startup, restart, and access procedures.

- [ ] **Data Validation**
  - Verify dashboard counts against independent SQL queries.
  - Confirm classification totals match `file_status`.
  - Validate state, season, and species grouping logic.

- [ ] **Resolve Open Questions**
  - Confirm classification and grouping conventions.
  - Determine application ownership and access permissions.

- [ ] **Future Improvements**
  - Optional location rollups.
  - Sample-attribute filters (size, growth stage, ground cover).
  - Metadata-completeness statistics.

### Acceptance Criteria

- Dashboard automatically restarts after failure.
- All displayed counts reconcile with the database.
- `dashboard/smoke_test.py` passes without exceptions.
- Deployment requires no additional infrastructure.

## 8. Open Decisions

1. Should images annotated without a detected plant count as **Have**?
2. Should sub-sites (`NC01`, `TX01`, `TX02`) be grouped under parent locations?
3. Are the three existing species-name merges correct?
4. Which account should maintain the running application?
5. Is access for anyone with a SUNNY account acceptable?

## 9. Key Design Principles

- **One source of truth:** Existing SQLite database.
- **Consistent classification:** Every image has one state and category.
- **Minimal infrastructure:** Reuse existing systems.
- **Consistent reporting:** Match existing season and data-processing conventions.
- **Actionable insights:** Prioritize coverage gaps, missing data, and processing backlog.
- **Low maintenance:** Simple deployment, updates, and troubleshooting.

---

**End Goal:** A low-maintenance dashboard that helps the team understand what Field-AgIR data exists, where coverage is insufficient, and what remains available for processing.
