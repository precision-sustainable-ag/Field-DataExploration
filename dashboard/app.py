"""Example dashboard for the Field-AgIR data: what we have, what we don't,
and what still needs processing. See docs/dashboard_design.md.

Runs on SUNNY against the live database (read-only). From the repo root:

    uv run --with streamlit --with plotly \
        streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8601

Then open it through VS Code's port forwarding, or
`ssh -N -L 8601:localhost:8601 <user>@sunny.ece.ncsu.edu` and http://localhost:8601.
"""
import io
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from omegaconf import OmegaConf
from PIL import Image

# Same paths the pipeline uses. Hydra mounts this file under "paths", so wrap it
# the same way for its ${paths.*} references to resolve.
REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = OmegaConf.create({"paths": OmegaConf.load(REPO_ROOT / "conf/paths/default.yaml")}).paths
DB_PATH = PATHS.db_path
FIELD_BATCHES = Path(PATHS.longterm_storage)

# Every image lands in exactly one state; each state belongs to one bucket.
# "Don't have" = missing the raw or the sample metadata, so it can't be processed.
STATE_BUCKET = {
    "Cutout": "Have",
    "Annotated, no plant found": "Have",
    "Developed, not annotated": "To be processed",
    "Raw on NFS, not developed": "To be processed",
    "Raw in blob only": "To be processed",
    "No raw (never uploaded)": "Don't have",
    "No metadata": "Don't have",
}
BUCKETS = ["Have", "To be processed", "Don't have"]

# Categorical slots 1-3 and the sequential blue ramp from the dataviz reference palette.
BUCKET_COLORS = {"Have": "#2a78d6", "To be processed": "#eb6834", "Don't have": "#1baf7a"}
FLAG_COLORS = {"Low": "#eb6834", "OK": "#2a78d6"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Species recorded under two names that share a class_id (see cutout_table_schema.md).
SPECIES_ALIASES = {
    "common waterhemp": "waterhemp",
    "junglerice": "jungle rice",
    "crabgrass (large or other spp.)": "large crabgrass",
}

IMAGES_QUERY = """
    SELECT fs.base_name, fs.location_code, fs.plant_type, fs.species, fs.batch_label,
           fs.raw_in_blob, fs.raw_in_nfs, fs.processed_jpg_in_nfs,
           s.master_ref_id IS NOT NULL AS has_metadata,
           c.status AS cutout_status, c.cutout_path
    FROM file_status fs
    LEFT JOIN samples s ON s.master_ref_id = fs.master_ref_id
    LEFT JOIN cutouts c ON c.base_name = fs.base_name AND c.cutout_index = 0
"""
# Joining images in SQL takes ~40 s over NFS; merging in pandas takes under a second.
CAPTURE_QUERY = """
    SELECT base_name, exif_datetime AS captured_at
    FROM images
    WHERE base_name IS NOT NULL AND exif_datetime IS NOT NULL
"""
SAMPLES_WITHOUT_IMAGES_QUERY = """
    SELECT master_ref_id, location_code, plant_type, species
    FROM samples
    WHERE master_ref_id NOT IN (SELECT master_ref_id FROM file_status WHERE master_ref_id IS NOT NULL)
"""


def classify(df: pd.DataFrame) -> np.ndarray:
    has_raw = df.raw_in_blob.eq(1) | df.raw_in_nfs.eq(1) | df.processed_jpg_in_nfs.eq(1)
    conditions = [
        df.cutout_status.isin(["detected_segmented", "segmented"]),
        df.cutout_status.eq("no_detection"),
        ~has_raw,
        df.has_metadata.eq(0),
        df.processed_jpg_in_nfs.eq(1),
        df.raw_in_nfs.eq(1),
    ]
    choices = [
        "Cutout",
        "Annotated, no plant found",
        "No raw (never uploaded)",
        "No metadata",
        "Developed, not annotated",
        "Raw on NFS, not developed",
    ]
    return np.select(conditions, choices, default="Raw in blob only")


def season(captured_at: pd.Series, plant_type: pd.Series) -> pd.Series:
    """Same rule as plot_by_season_db.py in Field-DataExploration: cover crops
    run October-September (labeled 2025/2026), everything else by calendar year."""
    year = captured_at.dt.year.astype("Int64")
    cover = plant_type.eq("COVERCROPS")
    start = year.where(~(cover & captured_at.dt.month.lt(10)), year - 1)
    label = year.astype("string").mask(cover, start.astype("string") + "/" + (start + 1).astype("string"))
    return label.fillna("Unknown")


def clean_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["location_code", "plant_type", "species"]:
        df[column] = df[column].replace("", None).fillna("Unknown")
    df["species"] = df["species"].replace(SPECIES_ALIASES)
    return df


@st.cache_data(ttl=3600, show_spinner="Loading the field database...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as conn:
        images = pd.read_sql_query(IMAGES_QUERY, conn)
        captured = pd.read_sql_query(CAPTURE_QUERY, conn)
        samples_without_images = pd.read_sql_query(SAMPLES_WITHOUT_IMAGES_QUERY, conn)
        refreshed_at = conn.execute("SELECT max(refreshed_at) FROM file_status").fetchone()[0]

    for column in ["raw_in_blob", "raw_in_nfs", "processed_jpg_in_nfs", "has_metadata"]:
        images[column] = images[column].fillna(0).astype(int)

    # A base_name can have a raw and a preview row in images; take the earliest capture time.
    captured = captured.groupby("base_name", as_index=False)["captured_at"].min()
    df = images.merge(captured, on="base_name", how="left")
    df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")

    df = clean_dimensions(df)
    df["state"] = classify(df)
    df["bucket"] = df["state"].map(STATE_BUCKET)
    df["season"] = season(df["captured_at"], df["plant_type"])
    return df, clean_dimensions(samples_without_images), refreshed_at


@st.cache_data(max_entries=500, show_spinner=False)
def thumbnail(rel_path: str) -> bytes:
    with Image.open(FIELD_BATCHES / rel_path) as img:
        img.thumbnail((360, 360))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return buf.getvalue()


def ordered_states(columns) -> list[str]:
    return [state for state in STATE_BUCKET if state in columns]


st.set_page_config(page_title="Field-AgIR data", layout="wide")
df, samples_without_images, refreshed_at = load_data()

st.title("Field-AgIR data")
st.caption(
    f"Read-only view of field_exploration.db. File status last rebuilt "
    f"{refreshed_at[:16].replace('T', ' ')} UTC; this page caches it for an hour."
)

f1, f2, f3 = st.columns(3)
plant_types = f1.multiselect("Plant type", sorted(df.plant_type.unique()), placeholder="All")
locations = f2.multiselect("Location", sorted(df.location_code.unique()), placeholder="All")
season_options = sorted((s for s in df.season.unique() if s != "Unknown"), reverse=True) + ["Unknown"]
seasons = f3.multiselect("Season", season_options, placeholder="All")

view = df
samples_view = samples_without_images
if plant_types:
    view = view[view.plant_type.isin(plant_types)]
    samples_view = samples_view[samples_view.plant_type.isin(plant_types)]
if locations:
    view = view[view.location_code.isin(locations)]
    samples_view = samples_view[samples_view.location_code.isin(locations)]
if seasons:
    view = view[view.season.isin(seasons)]

if view.empty:
    st.info("No images match these filters.")
    st.stop()

overview, coverage, gaps, todo_tab, browse = st.tabs(
    ["Overview", "Coverage", "Gaps", "To be processed", "Browse"]
)

with overview:
    counts = view.bucket.value_counts()
    tiles = st.columns(5)
    tiles[0].metric("Images", f"{len(view):,}")
    for tile, bucket in zip(tiles[1:4], BUCKETS):
        tile.metric(bucket, f"{counts.get(bucket, 0):,}")
    tiles[4].metric(
        "Samples with no images",
        f"{len(samples_view):,}",
        help="Sample records with no image at all. Not affected by the season filter.",
    )

    by_type = view.groupby(["plant_type", "bucket"]).size().reset_index(name="images")
    fig = px.bar(
        by_type, x="images", y="plant_type", color="bucket", orientation="h",
        color_discrete_map=BUCKET_COLORS, category_orders={"bucket": BUCKETS},
        labels={"images": "Images", "plant_type": "", "bucket": ""},
    )
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(bargap=0.4, height=340, legend=dict(orientation="h", y=1.12), margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

    state_table = view.pivot_table(
        index="state", columns="plant_type", values="base_name", aggfunc="count", fill_value=0
    )
    state_table = state_table.reindex(ordered_states(state_table.index))
    state_table["Total"] = state_table.sum(axis=1)
    state_table.insert(0, "Category", state_table.index.map(STATE_BUCKET))
    st.dataframe(state_table, width="stretch")

with coverage:
    measure = st.radio(
        "Count", ["Usable images (raw + metadata)", "All images", "Cutouts"], horizontal=True
    )
    subset = {
        "Usable images (raw + metadata)": view[view.bucket != "Don't have"],
        "All images": view,
        "Cutouts": view[view.state == "Cutout"],
    }[measure]
    if subset.empty:
        st.info("Nothing to show for these filters.")
    else:
        pivot = subset.pivot_table(index="species", columns="location_code", values="base_name", aggfunc="count")
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        text = pivot.map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
        fig = px.imshow(
            pivot, aspect="auto", color_continuous_scale=BLUE_RAMP,
            labels=dict(x="Location", y="Species", color="Images"),
        )
        fig.update_traces(
            text=text.values, texttemplate="%{text}", hoverongaps=False,
            hovertemplate="%{y} at %{x}<br>%{z:,} images<extra></extra>",
        )
        fig.update_xaxes(side="top")
        fig.update_layout(height=max(400, 22 * len(pivot) + 140))
        st.caption("Blank cells mean no images for that species at that location.")
        st.plotly_chart(fig, width="stretch")

with gaps:
    st.subheader("Don't have")
    st.caption("Images missing the raw or the sample metadata. These can't be processed.")
    missing = view[view.bucket == "Don't have"]
    if missing.empty:
        st.write("None for these filters.")
    else:
        missing_table = missing.pivot_table(
            index="location_code", columns="state", values="base_name", aggfunc="count", fill_value=0
        )
        missing_table["Total"] = missing_table.sum(axis=1)
        st.dataframe(missing_table.sort_values("Total", ascending=False), width="stretch")
    st.markdown(f"**Sample records with no images:** {len(samples_view):,}")
    if not samples_view.empty:
        st.dataframe(
            samples_view.groupby(["plant_type", "location_code"]).size().rename("samples")
            .reset_index().sort_values("samples", ascending=False),
            hide_index=True,
        )

    st.subheader("Low counts")
    g1, g2, g3 = st.columns(3)
    group_by = g1.selectbox("Group by", ["Species", "Location", "Species and location"])
    low_measure = g2.selectbox("Count", ["Usable images", "Cutouts"])
    pct = g3.slider("Flag groups below this % of the median", 5, 100, 25, step=5)

    keys = {"Species": ["species"], "Location": ["location_code"], "Species and location": ["species", "location_code"]}[group_by]
    counted = view[view.bucket != "Don't have"] if low_measure == "Usable images" else view[view.state == "Cutout"]
    grouped = counted.groupby(keys).size().rename("count").reset_index()
    if grouped.empty:
        st.info("Nothing to count for these filters.")
    else:
        median = grouped["count"].median()
        threshold = median * pct / 100
        grouped["flag"] = np.where(grouped["count"] < threshold, "Low", "OK")
        flagged = grouped[grouped.flag == "Low"].sort_values("count")
        st.caption(
            f"Median {median:,.0f} per group, so groups under {threshold:,.0f} are flagged: "
            f"{len(flagged)} of {len(grouped)}. Groups with none at all aren't listed; "
            f"they're the blank cells on the Coverage tab."
        )
        if len(keys) == 1:
            fig = px.bar(
                grouped.sort_values("count"), x="count", y=keys[0], color="flag", orientation="h",
                color_discrete_map=FLAG_COLORS, category_orders={"flag": ["Low", "OK"]},
                labels={"count": low_measure, keys[0]: "", "flag": ""},
            )
            fig.update_yaxes(categoryorder="total ascending")
            fig.add_vline(x=threshold, line_dash="dot", line_width=1, line_color="#52514e")
            fig.update_layout(bargap=0.3, height=max(320, 18 * len(grouped) + 100), legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, width="stretch")
        st.dataframe(flagged.drop(columns="flag"), hide_index=True)

with todo_tab:
    todo = view[view.bucket == "To be processed"]
    st.caption("Images with both a raw and sample metadata that haven't been turned into cutouts yet.")
    if todo.empty:
        st.write("Nothing waiting for these filters.")
    else:
        todo_table = todo.pivot_table(
            index=["location_code", "plant_type", "species"], columns="state",
            values="base_name", aggfunc="count", fill_value=0,
        )
        todo_table = todo_table[ordered_states(todo_table.columns)]
        todo_table["Total"] = todo_table.sum(axis=1)
        st.dataframe(todo_table.sort_values("Total", ascending=False).reset_index(), hide_index=True, width="stretch")
        st.download_button(
            f"Download the {len(todo):,} images as CSV",
            todo[["base_name", "state", "location_code", "plant_type", "species", "batch_label", "season"]].to_csv(index=False),
            file_name="to_be_processed.csv",
            mime="text/csv",
        )

with browse:
    with_cutouts = view[view.state == "Cutout"]
    if with_cutouts.empty:
        st.info("No cutouts for these filters.")
    else:
        species_counts = with_cutouts.species.value_counts()
        species = st.selectbox(
            "Species", species_counts.index, format_func=lambda s: f"{s} ({species_counts[s]:,} cutouts)"
        )
        if st.button("Show 8 random cutouts"):
            picks = with_cutouts[with_cutouts.species == species].sample(min(8, int(species_counts[species])))
            columns = st.columns(4)
            for i, row in enumerate(picks.itertuples()):
                with columns[i % 4]:
                    try:
                        st.image(thumbnail(row.cutout_path), caption=f"{row.base_name} · {row.location_code} · {row.batch_label}")
                    except FileNotFoundError:
                        st.warning(f"File not found: {row.cutout_path}")
