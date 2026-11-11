"""Field-AgIR data dashboard: what we have, what we don't, what still needs processing,
and what it all takes up on disk.

The app only reads a snapshot folder built by dashboard/snapshot.py on SUNNY (where
long-term storage is mounted) and copied here by scripts/push_snapshot.sh.

    FIELD_DASHBOARD_DATA        snapshot folder (default: dashboard/snapshot)
    GLOBUS_COLLECTION_ID        optional: Globus collection that holds field-batches
    GLOBUS_FIELD_BATCHES_PATH   optional: path of field-batches inside that collection
    GLOBUS_HTTPS_URL            optional: the collection's HTTPS URL, so links open the image itself

Start it with scripts/run_dashboard.sh (HTTPS on port 8501), or for a quick local look:

    uv run streamlit run dashboard/app.py
"""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path(os.environ.get("FIELD_DASHBOARD_DATA", HERE / "snapshot"))
GLOBUS_COLLECTION = os.environ.get("GLOBUS_COLLECTION_ID", "").strip()
# Copied from the Globus website's address bar, so they may still be URL-encoded (%2F).
GLOBUS_BATCHES_PATH = unquote(os.environ.get("GLOBUS_FIELD_BATCHES_PATH", "").strip()).rstrip("/")
GLOBUS_HTTPS = os.environ.get("GLOBUS_HTTPS_URL", "").strip().rstrip("/")
STALE_AFTER_DAYS = 8

# Colors from the dataviz reference palette: categorical slots 1-2 plus a neutral for "missing".
CAT_COLORS = {"Have": "#2a78d6", "To be processed": "#eb6834", "Don't have": "#898781"}
BELOW, ABOVE = "#eb6834", "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ACTIONS = {  # order = priority
    "Collect": ("#d03b3b", "Below target and nothing in the backlog. Needs new field images."),
    "Process + collect": ("#ec835a", "Processing the backlog helps but won't reach the target on its own."),
    "Process": ("#fab219", "The backlog should be enough to reach the target."),
    "Met": ("#0ca30c", "Already at or above the target."),
}
ROLE_LABEL = {"raws": "Raw images", "developed": "Developed JPGs", "cutouts": "Cutouts", "masks": "Masks", "unknown": "Unknown"}
DIM_COLS = ["plant_type", "location", "season", "species"]


# ---------- formatting ----------
def fmt(n) -> str:
    return f"{int(round(n)):,}" if n is not None and not (isinstance(n, float) and math.isnan(n)) else "–"


def pct(n, d) -> str:
    return f"{100 * n / d:.{1 if d and n / d < 0.1 else 0}f}%" if d else "–"


def size(n) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "–"
    units, i = ["B", "KB", "MB", "GB", "TB", "PB"], 0
    while n >= 1000 and i < len(units) - 1:
        n, i = n / 1000, i + 1
    return f"{n:.2f} {units[i]}" if i and n < 10 else f"{n:.1f} {units[i]}" if i and n < 100 else f"{round(n)} {units[i]}"


def short(v: float) -> str:
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"


def when(stamp) -> str:
    return str(stamp)[:16].replace("T", " ") + " UTC" if stamp else "unknown"


def dot(color: str) -> str:
    return f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{color};margin-right:5px"></span>'


# ---------- loading ----------
@st.cache_data(show_spinner="Loading the snapshot...")
def load(folder: str, mtime: float) -> dict:
    """Read data.json and turn its compact arrays into DataFrames. mtime is only part of the
    cache key, so a new snapshot is picked up as soon as it lands."""
    D = json.loads((Path(folder) / "data.json").read_text(encoding="utf-8"))
    dims = D["dims"]
    codes = lambda col, dim: pd.Categorical.from_codes(col, categories=dims[dim])  # noqa: E731

    rows = pd.DataFrame(D["rows"], columns=["pt", "loc", "season", "species", "state", "status", "images", "cutouts"])
    cube = pd.DataFrame({
        "plant_type": codes(rows.pt, "plant_type"), "location": codes(rows["loc"], "location_code"),
        "season": codes(rows.season, "season"), "species": codes(rows.species, "species"),
        "state": codes(rows.state, "state"), "status": [dims["status"][i] for i in rows.status],
        "images": rows.images, "cutouts": rows.cutouts,
    })
    cube["category"] = cube.state.map(D["state_category"]).astype(str)

    s = pd.DataFrame(D["samples"], columns=["pt", "loc", "species", "samples", "season"])
    samples = pd.DataFrame({
        "plant_type": codes(s.pt, "plant_type"), "location": codes(s["loc"], "location_code"),
        "species": codes(s.species, "species"), "season": codes(s.season, "season"), "samples": s.samples,
    })

    thumbs = pd.DataFrame(D["thumbs"]) if D["thumbs"] else pd.DataFrame(
        columns=["species", "location", "plant_type", "season", "name", "batch", "path", "w", "h", "thumb", "full"])
    for col, dim in [("species", "species"), ("location", "location_code"), ("plant_type", "plant_type"), ("season", "season")]:
        thumbs[col] = [dims[dim][i] for i in thumbs[col]] if len(thumbs) else thumbs[col]

    S = D["storage"]
    disk = pd.DataFrame(S.get("rows", []), columns=["pt", "loc", "product", "files", "bytes"])
    if len(disk):
        disk["plant_type"] = [dims["plant_type"][i] if i >= 0 else None for i in disk.pt]
        disk["location"] = [dims["location_code"][i] if i >= 0 else None for i in disk["loc"]]
        disk["product"] = [S["products"][i] for i in disk["product"]]
        disk["role"] = disk["product"].map(S["roles"]).fillna("unknown")
    return {"D": D, "cube": cube, "samples": samples, "thumbs": thumbs, "disk": disk}


# ---------- page ----------
st.set_page_config(page_title="Field-AgIR data", page_icon=":seedling:", layout="wide")

data_file = SNAPSHOT / "data.json"
if not data_file.exists():
    st.title("Field-AgIR data")
    st.error(f"No snapshot found at `{SNAPSHOT}`. Build one on SUNNY with `dashboard/snapshot.py` and copy it here "
             "with `scripts/push_snapshot.sh`, or point `FIELD_DASHBOARD_DATA` at a snapshot folder.")
    st.stop()

L = load(str(SNAPSHOT), data_file.stat().st_mtime)
D, cube_all, samples_all, thumbs_all, disk_all = L["D"], L["cube"], L["samples"], L["thumbs"], L["disk"]
STATES, CATEGORIES = D["dims"]["state"], D["categories"]
TOTAL = cube_all.groupby("category").images.sum()
CPI = cube_all.cutouts.sum() / max(1, TOTAL.get("Have", 0))  # cutouts per annotated image, incl. images with no plant

st.title("Field-AgIR data")
built = datetime.strptime(D["generated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
age = (datetime.now(timezone.utc) - built).days
st.caption(f"Database last rebuilt {when(D['refreshed_at'])} · snapshot built {when(D['generated_at'])}")
if age > STALE_AFTER_DAYS:
    st.warning(f"This snapshot is {age} days old. The weekly update from SUNNY may have stopped.")
failed = [c for c in D["checks"] if not c["ok"]]
if failed:
    st.error(f"Some counts may be wrong: {len(failed)} consistency check(s) failed. See Data issues.")

# ---------- global filters ----------
def reset_filters():
    for k in ("f_pt", "f_loc", "f_season"):
        st.session_state[k] = []


f1, f2, f3, f4 = st.columns([1, 1, 1, 0.6], vertical_alignment="bottom")
f_pt = f1.multiselect("Plant type", D["dims"]["plant_type"], placeholder="All", key="f_pt")
f_loc = f2.multiselect("Location", D["dims"]["location_code"], placeholder="All", key="f_loc")
f_season = f3.multiselect("Season", D["dims"]["season"], placeholder="All", key="f_season")
f4.button("Reset filters", on_click=reset_filters, width="stretch")


def apply_filters(df: pd.DataFrame, season: bool = True) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, chosen in (("plant_type", f_pt), ("location", f_loc), ("season", f_season if season else [])):
        if chosen and col in df:
            mask &= df[col].isin(chosen)
    return df[mask]


cube = apply_filters(cube_all)
n_images = int(cube.images.sum())
st.caption(f"{fmt(n_images)} images" if n_images == cube_all.images.sum()
           else f"Showing {fmt(n_images)} of {fmt(cube_all.images.sum())} images")
backlog_of = lambda df: df.images.where(df.category == "To be processed", 0)  # noqa: E731

# Lazy tabs: only the open tab's code runs. The open tab is kept in the URL (?tab=Browse), so links can point at a tab.
tabs = st.tabs(["Overview", "Coverage", "Priorities", "Data issues", "Browse", "Storage", "How it's counted"],
               key="tab", on_change="rerun", bind="query-params")


# ---------- overview ----------
with tabs[0]:
    if tabs[0].open:  # only the open tab runs, so the other tabs cost nothing
        by_cat = cube.groupby("category").images.sum()
        by_state = cube.groupby("state", observed=False).images.sum()
        plants = int(cube.loc[cube.state == "Cutout", "images"].sum())
        n_samples = int(apply_filters(samples_all).samples.sum())
        tiles = [("Images", n_images, "all image records" if not (f_pt or f_loc or f_season) else f"of {fmt(cube_all.images.sum())} in total")]
        tiles += [(c, by_cat.get(c, 0), f"{pct(by_cat.get(c, 0), n_images)} of images") for c in CATEGORIES]
        tiles += [("Cutouts", cube.cutouts.sum(), f"segmented plants, from {fmt(plants)} images"),
                  ("Samples with no images", n_samples, "field records no image points to")]
        for col, (label, value, sub) in zip(st.columns(len(tiles)), tiles):
            col.metric(label, fmt(value), border=True)
            col.caption(sub)

        left, right = st.columns(2)
        with left:
            st.subheader("Where the images stop")
            st.caption("Each bar is a subset of the one above it. Hover a bar to see how many images didn't reach it, and why.")
            have = by_cat.get("Have", 0)
            stages = [
                ("All images", n_images, "Every image record in file_status.", ""),
                ("Raw + metadata", n_images - by_cat.get("Don't have", 0), "Has a raw image and a matching sample record.", "missing the raw image or the metadata"),
                ("Developed", have + by_state.get("Developed, not annotated", 0), "Raw developed to JPG (or already annotated).", "have a raw that isn't developed yet"),
                ("Annotated", have, "Went through the annotation pipeline.", "developed but not annotated yet"),
                ("Plant found", plants, "Annotation found and segmented at least one plant.", "annotated, but no plant was found"),
            ]
            hover = [f"<b>{n}</b>: {fmt(v)} ({pct(v, n_images)} of all)<br>{d}" + (f"<br>Not here: {fmt(stages[i - 1][1] - v)} {lost}" if i else "")
                     for i, (n, v, d, lost) in enumerate(stages)]
            fig = go.Figure(go.Bar(
                y=[s[0] for s in stages], x=[s[1] for s in stages], orientation="h", marker_color=CAT_COLORS["Have"],
                text=[f"{fmt(v)} · {pct(v, n_images)}" for _, v, _, _ in stages], textposition="outside", cliponaxis=False,
                hovertext=hover, hoverinfo="text"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(visible=False, range=[0, max(1, n_images) * 1.25])
            fig.update_layout(height=260, margin=dict(l=0, r=10, t=10, b=10), bargap=0.35)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with right:
            st.subheader("Category by plant type")
            by_type = cube.groupby(["plant_type", "category"], observed=True).images.sum().reset_index()
            order = by_type.groupby("plant_type", observed=True).images.sum().sort_values().index.tolist()
            fig = go.Figure()
            for c in CATEGORIES:
                part = by_type[by_type.category == c].set_index("plant_type").reindex(order).images.fillna(0)
                fig.add_bar(y=[str(o) for o in order], x=part.values, name=c, orientation="h", marker_color=CAT_COLORS[c],
                            hovertemplate=f"%{{y}} · {c}<br>%{{x:,}} images<extra></extra>")
            fig.update_layout(barmode="stack", height=260, margin=dict(l=0, r=10, t=10, b=10), bargap=0.35,
                              legend=dict(orientation="h", y=1.15, x=0, traceorder="normal"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.subheader("Week by week")
        hist = pd.DataFrame(D["history"])
        if len(hist) >= 2:
            st.caption("Totals from each weekly build, all data (filters don't apply here).")
            fig = go.Figure()
            for c in CATEGORIES:
                fig.add_scatter(x=hist.refreshed_at.str[:10], y=hist[c], name=c, mode="lines+markers",
                                line=dict(color=CAT_COLORS[c], width=2), hovertemplate=f"%{{x}} · {c}<br>%{{y:,}} images<extra></extra>")
            fig.update_layout(height=280, margin=dict(l=0, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15, x=0))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.caption(f"The trend appears once there are two weekly builds. This is build {len(hist)}.")

        st.subheader("Images by state and plant type")
        table = cube.pivot_table(index="state", columns="plant_type", values="images", aggfunc="sum", observed=True, fill_value=0)
        table = table.reindex([s for s in STATES if s in table.index])
        table.columns = [str(c) for c in table.columns]
        table["Total"] = table.sum(axis=1)
        table.insert(0, "Category", [D["state_category"][s] for s in table.index])
        table.loc["Total"] = ["", *table.drop(columns="Category").sum().tolist()]
        st.dataframe(table, width="stretch", column_config={c: st.column_config.NumberColumn(format="localized")
                                                             for c in table.columns if c != "Category"})


# ---------- coverage ----------
with tabs[1]:
    if tabs[1].open:  # only the open tab runs, so the other tabs cost nothing
        st.subheader("Coverage")
        c1, c2 = st.columns(2)
        measure = c1.segmented_control("Count", ["Cutouts", "Backlog", "Expected cutouts"], default="Cutouts", required=True, key="cov_measure")
        col_label = c2.segmented_control("Columns", ["Location", "Season", "Plant type"], default="Location", required=True, key="cov_cols")
        col = {"Location": "location", "Season": "season", "Plant type": "plant_type"}[col_label]
        st.caption({
            "Cutouts": "Segmented plants, by the species of the image they came from. Blank cells have none.",
            "Backlog": "Images with a raw and metadata that haven't been annotated yet. Processing these could fill blank cells.",
            "Expected cutouts": f"Cutouts now plus what the backlog should add: {CPI:.1f} cutouts per backlog image, the average per annotated image so far.",
        }[measure])
        values = {"Cutouts": cube.cutouts, "Backlog": backlog_of(cube), "Expected cutouts": cube.cutouts + backlog_of(cube) * CPI}[measure]
        grid = cube.assign(v=values).groupby(["species", col], observed=True).v.sum().round().unstack(fill_value=0)
        species_present = cube.species.unique()
        grid = grid.reindex([s for s in grid.index if s in species_present])
        if grid.empty:
            st.info("Nothing for these filters.")
        else:
            if col != "season":
                grid = grid[sorted(grid.columns, key=str)]
            totals = grid.sum(axis=1)
            grid = grid.loc[totals.sort_values(ascending=False).index]
            totals = grid.sum(axis=1)
            z = grid.where(grid > 0)
            top = max(1.0, float(grid.values.max()))
            fig = go.Figure(go.Heatmap(
                z=(z + 1).apply(lambda c: c.map(math.log10)).values, x=[str(c) for c in grid.columns],
                y=[f"{s}  ({fmt(t)})" for s, t in totals.items()], text=z.map(lambda v: "" if pd.isna(v) else fmt(v)).values,
                texttemplate="%{text}", textfont=dict(size=11), customdata=grid.values, xgap=2, ygap=2,
                colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)], zmin=0, zmax=math.log10(top + 1),
                hovertemplate=f"%{{y}} · %{{x}}<br>%{{customdata:,}} {measure.lower()}<extra></extra>",
                colorbar=dict(title=dict(text=measure, side="right"), tickvals=[math.log10(v + 1) for v in [1, 10, 100, 1000, 10000, 100000] if v <= top],
                              ticktext=[short(v) for v in [1, 10, 100, 1000, 10000, 100000] if v <= top])))
            fig.update_xaxes(side="top", tickangle=-45 if len(grid.columns) > 8 else 0)
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=max(320, 24 * len(grid) + 140), margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Species totals in brackets. Colors are on a log scale; blank cells have none.")


# ---------- priorities ----------
NICE = sorted({f * 10 ** k for k in range(0, 7) for f in (1, 2, 5)})
if "target" not in st.session_state:
    st.session_state.target = int(D["target"])


def set_target(key: str):
    st.session_state.target = max(1, int(st.session_state[key]))


with tabs[2]:
    if tabs[2].open:  # only the open tab runs, so the other tabs cost nothing
        st.subheader("What to do next")
        group_label = st.segmented_control("Group by", ["Species", "Plant type"], default="Species", required=True, key="prio_group")
        key = "species" if group_label == "Species" else "plant_type"
        noun = "species" if key == "species" else "plant types"
        target = st.session_state.target
        st.session_state.t_input, st.session_state.t_slider = target, target  # keep both controls in step
        t1, t2, t3 = st.columns([1, 2.4, 0.8], vertical_alignment="bottom")
        t1.number_input("Target per group (cutouts)", min_value=1, step=10, key="t_input", on_change=set_target, args=("t_input",))
        t2.select_slider("Slide to set the target (log steps)", options=sorted(set(NICE) | {target}), key="t_slider",
                         on_change=set_target, args=("t_slider",), format_func=lambda v: fmt(v))
        t3.button("Use placeholder", on_click=lambda: st.session_state.update(target=int(D["target"])), width="stretch",
                  help=f"Median cutouts per species: {fmt(D['target'])}")

        groups = cube.assign(backlog=backlog_of(cube), dont=cube.images.where(cube.category == "Don't have", 0)) \
            .groupby(key, observed=True)[["cutouts", "backlog", "dont"]].sum().reset_index().rename(columns={key: "name"})
        groups["name"] = groups["name"].astype(str)
        groups["expected"] = (groups.cutouts + (groups.backlog * CPI).round()).astype(int)
        groups["shortfall"] = (target - groups.cutouts).clip(lower=0)
        groups["action"] = [("Met" if c >= target else "Process" if e >= target else "Process + collect" if b > 0 else "Collect")
                            for c, e, b in zip(groups.cutouts, groups.expected, groups.backlog)]
        placeholder = D["target_is_placeholder"] and target == D["target"]
        st.markdown(
            f"Each row is one {'species' if key == 'species' else 'plant type'}. The target is **{fmt(target)}** cutouts per group"
            f"{' (a placeholder: the median species, until we agree on a real number)' if placeholder else ''}. "
            f"**Expected** = cutouts now + {CPI:.1f} per backlog image, the average per annotated image so far.")

        bars = groups.sort_values(["cutouts", "name"], ascending=[True, False])  # plotly draws bottom-up
        n_below = int((groups.cutouts < target).sum())
        st.markdown(f"**Choosing the target** · {fmt(n_below)} of {fmt(len(groups))} {noun} below {fmt(target)} cutouts. "
                    f"{dot(BELOW)}below {dot(ABOVE)}at or above · click a bar to use its count as the target.", unsafe_allow_html=True)
        if bars.empty:
            st.info("Nothing for these filters.")
        else:
            top_v = max(10, target, int(bars.cutouts.max())) * 1.1
            top = next(f * 10 ** math.floor(math.log10(top_v)) for f in (1, 2, 5, 10) if f * 10 ** math.floor(math.log10(top_v)) >= top_v)
            fig = go.Figure(go.Bar(
                y=bars.name, x=bars.cutouts.clip(lower=1), orientation="h",
                marker_color=[BELOW if v < target else ABOVE for v in bars.cutouts],
                text=[fmt(v) for v in bars.cutouts], textposition="outside", cliponaxis=False,
                customdata=bars[["cutouts", "backlog", "expected", "action"]].values,
                hovertemplate="<b>%{y}</b><br>%{customdata[0]:,} cutouts · %{customdata[1]:,} backlog images"
                              "<br>expected %{customdata[2]:,} · %{customdata[3]}<extra></extra>",
                unselected=dict(marker=dict(opacity=1)), selected=dict(marker=dict(opacity=1))))
            fig.add_vline(x=target, line_width=2, line_color="#7f7f7f")
            fig.add_annotation(x=math.log10(target), y=1, xref="x", yref="paper", yanchor="bottom", showarrow=False,
                               text=f"<b>Target {fmt(target)}</b>", bgcolor="#52514e", font=dict(color="#ffffff", size=12),
                               borderpad=3)
            fig.update_xaxes(type="log", range=[0, math.log10(top)], dtick=1, tickformat="~s",
                             title="cutouts per group (log scale)", side="bottom")
            height = 24 * len(bars) + 110
            fig.update_layout(height=height, margin=dict(l=0, r=40, t=36, b=10), bargap=0.3, dragmode=False)
            box = st.container(height=min(640, height + 20)) if height > 640 else st.container()
            event = box.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="target_chart",
                                     config={"displayModeBar": False})
            points = (event or {}).get("selection", {}).get("points", []) if event else []
            if points:  # a click selects one bar; apply it once
                picked = (key, points[0].get("y"))
                if picked != st.session_state.get("last_pick"):
                    st.session_state.last_pick = picked
                    st.session_state.target = max(1, int(groups.set_index("name").cutouts.get(points[0].get("y"), target)))
                    st.rerun()

        counts = groups.action.value_counts()
        st.markdown(" &nbsp; ".join(f"{dot(c)}{a} **{counts.get(a, 0)}**" for a, (c, _) in ACTIONS.items()), unsafe_allow_html=True,
                    help="\n\n".join(f"**{a}**: {tip}" for a, (_, tip) in ACTIONS.items()))
        show_met = st.checkbox("Show groups that meet the target")
        shown = groups if show_met else groups[groups.action != "Met"]
        shown = shown.sort_values(["shortfall", "name"], ascending=[False, True])
        out = shown.rename(columns={"name": "Species" if key == "species" else "Plant type", "cutouts": "Cutouts",
                                    "backlog": "Backlog images", "expected": "Expected cutouts", "shortfall": "Short of target",
                                    "dont": "Can't process", "action": "Action"})[
            ["Species" if key == "species" else "Plant type", "Cutouts", "Backlog images", "Expected cutouts", "Short of target", "Can't process", "Action"]]
        st.dataframe(out, width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="localized") for c in out.columns[1:6]})
        st.download_button("Download table as CSV", out.assign(Target=target).to_csv(index=False), "priorities.csv", "text/csv")


# ---------- data issues ----------
with tabs[3]:
    if tabs[3].open:  # only the open tab runs, so the other tabs cost nothing
        st.subheader("Consistency checks")
        st.caption("Run against the database when the snapshot was built. A failed check means some counts are wrong.")
        st.table(pd.DataFrame([{
            "Check": c["check"], "What it means": c.get("what", ""), "Expected": fmt(c["expected"]), "Actual": fmt(c["actual"]),
            "Result": "Pass" if c["ok"] else "Fail: " + c["hint"]} for c in D["checks"]]).set_index("Check"))

        left, right = st.columns(2)
        with left:
            st.subheader("Images we can't process")
            st.caption("Missing the raw image or the sample metadata.")
            dont = cube[cube.category == "Don't have"].pivot_table(index="location", columns="state", values="images",
                                                                  aggfunc="sum", observed=True, fill_value=0)
            if dont.empty:
                st.caption("None for these filters.")
            else:
                dont.columns = [str(c) for c in dont.columns]
                dont["Total"] = dont.sum(axis=1)
                st.dataframe(dont.sort_values("Total", ascending=False), width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="localized") for c in dont.columns})
        with right:
            st.subheader("Samples with no images")
            st.caption("A sample is a field record in the samples table (species, location, plant type), identified by "
                       "master_ref_id; each image points to its sample. These samples have no image pointing to them: raws not "
                       "uploaded, an ID mismatch, or never photographed. Season is when the sample was recorded in the field app.")
            ns = apply_filters(samples_all).groupby(["plant_type", "location", "season"], observed=True).samples.sum()
            ns = ns[ns > 0].sort_values(ascending=False).reset_index()
            if ns.empty:
                st.caption("None for these filters.")
            else:
                ns.columns = ["Plant type", "Location", "Season recorded", "Samples"]
                st.dataframe(ns, width="stretch", hide_index=True)
                st.caption(f"Total: {fmt(ns.Samples.sum())}")

        left, right = st.columns(2)
        with left:
            st.subheader("Unrecognised annotation status")
            known = ", ".join(f"`{s}`" for s in D["plant_statuses"] + D["no_plant_statuses"])
            st.caption(f"The classification recognises {known}. Images whose first cutout row has any other status are counted "
                       "in the backlog by their raw/developed state.")
            odd = cube[cube.status != ""].groupby(["status", "state"], observed=True).images.sum().reset_index()
            if odd.empty:
                st.caption("None. Every annotation status is recognised.")
            else:
                odd.columns = ["Status", "Counted as", "Images"]
                st.dataframe(odd.sort_values("Images", ascending=False), width="stretch", hide_index=True)
        with right:
            st.subheader("Unknown values")
            st.caption('Images with a blank field. They are counted, but land in an "Unknown" row or column.')
            st.dataframe(pd.DataFrame([{"Field": label, "Images": int(cube.loc[cube[col].astype(str) == "Unknown", "images"].sum()),
                                        "Share": pct(cube.loc[cube[col].astype(str) == "Unknown", "images"].sum(), n_images)}
                                       for label, col in [("Species", "species"), ("Location", "location"),
                                                          ("Plant type", "plant_type"), ("Season (no capture time)", "season")]]),
                         width="stretch", hide_index=True, column_config={"Images": st.column_config.NumberColumn(format="localized")})


# ---------- browse ----------
LINK_MODE = ("image" if GLOBUS_HTTPS and GLOBUS_BATCHES_PATH else
             "folder" if GLOBUS_COLLECTION and GLOBUS_BATCHES_PATH else "")


def globus_folder_link(cutout_path: str) -> str:
    if not (GLOBUS_COLLECTION and GLOBUS_BATCHES_PATH):
        return ""
    return (f"https://app.globus.org/file-manager?origin_id={quote(GLOBUS_COLLECTION)}"
            f"&origin_path={quote(GLOBUS_BATCHES_PATH + '/' + str(Path(cutout_path).parent) + '/')}")


def globus_link(cutout_path: str) -> str:
    """The image itself when the collection's HTTPS URL is set, else its folder in the Globus file manager."""
    if LINK_MODE == "image":
        return f"{GLOBUS_HTTPS}{quote(GLOBUS_BATCHES_PATH + '/' + cutout_path)}"
    return globus_folder_link(cutout_path)


@st.cache_data(show_spinner="Loading the cutout list...")
def load_cutouts(folder: str, mtime: float, link_mode: str) -> pd.DataFrame:
    """Every cutout in the snapshot, with a Globus link per row. mtime and link_mode are cache keys."""
    df = pd.read_parquet(Path(folder) / "cutouts.parquet")
    for col in ("species", "plant_type", "location", "season", "batch"):
        df[col] = df[col].astype("string").fillna("")
    df["link"] = df.cutout_path.map(globus_link) if link_mode else ""
    # Repeated values as categories: about a third of the data to send to the browser for the full list.
    for col in ("species", "plant_type", "location", "season", "batch") + (("link",) if link_mode == "folder" else ()):
        df[col] = df[col].astype("category")
    return df


@st.dialog("Cutout", width="large")
def show_cutout(t: dict):
    full = SNAPSHOT / t["full"] if t.get("full") else None
    st.markdown(f"**{t['species']}** · {t['name']}  \n{t['plant_type']} · {t['location']} · {t['season']}"
                f"{' · batch ' + t['batch'] if t['batch'] else ''} · {fmt(t['w'])} × {fmt(t['h'])} px  \n`{t['path']}`")
    b1, b2 = st.columns(2)
    if full and full.exists():
        b1.download_button("Download full resolution (WebP)", full.read_bytes(), f"{t['name']}.webp", "image/webp", width="stretch")
    link = globus_link(t["path"])
    if link:
        b2.link_button("Open the original in Globus", link, width="stretch")
    # Scale so the whole cutout fits on screen (about 620 px tall); the download keeps every pixel.
    shown_width = max(200, min(1100, int(620 * t["w"] / max(1, t["h"]))))
    st.image(str(full if full and full.exists() else SNAPSHOT / t["thumb"]), width=shown_width)


BROWSE_CONTROLS = [("plant_type", "Plant type", "b_pt"), ("location", "Location", "b_loc"),
                   ("season", "Season", "b_season"), ("species", "Species", "b_species")]


def browse_matches(df: pd.DataFrame, skip: str | None = None) -> pd.DataFrame:
    for col, _, k in BROWSE_CONTROLS:
        v = st.session_state.get(k, "All")
        if col != skip and v != "All":
            df = df[df[col] == v]
    return df


with tabs[4]:
    if tabs[4].open:  # only the open tab runs, so the other tabs cost nothing
        st.subheader("All cutouts")
        list_file = SNAPSHOT / "cutouts.parquet"
        if not list_file.exists():
            st.info("This snapshot has no cutout list yet. It appears after the next snapshot from SUNNY.")
        else:
            everything = load_cutouts(str(SNAPSHOT), list_file.stat().st_mtime, LINK_MODE)
            base = apply_filters(everything)
            cols = st.columns(len(BROWSE_CONTROLS) + 1, vertical_alignment="bottom")
            for (col, label, k), c in zip(BROWSE_CONTROLS, cols):
                avail = browse_matches(base, skip=col)[col].value_counts()
                avail = avail[avail > 0]  # categories keep values that no longer match
                opts = ["All"] + sorted(avail.index, key=lambda v: (D["dims"]["season"].index(v) if col == "season" and v in D["dims"]["season"] else 0, str(v)))
                if st.session_state.get(k, "All") not in opts:
                    st.session_state[k] = "All"
                c.selectbox(label, opts, key=k, format_func=lambda v, a=avail: f"All ({fmt(a.sum())})" if v == "All" else f"{v} ({fmt(a.get(v, 0))})")
            search = cols[-1].text_input("Image name contains", key="b_search", placeholder="e.g. NCA00123")
            view = browse_matches(base)
            if search.strip():
                view = view[view.base_name.str.contains(search.strip(), case=False, regex=False)]

            how = {"image": "Click a link to open the image in Globus (sign in with your NCSU account).",
                   "folder": "Click a link to open that cutout's folder in Globus (sign in with your NCSU account), then click the file.",
                   "": "Globus links aren't set up yet; paths are relative to field-batches on long-term storage."}[LINK_MODE]
            st.caption(f"**{fmt(len(view))}** of {fmt(len(everything))} cutouts match. {how} The filters at the top apply too.")
            table = view.rename(columns={"species": "Species", "plant_type": "Plant type", "location": "Location", "season": "Season",
                                         "batch": "Batch", "base_name": "Image", "cutout_index": "Cutout #",
                                         "cutout_path": "File path", "link": "Link"})
            shown_cols = ["Species", "Plant type", "Location", "Season", "Batch", "Image", "Cutout #", "File path"] + (["Link"] if LINK_MODE else [])
            st.dataframe(table[shown_cols], width="stretch", hide_index=True, height=600,
                         column_config={"Link": st.column_config.LinkColumn("Link", display_text="View image" if LINK_MODE == "image" else "Open folder"),
                                        "Cutout #": st.column_config.NumberColumn(format="plain")})
            root = D.get("field_batches") or D["storage"].get("root") or ""

            def browse_csv(rows=table[shown_cols], root=root) -> str:
                out = rows
                if root:
                    out = out.assign(**{"Full path": root.rstrip("/") + "/" + out["File path"]})
                return out.to_csv(index=False)

            st.download_button(f"Download these {fmt(len(view))} rows as CSV", browse_csv, "cutouts.csv", "text/csv", on_click="ignore")

            if not thumbs_all.empty:
                with st.expander(f"Sample pictures (up to {D['thumbs_per_group']} per species × location)"):
                    gallery = browse_matches(apply_filters(thumbs_all))
                    st.caption("A random handful picked when the snapshot was built, following the filters above. Click View for full size.")
                    if gallery.empty:
                        st.caption("No sample pictures for these filters.")
                    records = gallery.sort_values(["species", "name"]).head(48).to_dict("records")
                    for start in range(0, len(records), 6):
                        for t, c in zip(records[start:start + 6], st.columns(6)):
                            with c:
                                st.image(str(SNAPSHOT / t["thumb"]), width="stretch")
                                st.caption(f"**{t['species']}**  \n{t['name']}  \n{t['location']} · {t['season']}")
                                if st.button("View", key=f"view_{t['name']}", width="stretch"):
                                    show_cutout(t)


# ---------- storage ----------
with tabs[5]:
    if tabs[5].open:  # only the open tab runs, so the other tabs cost nothing
        S = D["storage"]
        if not S.get("scanned"):
            st.info(f"Storage wasn't measured in this snapshot: {S.get('reason', 'unknown reason')}.")
        else:
            disk = disk_all.copy()
            if f_pt:
                disk = disk[disk.plant_type.isin(f_pt)]
            if f_loc:
                disk = disk[disk.location.isin(f_loc)]
            all_by_role = disk_all.groupby("role")[["files", "bytes"]].sum()
            avg = lambda role: (all_by_role.loc[role, "bytes"] / all_by_role.loc[role, "files"]) if role in all_by_role.index and all_by_role.loc[role, "files"] else None  # noqa: E731
            total, files = int(disk.bytes.sum()), int(disk.files.sum())
            by_role = disk.groupby("role").bytes.sum()

            st_counts = cube.groupby("state", observed=False).images.sum()
            backlog = int(backlog_of(cube).sum())
            not_dev = int(st_counts.get("Raw on NFS, not developed", 0) + st_counts.get("Raw in blob only", 0))
            blob_only = int(st_counts.get("Raw in blob only", 0))
            future = [
                ("Cutouts from the backlog", f"{fmt(backlog)} backlog images × {CPI:.1f} cutouts each", round(backlog * CPI), avg("cutouts"), "cutouts"),
                ("Developed JPGs still to make", f"{fmt(not_dev)} images with a raw but no JPG", not_dev, avg("developed"), "developed"),
                ("Raws to copy from blob to NFS", f"{fmt(blob_only)} raws only in blob (Azure) storage", blob_only, avg("raws"), "raws"),
            ]
            future_total = sum(n * a for _, _, n, a, _ in future if a)

            tiles = [("On disk now", size(total), f"{fmt(files)} files in field-batches")]
            tiles += [(ROLE_LABEL[r], size(by_role.get(r, 0)), f"about {size(avg(r))} each" if avg(r) else "no folder found")
                      for r in ("raws", "developed", "cutouts")]
            tiles += [("Still to come", "≈ " + size(future_total), "if the backlog is processed")]
            for col, (label, value, sub) in zip(st.columns(len(tiles)), tiles):
                col.metric(label, value, border=True)
                col.caption(sub)

            st.subheader("By data product")
            st.caption(f"Each product is a subfolder inside the batch folders under `{S['root']}`. "
                       "`unknown` is files sitting directly in a batch folder.")
            prods = disk.groupby("product").agg(files=("files", "sum"), bytes=("bytes", "sum")).sort_values("bytes", ascending=False)
            st.dataframe(pd.DataFrame({
                "Folder": prods.index, "Holds": [ROLE_LABEL.get(S["roles"].get(p, "unknown"), "Unknown") for p in prods.index],
                "Files": prods.files.values, "Size": [size(b) for b in prods.bytes], "Average file": [size(b / f if f else 0) for f, b in zip(prods.files, prods.bytes)],
                "Share": (prods.bytes / max(1, total)).values,
                "File types": [", ".join(f"{e} ({fmt(f)})" for e, f, _ in S["extensions"].get(p, [])) for p in prods.index]}),
                width="stretch", hide_index=True,
                column_config={"Files": st.column_config.NumberColumn(format="localized"),
                               "Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

            unknown = S.get("unknown", {})
            if unknown:
                st.subheader("Unknown folders and files")
                st.caption(f"Data the dashboard can't identify as raws, developed JPGs, cutouts or masks. Paths are relative to "
                           f"`{S['root']}` and cover all data (filters don't apply).")
                for prod, u in unknown.items():
                    prod_bytes = int(disk_all.loc[disk_all["product"] == prod, "bytes"].sum())
                    what = (f"{fmt(u['total'])} loose file{'s' * (u['total'] != 1)} directly in batch folders" if u["kind"] == "files"
                            else f"subfolder in {fmt(u['total'])} batch folder{'s' * (u['total'] != 1)}")
                    with st.expander(f"`{prod}` · {what} · {size(prod_bytes)}"):
                        items = pd.DataFrame(u["items"], columns=["Path", "Files", "Bytes"])
                        items["Size"] = items.Bytes.map(size)
                        st.dataframe(items[["Path", "Files", "Size"]], width="stretch", hide_index=True, height=min(400, 35 * len(items) + 40))
                        if u["total"] > len(u["items"]):
                            st.caption(f"Showing the {fmt(len(u['items']))} largest of {fmt(u['total'])}.")
                all_paths = pd.DataFrame([(p, u["kind"], S["root"].rstrip("/") + "/" + path, f, b)
                                          for p, u in unknown.items() for path, f, b in u["items"]],
                                         columns=["product", "kind", "path", "files", "bytes"])
                st.download_button("Download all paths as CSV", all_paths.to_csv(index=False), "unknown_storage_paths.csv", "text/csv")

            st.subheader("Still to come")
            st.caption("Rough storage the backlog will need, using today's average file size for each product. Counts follow the filters.")
            st.dataframe(pd.DataFrame(
                [{"What": w, "Based on": b, "Files": n, "Average file": size(a) if a else "–",
                  "Estimated size": size(n * a) if a else f"no {r} folder found"} for w, b, n, a, r in future]
                + [{"What": "Total", "Based on": "", "Files": None, "Average file": "", "Estimated size": "≈ " + size(future_total)}]),
                width="stretch", hide_index=True, column_config={"Files": st.column_config.NumberColumn(format="localized")})

            def size_bars(df: pd.DataFrame, by: str, title: str, holder):
                agg = df.dropna(subset=[by]).groupby(by).bytes.sum().sort_values()
                holder.subheader(title)
                if agg.empty:
                    holder.caption("Nothing for these filters.")
                    return
                fig = go.Figure(go.Bar(y=agg.index, x=agg.values, orientation="h", marker_color=CAT_COLORS["Have"],
                                       text=[size(v) for v in agg.values], textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}<br>%{text}<extra></extra>"))
                fig.update_xaxes(visible=False, range=[0, agg.max() * 1.25])
                fig.update_layout(height=max(160, 26 * len(agg) + 40), margin=dict(l=0, r=10, t=10, b=10), bargap=0.3)
                holder.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            left, right = st.columns(2)
            size_bars(disk, "location", "By location", left)
            size_bars(disk, "plant_type", "By plant type", right)

            st.subheader("Storage over time")
            sh = pd.DataFrame(D["history"])
            sh = sh[sh.get("storage_bytes").notna()] if "storage_bytes" in sh else sh.iloc[0:0]
            if len(sh) >= 2:
                fig = go.Figure(go.Scatter(x=sh.refreshed_at.str[:10], y=sh.storage_bytes, mode="lines+markers",
                                           line=dict(color=CAT_COLORS["Have"], width=2), text=[size(v) for v in sh.storage_bytes],
                                           hovertemplate="%{x}<br>%{text}<extra></extra>"))
                fig.update_yaxes(tickformat="~s", ticksuffix="B")
                fig.update_layout(height=260, margin=dict(l=0, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.caption("The trend appears once two weekly snapshots have measured storage.")
            unmatched = S.get("unmatched_batches", 0)
            unreadable = f", {fmt(S['errors'])} entries could not be read" if S["errors"] else ""
            st.caption(
                f"Measured {when(D['generated_at'])} by walking {fmt(S['batches'])} batch folders ({S['seconds']} s{unreadable}). "
                f"Sizes in decimal units (1 TB = 1,000 GB). Files are matched to their image by name ({pct(S['matched_share'], 1)} of bytes); "
                "the rest take the plant type and location most images in their batch folder have. The season filter doesn't apply here"
                + (f", and {fmt(unmatched)} batch folder(s) not in file_status are left out while filters are on." if unmatched else ".")
                + " Blob (Azure) storage isn't measured; only the raws still to copy are estimated.")


# ---------- definitions ----------
with tabs[6]:
    if tabs[6].open:  # only the open tab runs, so the other tabs cost nothing
        rules = {
            "Cutout": "First cutout row has status " + " or ".join(f"`{s}`" for s in D["plant_statuses"]) + ".",
            "Annotated, no plant found": "First cutout row has status " + " or ".join(f"`{s}`" for s in D["no_plant_statuses"]) + ".",
            "No raw (never uploaded)": "No raw in blob storage, no raw on NFS and no developed JPG on NFS.",
            "No metadata": "No matching `samples` row for the image's `master_ref_id`.",
            "Developed, not annotated": "Developed JPG on NFS, not yet annotated.",
            "Raw on NFS, not developed": "Raw on NFS, not developed.",
            "Raw in blob only": "Raw only in blob (Azure) storage.",
        }
        st.subheader("How images are counted")
        st.markdown("One row in `file_status` is one image. Each image gets exactly one state: the first rule below that matches, top to bottom.")
        st.table(pd.DataFrame([{"#": i + 1, "State": s, "Category": D["state_category"][s], "Rule": rules.get(s, "")}
                               for i, s in enumerate(STATES)]).set_index("#"))
        aliases = ", ".join(f"`{a}` → `{b}`" for a, b in D["aliases"].items()) or "None."
        st.markdown(f"""
**Categories.** *Have*: annotation is done. *To be processed*: raw image and metadata exist, processing isn't finished.
*Don't have*: the raw or the metadata is missing, so it can't be processed.

**Samples.** A sample is a field record in `samples` (species, location, plant type), identified by `master_ref_id`.
Images take their species and location from their sample. "Samples with no images" are records no image points to.

**Cutouts.** Every segmented plant in `cutouts` for images in the Cutout state. One image can give several cutouts.
Species comes from the image's sample record, not from individual cutouts.

**Seasons.** From the earliest EXIF capture time in `images`. Plant type `{D['cover_crop_type']}` runs October to
September (e.g. `2025/2026`); everything else is the calendar year. No capture time → `Unknown`.

**Species names merged.** {aliases}

**Priorities.** For each group: *Met* if cutouts ≥ target; *Process* if cutouts plus the expected yield of the backlog
reach the target; *Process + collect* if the backlog helps but falls short; *Collect* if there's no backlog. Expected
yield is {CPI:.2f} cutouts per backlog image: all cutouts so far divided by all annotated images (including those where
no plant was found).

**Storage.** The snapshot walks every batch folder under field-batches and adds up file sizes by the first subfolder
(the data product). Raws, developed JPGs and cutouts are recognised from folder names and file types; anything else is
*Unknown*, listed with its paths. "Still to come" multiplies backlog counts by today's average file sizes.

**Where the numbers come from.** `dashboard/snapshot.py` runs on SUNNY against `field_exploration.db` (read-only),
writes this snapshot, and `scripts/push_snapshot.sh` copies it here. The backlog list is in the snapshot folder as
`to_be_processed.csv`.
""")
        todo_csv = SNAPSHOT / "to_be_processed.csv"
        if todo_csv.exists():
            st.download_button("Download the full backlog (to_be_processed.csv)", todo_csv.read_bytes(), "to_be_processed.csv", "text/csv")
