import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)


def ensure_palette_covers(palette: dict, values) -> dict:
    """Extends a fixed hue palette with fallback colors for any value actually
    present in the data but missing from the palette (e.g. a new PlantType like
    SOILS/COTTONFLOWERS that predates this palette). seaborn raises a hard
    ValueError on a dict palette missing a key instead of degrading gracefully -
    this keeps that from crashing report generation."""
    missing = sorted({v for v in values if pd.notna(v)} - set(palette))
    if not missing:
        return palette
    log.warning(f"Palette missing colors for {missing}, assigning fallback colors")
    fallback_colors = sns.color_palette("tab10", n_colors=len(missing))
    return {**palette, **dict(zip(missing, fallback_colors))}


def plot_unique_samples(
    df: pd.DataFrame,
    hue_col: str,
    palette: dict,
    title: str,
    save_path: str,
    known_states: list,
    hue_order: list = None,
) -> None:
    """Bar plot of unique MasterRefID counts by UsState/hue_col, backfilling any
    known location with zero matching samples so it still shows up as a zero bar.
    Shared by the all-years and current-season 'samples by state and plant type'
    plots (report.py/plot_by_season.py) - they differed only in hue_col, palette,
    title and save_path, and only one of the two backfilled missing states
    (plan section 3.6)."""
    data = df[df["HasMatchingJpgAndRaw"] == True]
    unique_ids_count = (
        data.groupby(["UsState", hue_col])["MasterRefID"].nunique().reset_index()
    )

    existing_states = set(unique_ids_count["UsState"])
    missing_states = [state for state in known_states if state not in existing_states]
    unique_ids_count = pd.concat(
        [unique_ids_count, pd.DataFrame({"UsState": missing_states})], ignore_index=True
    ).sort_values(by="UsState")

    palette = ensure_palette_covers(palette, unique_ids_count[hue_col])
    if hue_order is not None:
        hue_order = list(hue_order) + [v for v in palette if v not in hue_order]

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_plot = sns.barplot(
            data=unique_ids_count,
            x="UsState",
            y="MasterRefID",
            hue=hue_col,
            palette=palette,
            hue_order=hue_order,
            ax=ax,
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(title)
        ax.text(0.0725, -0.125, "$^{*}$HasMatchingJpgAndRaw = True", ha="center", fontsize=9, transform=ax.transAxes)
        ax.set_ylabel("# MasterRefIDs (samples)")
        ax.set_xlabel("State Location")
        ax.legend(title="Plant Type")
        for bar_container in bar_plot.containers:
            ax.bar_label(bar_container, label_type="edge", padding=3, fontsize=7)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
