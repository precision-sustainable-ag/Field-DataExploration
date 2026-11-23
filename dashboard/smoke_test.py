"""Headless smoke test: runs the app against a snapshot and clicks through every tab and
control, failing on any exception. No browser or server needed.

    uv run python dashboard/smoke_test.py [snapshot folder]

The folder defaults to $FIELD_DASHBOARD_DATA, then dashboard/snapshot.
"""
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
APP = str(HERE / "app.py")


def check(at: AppTest, step: str) -> None:
    if at.exception:
        raise SystemExit(f"FAIL after {step}:\n" + "\n".join(str(e.value) for e in at.exception))
    if any("No snapshot found" in e.value for e in at.error):
        raise SystemExit(f"FAIL: {at.error[0].value}")
    print(f"ok  {step}")


def open_tab(at: AppTest, label: str) -> None:
    """Tabs are lazy and bound to ?tab=..., so switching the query parameter opens one."""
    at.query_params["tab"] = label
    at.run()
    check(at, f"opened {label}")


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIELD_DASHBOARD_DATA", str(HERE / "snapshot"))
    os.environ["FIELD_DASHBOARD_DATA"] = folder
    at = AppTest.from_file(APP, default_timeout=120).run()
    check(at, f"first load ({folder})")

    tabs = [t.label for t in at.tabs]
    expected = ["Overview", "Coverage", "Priorities", "Data issues", "Browse", "Storage", "How it's counted"]
    assert tabs == expected, f"tabs are {tabs}"
    print(f"ok  tabs: {', '.join(tabs)}")

    open_tab(at, "Coverage")
    for value in ["Backlog", "Expected cutouts", "Cutouts"]:
        at.segmented_control(key="count").set_value(value).run()
        check(at, f"Coverage count = {value}")
    for value in ["Season", "Plant type", "Location"]:
        at.segmented_control(key="columns").set_value(value).run()
        check(at, f"Coverage columns = {value}")

    open_tab(at, "Priorities")
    at.number_input(key="t_input").set_value(50).run()
    check(at, "Priorities target typed (50)")
    assert at.session_state.target == 50, at.session_state.target
    at.select_slider(key="t_slider").set_value(2000).run()
    check(at, "Priorities target slid (2,000)")
    assert at.session_state.target == 2000, at.session_state.target
    at.segmented_control(key="group").set_value("Plant type").run()
    check(at, "Priorities grouped by plant type")
    at.checkbox[0].check().run()
    check(at, "Priorities: show groups that meet the target")
    at.segmented_control(key="group").set_value("Species").run()
    check(at, "Priorities grouped by species")

    open_tab(at, "Data issues")
    for widget in ("plant_type", "location", "season"):
        options = at.multiselect(key=widget).options
        if options:
            at.multiselect(key=widget).set_value([options[0]]).run()
            check(at, f"global filter {widget} = {options[0]}")
    at.button[0].click().run()  # Reset filters
    check(at, "filters reset")

    open_tab(at, "Browse")
    for k in ("browse_plant_type", "browse_location", "browse_season", "browse_species"):
        box = at.selectbox(key=k)
        if len(box.options) > 1:
            box.set_value(box.options[1]).run()
            check(at, f"Browse {k} = {at.selectbox(key=k).value}")
            at.selectbox(key=k).set_value("All").run()

    # The reported bug: picking WEEDS emptied the other lists until a page refresh. Now a list keeps
    # its own choices, the others narrow to what exists, and nothing ends up empty.
    all_types = at.selectbox(key="browse_plant_type").options
    if "WEEDS" in all_types:
        at.selectbox(key="browse_plant_type").set_value("WEEDS").run()
        check(at, "Browse plant type = WEEDS")
        assert at.selectbox(key="browse_plant_type").options == all_types, "plant type list shrank"
        species = at.selectbox(key="browse_species").options
        assert len(species) > 1, "species list empty after WEEDS"
        at.selectbox(key="browse_species").set_value(species[1]).run()
        check(at, f"Browse species = {species[1]}")
        assert "WEEDS" in at.selectbox(key="browse_plant_type").options
        assert len(at.selectbox(key="browse_location").options) > 1, "location list empty"
        assert len(at.dataframe[0].value) > 0, "table empty"
        at.selectbox(key="browse_species").set_value("All").run()
        assert at.selectbox(key="browse_plant_type").options == all_types, "plant types did not come back"
        check(at, "Browse: lists narrow and come back without a refresh")

    # A shared link with a choice that no longer exists falls back to All instead of an empty table.
    link = AppTest.from_file(APP, default_timeout=120)
    link.query_params.update(tab="Browse", browse_species="no such species", plant_type="WEEDS")
    link.run()
    check(link, "Browse: opened from a link with an unknown species")
    assert link.selectbox(key="browse_species").value == "All", link.selectbox(key="browse_species").value
    assert link.multiselect(key="plant_type").value == ["WEEDS"], link.multiselect(key="plant_type").value
    at.selectbox(key="browse_location").set_value(at.selectbox(key="browse_location").options[1]).run()
    at.button(key="browse_reset").click().run()
    check(at, "Browse: Reset")
    assert all(at.selectbox(key=k).value == "All" for k in ("browse_plant_type", "browse_location", "browse_season", "browse_species"))

    assert at.dataframe, "Browse table missing"
    at.text_input(key="search").set_value("A0").run()
    check(at, "Browse: search by image name")
    at.text_input(key="search").set_value("").run()
    views = [b for b in at.button if b.key and b.key.startswith("view_")]
    if views:
        views[0].click().run()
        check(at, "Browse: opened a cutout full size")
    open_tab(at, "Storage")
    open_tab(at, "How it's counted")
    print("PASS")


if __name__ == "__main__":
    main()
