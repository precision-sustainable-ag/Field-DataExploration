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


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIELD_DASHBOARD_DATA", str(HERE / "snapshot"))
    os.environ["FIELD_DASHBOARD_DATA"] = folder
    at = AppTest.from_file(APP, default_timeout=120).run()
    check(at, f"first load ({folder})")

    tabs = [t.label for t in at.tabs]
    expected = ["Overview", "Coverage", "Priorities", "Data issues", "Browse", "Storage", "How it's counted"]
    assert tabs == expected, f"tabs are {tabs}"
    print(f"ok  tabs: {', '.join(tabs)}")

    for value in ["Backlog", "Expected cutouts", "Cutouts"]:
        at.segmented_control(key="cov_measure").set_value(value).run()
        check(at, f"Coverage count = {value}")
    for value in ["Season", "Plant type", "Location"]:
        at.segmented_control(key="cov_cols").set_value(value).run()
        check(at, f"Coverage columns = {value}")

    at.number_input(key="t_input").set_value(50).run()
    check(at, "Priorities target typed (50)")
    assert at.session_state.target == 50, at.session_state.target
    at.select_slider(key="t_slider").set_value(2000).run()
    check(at, "Priorities target slid (2,000)")
    assert at.session_state.target == 2000, at.session_state.target
    at.segmented_control(key="prio_group").set_value("Plant type").run()
    check(at, "Priorities grouped by plant type")
    at.checkbox[0].check().run()
    check(at, "Priorities: show groups that meet the target")
    at.segmented_control(key="prio_group").set_value("Species").run()
    check(at, "Priorities grouped by species")

    for widget in ("f_pt", "f_loc", "f_season"):
        options = at.multiselect(key=widget).options
        if options:
            at.multiselect(key=widget).set_value([options[0]]).run()
            check(at, f"global filter {widget} = {options[0]}")
    at.button[0].click().run()  # Reset filters
    check(at, "filters reset")

    for k in ("b_pt", "b_loc", "b_season", "b_species"):
        box = at.selectbox(key=k)
        if len(box.options) > 1:
            box.set_value(box.options[1].rsplit(" (", 1)[0]).run()  # options are shown as "name (count)"
            check(at, f"Browse {k} = {at.selectbox(key=k).value}")
            at.selectbox(key=k).set_value("All").run()
    views = [b for b in at.button if b.key and b.key.startswith("view_")]
    if views:
        views[0].click().run()
        check(at, "Browse: opened a cutout full size")
    print("PASS")


if __name__ == "__main__":
    main()
