# Field-AgIR dashboard

A Streamlit dashboard of what Field-AgIR data we have, what we don't, what still needs
processing, and what it takes up on disk. Same tabs as the earlier HTML report:
Overview, Coverage, Priorities, Data issues, Browse, Storage, How it's counted.

## How it fits together

```
SUNNY (research drive mounted)                    hopper-os (VPN only)
  dashboard/snapshot.py  ──builds──▶ snapshot/      scripts/run_dashboard.sh
  scripts/push_snapshot.sh ──rsync over SSH──▶  ~/field-dashboard-data/ ──▶ https://10.76.169.110:8501
```

- `snapshot.py` reads `field_exploration.db` read-only, measures `field-batches` on disk,
  and writes `data.json`, cutout thumbnails and full-resolution copies, and CSVs.
- `app.py` only reads that folder, so hopper-os needs no access to the research drive.
- The app serves HTTPS itself (self-signed certificate, no web server or admin rights).
  hopper-os is only reachable on the NCSU VPN, so only VPN users can open it.

## One-time setup on hopper-os

```bash
cd ~/envs/Field-DataExploration           # branch sbaghba/dashboard
./setup.sh                                # installs uv and the environment
uv add "streamlit==1.65.0" "plotly==7.1.0" pillow
printf 'dashboard/snapshot/\nscripts/dashboard.env\n' >> .gitignore
cp scripts/dashboard.env.example scripts/dashboard.env
scripts/make_cert.sh                      # certificate for this server's IP
```

Streamlit 1.65 is the newest release that supports Python 3.10, which the project pins.

## One-time setup on SUNNY

```bash
git clone https://github.com/precision-sustainable-ag/Field-DataExploration.git ~/envs/Field-DataExploration
cd ~/envs/Field-DataExploration && git checkout sbaghba/dashboard && uv sync
cp scripts/dashboard.env.example scripts/dashboard.env
ssh-keygen -t ed25519                     # skip if ~/.ssh/id_ed25519 exists
ssh-copy-id sbaghba@10.76.169.110         # lets SUNNY copy to hopper-os without a password
scripts/push_snapshot.sh                  # first snapshot (slow: makes all sample pictures)
```

The first snapshot converts every sample picture (a few per species, location and season)
and prints its progress, e.g. `made 1,200/4,000 pictures (3 min)`. Later runs reuse the
pictures that are still picked and only make new ones, so they take a minute or two.
`--workers N` sets how many pictures are made in parallel (default: up to 8).

## Start it, and keep it running

On hopper-os:

```bash
scripts/run_dashboard.sh start            # also: stop, restart, status
uv run python dashboard/smoke_test.py ~/field-dashboard-data
crontab -e                                # add the two lines below
```

```
@reboot      $HOME/envs/Field-DataExploration/scripts/check_dashboard.sh
*/10 * * * * $HOME/envs/Field-DataExploration/scripts/check_dashboard.sh
```

On SUNNY, refresh the snapshot every morning (`crontab -e`):

```
15 6 * * * $HOME/envs/Field-DataExploration/scripts/push_snapshot.sh >> $HOME/push_snapshot.log 2>&1
```

## Opening it

On the NCSU VPN, open **https://10.76.169.110:8501** (or https://lightning.cals.ncsu.edu:8501).
The browser warns once that the certificate isn't from a known authority: choose Advanced,
then continue. The connection is encrypted; a trusted certificate has to come from the team
that runs lightning.

Light or dark: the ⋮ menu at the top right has System / Light / Dark. The browser remembers it.

Links keep what's on screen. The tab, the filters at the top, the Coverage and Priorities
choices and the Browse choices are all in the address, so copying it shares that exact view,
e.g. `https://10.76.169.110:8501/?tab=Browse&plant_type=WEEDS&browse_species=waterhemp`.
A choice that no longer exists in the data falls back to All.

The Overview numbers show the change since the previous database rebuild (only when no filters
are set), and **Backlog by step** splits "To be processed" into not copied from blob, not
developed and not annotated, with the backlog CSV next to it. The per-step changes start after
two rebuilds with this version, because older history rows don't have those counts.

If the database itself hasn't been rebuilt for more than 8 days, a yellow banner says so: the
snapshot still updates daily, but the numbers stay the same until `field_exploration.db` is
rebuilt.

If the morning update fails, the page shows a red banner with when and why (from
`status.json`, which `push_snapshot.sh` sends even when the build fails). If no update arrives
for two days, a yellow banner says so.

## Updating

`git pull`, then `scripts/run_dashboard.sh restart` on hopper-os. Changes to `snapshot.py`
also need a `git pull` on SUNNY; they show up after the next snapshot.

## Globus links on Browse

Browse lists every cutout (species, location, season, batch, file path), with all
matching rows on screen and a CSV download. The Browse dropdowns narrow each other to
combinations that exist; **Reset** sets them back to All. Each row has two Globus links:

- **View image** opens the cutout file itself (`GLOBUS_HTTPS_URL` + path).
- **Open folder** opens its folder in the Globus file manager (`GLOBUS_COLLECTION_ID` + path).

The values for NC State Research Storage are already in `dashboard.env.example`:

```
GLOBUS_COLLECTION_ID=2f7f6170-8d5c-11e9-8e6a-029d279f7e24
GLOBUS_FIELD_BATCHES_PATH=/rsstu/users/r/raatwell/longterm_images3/field-batches
GLOBUS_HTTPS_URL=https://m-ae9b45.7ce1a.03c0.gaccess.io
```

After changing them, `scripts/run_dashboard.sh restart`. Viewers sign in to Globus with their
NCSU account and need access to that collection.

"Sample pictures" (collapsed, under the table) shows a few cutouts per species, location and
season, so every season appears. The same pictures stay from day to day unless new data
changes the pick.

## Troubleshooting

| Symptom | Check |
|---|---|
| "No snapshot found" | `ls ~/field-dashboard-data/data.json`; run `scripts/push_snapshot.sh` on SUNNY |
| Red "latest daily update failed" banner | The banner gives the step and error; full output in `~/push_snapshot.log` on SUNNY |
| Yellow "No new data for N days" | SUNNY's cron: `crontab -l`, `tail ~/push_snapshot.log` |
| No ⋮ theme switch | `.streamlit/config.toml` must have both `[theme.light]` and `[theme.dark]` |
| Page doesn't load | `scripts/run_dashboard.sh status`; log in `~/.local/state/field-dashboard/` |
| No cutouts on Browse | snapshot printed "No cutout could be read": pass `--storage` to `snapshot.py` |
