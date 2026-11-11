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
scripts/push_snapshot.sh                  # first snapshot (a few minutes)
```

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

On the NCSU VPN, open **https://10.76.169.110:8501**. The browser warns once that the
certificate isn't from a known authority: choose Advanced, then continue. The connection is
encrypted; a certificate browsers trust would need NC State IT.

## Updating

`git pull`, then `scripts/run_dashboard.sh restart` on hopper-os. Changes to `snapshot.py`
also need a `git pull` on SUNNY; they show up after the next snapshot.

Each tab has its own address, so you can share a link straight to one, e.g.
`https://10.76.169.110:8501/?tab=Browse`.

## Globus links on Browse

Browse lists every cutout (species, location, season, batch, file path), with all
matching rows on screen and a CSV download. Each row can link to the real file in Globus:

1. In the Globus website (app.globus.org), sign in with NCSU and open the
   `field-batches` folder on the collection that holds long-term storage.
2. From the browser's address bar, copy `origin_id=...` into `GLOBUS_COLLECTION_ID`
   and `origin_path=...` into `GLOBUS_FIELD_BATCHES_PATH` in `scripts/dashboard.env`.
3. Optional: if the collection's overview page shows an **HTTPS URL**, put it in
   `GLOBUS_HTTPS_URL`. Links then open the image itself instead of its folder.
4. `scripts/run_dashboard.sh restart`.

Viewers need Globus access to that collection to open the links.

## Troubleshooting

| Symptom | Check |
|---|---|
| "No snapshot found" | `ls ~/field-dashboard-data/data.json`; run `scripts/push_snapshot.sh` on SUNNY |
| "This snapshot is N days old" | SUNNY's cron: `tail ~/push_snapshot.log` |
| Page doesn't load | `scripts/run_dashboard.sh status`; log in `~/.local/state/field-dashboard/` |
| No cutouts on Browse | snapshot printed "No cutout could be read": pass `--storage` to `snapshot.py` |
