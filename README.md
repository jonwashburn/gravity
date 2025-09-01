# Gravity: Information-Limited Gravity (ILG) – Production

This repo contains a top-level production pipeline to reproduce the galaxy rotation results and artifacts referenced in the paper.

- Scripts: `active/scripts/`
  - `build_sparc_master_table.py`
  - `ledger_final_combined.py`
  - `reproduce_048_fit.py`
  - `visualize_best_fits.py`
  - `plot_example_rc.py`
  - `make_release_bundle.sh`
- Data: `active/data/Rotmod_LTG` (symlink to SPARC-like rotation curves)
- Results: `active/results/` (master table, combined solver outputs, reproduction artifacts)
- Paper sources: `active/paper/` (`dark-matter-galaxy-rotation.tex`, `Gravity-derived.tex`)

Legacy materials are preserved under `archives/`.

## Quickstart

1) Create a venv and install requirements:

```
cd production
python3 -m venv env/.venv
source env/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/requirements.txt
```

2) Build the master table and make an example figure:

```
cd scripts
python build_sparc_master_table.py
python plot_example_rc.py  # emits ../results/example_rc.pdf
```

3) Run the combined solver:

```
python ledger_final_combined.py
# Outputs to ../results/
```

4) Reproduce the 0.48 fit and visualize:

```
python reproduce_048_fit.py
python visualize_best_fits.py
# Outputs to ../results/
```

5) Create a release bundle (paper + scripts + minimal artifacts):

```
chmod +x make_release_bundle.sh
./make_release_bundle.sh
# Archives under active/results/release/
```

## Notes
- Legacy docs moved to `archives/legacy/`.
- `.gitignore` excludes local virtualenv under `active/env/.venv/`.
- For reproducibility, pin your environment or use containers if desired.


## Structure
- active/: current pipeline (scripts, data link, results, paper)
- legacy/: archived materials (archives moved here)

Data link: active/data/Rotmod_LTG -> ../../legacy/archives/snapshot-20250816-182339-tree/data/Rotmod_LTG
