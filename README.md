# ncaaw-transfer-dashboard

Women’s Division I basketball player analysis repo and dashboard.

This repo is the current source of truth for the WBB product. Copied men’s notebooks, thresholds, and app structure are kept only as methodology references.

Current archetype labels and scores are provisional. They are not finalized women’s models yet and are currently adapted from the men’s reference website.

## Run the app locally

This project is intended to run in a Python 3.13 environment with the packages listed in `requirements.txt` installed.

Start the app from the `app/` directory:

```bash
cd app
python -m shiny run --reload app.py
```

The dashboard is a Python Shiny app. It is not a static GitHub Pages site as currently built.
## Static export for GitHub Pages

This repo also supports a Shinylive static export for GitHub Pages.

Rebuild the static site from the repo root with:

```bash
/Users/adriankong/miniforge3/envs/py313/bin/shinylive export app docs --verbose
```

The generated Pages bundle lives in `docs/`.

## Data

The app expects a processed women’s Division I player dataset. The default processed input path is:

`data/processed/wbb_d1_processed_players.csv`

The raw merged women’s files are currently retained in this repo as reference/source inputs while the pipeline is still being finalized.

## Project structure

- `app/` contains the Shiny dashboard app and data loading code.
- `scripts/` contains dataset build and processing scripts.
- `data/` contains raw/reference inputs and processed outputs.
- `docs/` contains the static Shinylive export used for GitHub Pages.
- `notebooks/` contains exploratory and reference notebooks.

## Notes

- Women’s Division I is the only dashboard scope for this app right now.
- The PCA plot uses women’s processed data and women’s archetype scoring outputs.
- Similarity display is currently shown as a `Similarity Score`, not a literal percent match.

## Shipping scope

The current deployable app is the women’s Division I dashboard plus its processed dataset and static `docs/` export. Reference notebooks and copied raw files can stay in the repo as working materials, but they are not required for the GitHub Pages site itself.
