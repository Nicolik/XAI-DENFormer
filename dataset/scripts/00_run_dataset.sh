#!/usr/bin/env bash
set -euo pipefail

python -m dataset.scripts.01_run_ohe
python -m dataset.scripts.02_run_pie_chart
python -m dataset.scripts.03_run_continent_splits
python -m dataset.scripts.04_run_temporal_splits
python -m cdhit.scripts.03_annotate_folds
python -m dataset.scripts.05_make_stats_2x2_panel
