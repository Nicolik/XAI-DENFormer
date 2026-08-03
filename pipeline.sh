#!/usr/bin/env bash
set -euo pipefail

# DENFormer public pipeline cheat sheet.
# Run commands from the repository root.
# Uncomment only the block you need; most steps expect paths to be configured in paths.py.

# =============================================================================
# 0) Environment
# =============================================================================
# bash install.sh

# =============================================================================
# 1) CD-HIT preprocessing and cluster-aware folds
# =============================================================================
# Full CD-HIT preprocessing workflow:
# bash cdhit/scripts/00_run_cdhit.sh

# Equivalent individual steps:
# python -m cdhit.scripts.01_run_cdhit_cluster_visualize
# python -m cdhit.scripts.02_make_cluster_aware_kfold_with_splits
# python -m cdhit.scripts.03_annotate_folds

# =============================================================================
# 2) Dataset preprocessing, OHE embeddings, split statistics and dataset figures
# =============================================================================
# Full dataset workflow:
# bash dataset/scripts/00_run_dataset.sh

# Equivalent individual steps:
# python -m dataset.scripts.01_run_ohe
# python -m dataset.scripts.02_run_pie_chart
# python -m dataset.scripts.03_run_continent_splits
# python -m dataset.scripts.04_run_temporal_splits
# python -m cdhit.scripts.03_annotate_folds
# python -m dataset.scripts.05_make_stats_2x2_panel

# =============================================================================
# 3) MSA and serotype variability analysis
# =============================================================================
# Main MSA workflow:
# python -m msa.run --threads 8

# Useful variants:
# python -m msa.run --skip-cdhit-all
# python -m msa.run --skip-cdhit-all --skip-serotype-variability
# python -m msa.run --skip-cdhit-all --skip-serotype-recap

# =============================================================================
# 4) Classifier / XAI workflow
# =============================================================================
# Configure the RUN_* flags and experiment lists in classifier/workflow/workflow.py,
# then run the workflow locally:
# python classifier/workflow/scripts/00_run_workflow.py

# =============================================================================
# 5) Paper quantitative reports
# =============================================================================
# All reports with median and min-max summaries:
# python classifier/workflow/reports/run_reports.py --summary median --error-bar minmax --result-table-percent

# All reports with mean and standard-deviation summaries:
# python classifier/workflow/reports/run_reports.py --summary mean --error-bar std --result-table-percent

# Individual report entry points:
# python classifier/workflow/reports/make_confmat.py
# python classifier/workflow/reports/make_metrics.py --summary median --error-bar minmax --result-table-percent
# python classifier/workflow/reports/make_model_tradeoff.py --summary median --error-bar minmax
# python -m classifier.workflow.reports.make_performance_statistics
#
# Final manuscript settings for performance statistics:
# python -m classifier.workflow.reports.make_performance_statistics \
#   --bootstrap-reps 10000 \
#   --permutation-reps 100000

# =============================================================================
# 6) Optional direct XAI plotting commands
# =============================================================================
# Strategy-consensus panels for all DENFormer pooling variants:
# for pooling in first mean max; do
#   python classifier/workflow/scripts/07_run_attn_strategy_consensus.py \
#     --pooling "$pooling" --one-hot --epochs 100 --k 1 \
#     --strategy-run-names continent timebin cdhit
#   python classifier/workflow/scripts/15_run_gxi_strategy_consensus.py \
#     --pooling "$pooling" --one-hot --epochs 100 --k 1 \
#     --strategy-run-names continent timebin cdhit
# done

# Aggregate XAI embedding plots:
# python classifier/workflow/scripts/10_aggregate_xai_embedding_plots.py --overwrite
# python classifier/workflow/scripts/10_aggregate_xai_embedding_plots.py --skip-per-fold --overwrite
