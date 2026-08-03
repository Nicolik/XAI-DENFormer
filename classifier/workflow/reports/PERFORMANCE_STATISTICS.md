# Model-performance statistical comparisons

## Why this workflow exists

The geographical, temporal, and CD-HIT evaluations contain only a small number of structured outer partitions. Inferential tests on five or six fold-level metric values have low resolution and do not match the reviewers' request for paired analyses on the same held-out genomes. The report workflow therefore treats fold summaries as descriptive and performs inference from aligned out-of-fold predictions.

## Prespecified comparisons

Two families are analyzed separately:

1. Benchmark comparisons: DENFormer-mean versus Longformer, Performer, FFNN, and Logistic Regression.
2. Architecture ablation: all three pairwise comparisons among DENFormer-first, DENFormer-mean, and DENFormer-max.

Each comparison is repeated under the CD-HIT cluster-aware, geographical, and temporal protocols. Holm correction is applied within each family across its prespecified protocol-by-model comparisons.

## Statistical methods

- Primary endpoint: paired difference in accuracy, Model A minus Model B.
- Primary test: two-sided paired sign-flip permutation using the serotype-specific CD-HIT cluster as the exchangeability unit.
- Confidence interval: paired nonparametric bootstrap resampling complete CD-HIT clusters with replacement, stratified by true serotype.
- Sensitivity test: exact two-sided McNemar test on genome-level correctness indicators.
- Secondary endpoints: macro-F1 and per-serotype F1 differences with cluster-bootstrap confidence intervals.

The cluster resampling preserves paired predictions and reduces the risk of treating highly similar genomes as independent observations. Exact McNemar is retained because it is a conventional paired classifier comparison and was explicitly suggested by a reviewer, but the cluster-aware inference should receive primary emphasis.

## Important scope limitation

Every model configuration was trained once using a fixed random seed. These analyses quantify uncertainty in the available held-out predictions; they do not estimate seed-to-seed training variability and should not be described as doing so.

## Run

From the repository root:

```bash
python -m classifier.workflow.reports.make_performance_statistics
```

The default output directory is:

```text
<Data>/logs/NCBI+GISAID-META/aggregate_metrics/performance_statistics/
```

Main outputs:

- `Supplementary_File_S3_Model_Performance_Statistics.xlsx`: complete multi-sheet workbook.
- `csv/`: analysis-friendly exports.

The workbook intentionally does not generate a separate Table S3. Reader-facing primary results are separated into `Benchmark_primary` and `Ablation_primary`; complete fields remain available in `Accuracy_full`.

For a quick validation run before the final analysis:

```bash
python -m classifier.workflow.reports.make_performance_statistics \
  --bootstrap-reps 200 \
  --permutation-reps 1000
```

Use the default 10,000 bootstrap replicates and 100,000 permutation replicates for the manuscript output.

## Rebuild only the Excel workbook

To update the workbook layout from the cached CSV tables without rerunning the
bootstrap or permutation analyses, use:

```bash
python -m classifier.workflow.reports.export_performance_statistics_workbook
```

The command reads the existing files under `performance_statistics/csv/` and
rewrites only `Supplementary_File_S3_Model_Performance_Statistics.xlsx`. The
first export-only run can recover `Methods_parameters` from the existing
workbook; it also stores those values in `csv/methods_parameters.csv` for future
re-exports.

## Legacy prediction files

Older `predictions_test.npz` files may not contain the global `indices` array.
The performance-statistics command now detects this case and reconstructs the
indices from the ordered `test_idx` values in the corresponding CD-HIT,
continent, or temporal split CSV. This does not require rerunning inference: the
inference data loaders use the stored test-index order with `shuffle=False`.
The source used for each comparison is recorded in the `QC_alignment` worksheet.
