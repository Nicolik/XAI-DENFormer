# Computational-efficiency benchmark

This benchmark measures all classifier configurations under one controlled CUDA allocation:

- Logistic Regression;
- FFNN;
- DENFormer-first;
- DENFormer-mean;
- DENFormer-max;
- Longformer;
- Performer.

The default protocol uses the first CD-HIT cluster-aware fold, batch size 32, one complete warm-up epoch, three measured epochs, ten inference warm-up batches, and three measured full-test inference passes. Training models are initialized from scratch. Inference uses the original 100-epoch checkpoints.

Run the benchmark from the repository root:

```bash
python -u classifier/workflow/scripts/17_run_efficiency_benchmark.py --force-benchmark
```

Outputs are written below the configured data directory:

```text
<Data>/benchmarks/efficiency/run_<UTC timestamp>_<run identifier>/
```

The main files are:

- `training_measurements.csv`: warm-up and measured epoch-level records;
- `inference_measurements.csv`: repeated test-pass records;
- `efficiency_summary.csv`: complete model-level statistics;
- `efficiency_paper_table.csv`: compact table for manuscript reporting;
- `efficiency_paper_table.md`: Markdown version of the compact table;
- `efficiency_paper_table.xlsx`: formatted Excel table ready for supplementary reporting;
- `environment.json`: hardware, CUDA, software, dataset, and protocol metadata;
- `efficiency_results.json`: machine-readable combined results.

## Reusing completed results

By default, the benchmark entry point checks `benchmarks/efficiency/latest_run.txt`. If a completed run containing the measurement and summary CSV files is found, it regenerates the paper-table CSV, Markdown, and Excel files and exits without loading the dataset, checkpoints, or CUDA device.

To export only the table from the latest completed run:

```bash
python -u classifier/workflow/scripts/17_run_efficiency_benchmark.py --table-only
```

To reuse a specific run:

```bash
python -u classifier/workflow/scripts/17_run_efficiency_benchmark.py \
  --table-only \
  --source-run <Data>/benchmarks/efficiency/run_<identifier>
```

The benchmark supports scheduler wrappers, but machine-specific SLURM files and paths are intentionally kept outside the public export.
