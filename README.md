# DENFormer

DENFormer is a research codebase for dengue virus serotype classification from complete genomic sequences and for downstream explainability analyses of learned genomic signals.

The repository includes utilities to prepare dengue genome datasets, generate CD-HIT cluster-aware splits, run multiple sequence alignment (MSA)-based variability analyses, train and evaluate classification models, compute attention- and gradient-based explanation profiles, and generate the quantitative and XAI panels used for paper-level reporting.

## Repository structure

```text
.
├── cdhit/                 # CD-HIT preprocessing, cluster parsing, and cluster-aware folds
├── dataset/               # Dataset preparation, one-hot encoding, split statistics, and dataset figures
├── msa/                   # MSA and serotype variability analyses
├── classifier/
│   ├── model/             # Neural-network model definitions
│   └── workflow/          # Training, inference, XAI, aggregation, consensus panels, and reports
├── paths.py               # Data/output path configuration
├── pipeline.sh            # Command cheat sheet for the main workflows
└── install.sh             # Installation helper
```

## Data availability and expected inputs

This repository does not redistribute the raw dengue genome data used in the associated experiments. In particular, sequences obtained from databases with access or redistribution restrictions must be downloaded by users from the original sources and used according to the corresponding terms of use.

The code expects a data directory with subfolders for genomes, CD-HIT outputs, splits, reference sequences, embeddings, logs, MSA outputs, and statistics. In the public export, the data root is configured through the `DENFORMER_DATA_DIR` environment variable:

```bash
export DENFORMER_DATA_DIR=/path/to/data
```

If the variable is not set, the public version falls back to a local `data/` directory.
Local deployments can customize paths by setting DENFORMER_DATA_DIR or by adapting paths.py.

## Installation

A conda/mamba environment is recommended because the workflow depends on both Python packages and bioinformatics tools such as CD-HIT and MAFFT.

```bash
mamba env create -f environment.yml
conda activate denformer
```

Alternatively, use the helper script:

```bash
bash install.sh
```

The main dependencies include PyTorch, NumPy, pandas, scikit-learn, SciPy, Biopython, matplotlib, statsmodels, openpyxl, CD-HIT, MAFFT, Hugging Face Transformers, and Performer-PyTorch.

## Quick start

The root-level `pipeline.sh` file is a commented command cheat sheet. Commands are intentionally commented because several steps can be computationally expensive and require the expected data layout.

Typical execution order:

```bash
# 1. CD-HIT preprocessing and cluster-aware folds
bash cdhit/scripts/00_run_cdhit.sh

# 2. Dataset preparation, embeddings, split statistics, and dataset figures
bash dataset/scripts/00_run_dataset.sh

# 3. MSA and serotype variability analyses
python -m msa.run --threads 8

# 4. Classifier training, inference, XAI, and aggregation workflow
python classifier/workflow/scripts/00_run_workflow.py

# 5. Paper-level quantitative reports
python classifier/workflow/reports/run_reports.py --summary median --error-bar minmax --result-table-percent
```

## CD-HIT preprocessing

The CD-HIT module prepares sequence clustering outputs and cluster-aware cross-validation folds.

Full workflow:

```bash
bash cdhit/scripts/00_run_cdhit.sh
```

Equivalent individual steps:

```bash
python -m cdhit.scripts.01_run_cdhit_cluster_visualize
python -m cdhit.scripts.02_make_cluster_aware_kfold_with_splits
python -m cdhit.scripts.03_annotate_folds
```

## Dataset preparation

The dataset workflow generates one-hot encoded representations, dataset statistics, geographical and temporal splits, and dataset summary panels.

```bash
bash dataset/scripts/00_run_dataset.sh
```

Equivalent individual steps:

```bash
python -m dataset.scripts.01_run_ohe
python -m dataset.scripts.02_run_pie_chart
python -m dataset.scripts.03_run_continent_splits
python -m dataset.scripts.04_run_temporal_splits
python -m cdhit.scripts.03_annotate_folds
python -m dataset.scripts.05_make_stats_2x2_panel
```

## MSA and serotype variability

The `msa` package runs the alignment and serotype variability analyses used to provide biological context for the XAI profiles.

Main entry point:

```bash
python -m msa.run --threads 8
```

Useful variants:

```bash
python -m msa.run --skip-cdhit-all
python -m msa.run --skip-cdhit-all --skip-serotype-variability
python -m msa.run --skip-cdhit-all --skip-serotype-recap
```

The shell wrapper is also available:

```bash
bash msa/scripts/run_msa.sh --threads 8
```

## Classifier and XAI workflow

The classifier workflow is implemented in:

```text
classifier/workflow/
```

Run the main workflow with:

```bash
python classifier/workflow/scripts/00_run_workflow.py
```

Before running it, configure the experiment flags and model/split settings in:

```text
classifier/workflow/workflow.py
```

The workflow supports the following analysis steps:

- model training;
- standard inference;
- attention extraction;
- attention profile aggregation and genomic-region panels;
- gradient × input attribution;
- XAI embedding projections;
- strategy-consensus XAI panels;
- overall XAI profiles;
- optional quantitative report generation.

Scripts in `classifier/workflow/scripts/` follow the main workflow order, while scripts in `classifier/workflow/reports/` generate quantitative reports and paper-level summary figures. Prediction-level model comparisons are documented in `classifier/workflow/reports/PERFORMANCE_STATISTICS.md`.

## Reports and paper figures

Run all quantitative reports with:

```bash
python classifier/workflow/reports/run_reports.py --summary median --error-bar minmax --result-table-percent
```

Reports can also be generated individually:

```bash
python classifier/workflow/reports/make_confmat.py
python classifier/workflow/reports/make_metrics.py --summary median --error-bar minmax --result-table-percent
python classifier/workflow/reports/make_model_tradeoff.py --summary median --error-bar minmax
python -m classifier.workflow.reports.make_performance_statistics \
  --bootstrap-reps 10000 \
  --permutation-reps 100000

# Rebuild only the Excel workbook from cached CSV outputs.
python -m classifier.workflow.reports.export_performance_statistics_workbook
```

The statistical command writes `Supplementary_File_S3_Model_Performance_Statistics.xlsx` plus analysis-friendly CSV exports. It does not generate a separate Table S3 workbook. The reader-facing primary worksheets are `Benchmark_primary` and `Ablation_primary`; detailed primary, secondary, quality-control, and method fields are retained in the remaining worksheets. The export-only command rewrites the workbook from those cached CSVs without rerunning bootstrap or permutation analyses.

## Installation check

After installation, the following commands can be used to verify that the main modules are importable and that the command-line entry points are available:

```bash
python -m compileall -q classifier cdhit dataset msa paths.py
python -m msa.run --help
python -m classifier.workflow.reports.run_reports --help
```

## Citation

If you use this repository for research, please cite the associated paper.

## License

This project is released under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for details.
