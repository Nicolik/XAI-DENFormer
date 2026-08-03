# Sequence-length and sequence-quality baseline

This report tests whether dengue serotype can be predicted without nucleotide-order information, using only simple properties of each original FASTA record.

## Features

The script extracts original sequence length, implied terminal-padding fraction, N fraction, non-N IUPAC ambiguity fraction, gap fraction, other non-ACGT fraction, total non-ACGT fraction, and GC fraction among unambiguous A/C/G/T bases.

## Models

The inexpensive baselines are majority-class prediction; Logistic Regression using length only, N fraction only, GC fraction only, length plus GC, length/N/GC, or the complete quality-feature set; and HistGradientBoosting using the complete quality-feature set. No hyperparameter search is performed.

## Exact correspondence with the original experiments

The baseline uses the same sources as the original model evaluation:

- `seq_ids_ohe.txt` defines the sample order of `dataset_ohe.h5`;
- `label_matrix.txt` provides the ground-truth targets;
- `classifier.workflow.utils.load_and_validate_folds` loads the geographical, temporal, and CD-HIT train/validation/test indices.

No descriptive `serotype`, `sequence_id`, or other metadata columns from the split CSV files are interpreted by this analysis.

## Run

From the repository root:

```bash
python -m classifier.workflow.reports.make_sequence_quality_baseline
```

This is a CPU-only analysis. It does not rebuild one-hot embeddings, train DENFormer, run neural-model inference, or recompute XAI analyses.

Default output directory:

```text
<Data>/logs/NCBI+GISAID-META/aggregate_metrics/sequence_quality_baseline/
```

The detailed reader-facing file is `Sequence_Quality_Baseline.xlsx`. The Word-ready table is saved separately as `Supplementary_Table_Sequence_Quality_Baselines.xlsx`. It contains only the table: baseline, metric, and the three validation protocols. Each baseline occupies three metric rows (accuracy, balanced accuracy, and macro-F1), and every numerical cell contains one atomic percentage value. Both generated workbooks use Helvetica 8 as the shared default font. Detailed predictions are stored separately as compressed CSV.

## Interpretation

Length-only and N-only baselines showed limited predictive performance. GC-containing baselines captured substantial serotype-associated information but remained below the genome-based models. These analyses show that simple global properties alone do not account for the reported classification performance; they do not prove that the sequence models never use padding or non-ACGT positions.
