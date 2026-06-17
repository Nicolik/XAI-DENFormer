from pathlib import Path
import os

# ============================================================
# PUBLIC PATH CONFIGURATION
# ============================================================
# Set DENFORMER_DATA_DIR to the directory containing the project data.
# Example:
#   export DENFORMER_DATA_DIR=/path/to/data
# If unset, a local ./data directory is used.

_data_dir = os.environ.get("DENFORMER_DATA_DIR", "data")


def as_path(path) -> Path:
    return Path(path).expanduser().resolve()


data_dir = as_path(_data_dir)

# ============================================================
# DATA DIRECTORIES
# ============================================================

genomes_dir = data_dir / "genomes" / "NCBI+GISAID-META"
cdhit_dir = data_dir / "cdhit" / "NCBI+GISAID-META"
splits_dir = data_dir / "splits" / "NCBI+GISAID-META"
refseq_dir = data_dir / "refseq"

msa_dir = data_dir / "msa" / "refseq"
msa_refseq_map_dir = msa_dir / "map"
# Backward-compatible alias. New code should use msa_refseq_map_dir.
map_dir = msa_refseq_map_dir
msa_region_fasta_dir = msa_dir / "region_fastas"
msa_alignment_dir = msa_dir / "alignments"
msa_results_dir = msa_dir / "results"
msa_plots_dir = msa_dir / "plots"

logs_dir = data_dir / "logs" / "NCBI+GISAID-META"
logs_mock_dir = data_dir / "logs" / "mock"

embeddings_dir = data_dir / "embeddings" / "NCBI+GISAID-META"
stats_dir = data_dir / "stats" / "NCBI+GISAID-META"

# ============================================================
# AUTO-CREATE DIRECTORIES
# ============================================================

ALL_DIRS = [
    genomes_dir,
    cdhit_dir,
    splits_dir,
    map_dir,
    refseq_dir,
    msa_dir,
    msa_refseq_map_dir,
    msa_region_fasta_dir,
    msa_alignment_dir,
    msa_results_dir,
    msa_plots_dir,
    logs_dir,
    logs_mock_dir,
    embeddings_dir,
    stats_dir,
]

for p in ALL_DIRS:
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# SPLIT FILES
# ============================================================

split_files = {
    "continent": splits_dir / "dengue_leave_one_continent_out_splits.csv",
    "timebin": splits_dir / "dengue_leave_one_timebin_out_c2006_b5_splits.csv",
    "cdhit": splits_dir / "dengue_cdhit_cluster_aware_kfold_5_identity_0.95_splits.csv",
}
