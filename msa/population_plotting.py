from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import REGION_ORDER
from .plotting import (
    DNA_ENTROPY_MAX_BITS,
    _find_coordinate_map,
    _plot_stacked_genome_profiles,
    _site_profile_to_genome_coordinates,
    load_gene_boundaries,
)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {r.replace("'", ""): i for i, r in enumerate(REGION_ORDER)}
    out = df.copy()
    out["_order"] = out["region"].map(order).fillna(999)
    return out.sort_values("_order").drop(columns="_order")


def _barplot(df: pd.DataFrame, y_col: str, ylabel: str, out_path: Path) -> None:
    df = _ordered(df)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(df["region"], df[y_col])
    ax.set_xlabel("DENV genomic region")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _grouped_identity_plot(df: pd.DataFrame, out_path: Path) -> None:
    df = _ordered(df)
    regions = list(df["region"])
    x = range(len(regions))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar([i - width / 2 for i in x], df["mean_within_serotype_identity"], width, label="Within serotype")
    ax.bar([i + width / 2 for i in x], df["mean_between_serotype_identity"], width, label="Between serotypes")
    ax.set_xticks(list(x))
    ax.set_xticklabels(regions, rotation=45)
    ax.set_xlabel("DENV genomic region")
    ax.set_ylabel("Mean pairwise identity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_population_summary(results_dir: Path, plots_dir: Path) -> None:
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "population_region_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    df = pd.read_csv(summary_path)

    _barplot(df, "mean_global_entropy", "Mean global nucleotide entropy", plots_dir / "cdhit_region_mean_global_entropy.png")
    if "mean_global_consensus_similarity" in df.columns:
        _barplot(df, "mean_global_consensus_similarity", "Mean global consensus similarity", plots_dir / "cdhit_region_mean_global_consensus_similarity.png")
    _barplot(df, "mean_within_serotype_entropy", "Mean within-serotype entropy", plots_dir / "cdhit_region_mean_within_serotype_entropy.png")
    _barplot(df, "mean_serotype_informativeness", "Mean serotype informativeness", plots_dir / "cdhit_region_mean_serotype_informativeness.png")
    _barplot(df, "identity_gap_within_minus_between", "Identity gap: within - between", plots_dir / "cdhit_region_identity_gap.png")
    _barplot(df, "global_variable_site_fraction", "Global variable site fraction", plots_dir / "cdhit_region_global_variable_site_fraction.png")
    _grouped_identity_plot(df, plots_dir / "cdhit_region_within_between_identity.png")
    print(f"[OK] Saved CD-HIT population plots in {plots_dir}")


def plot_population_nucleotide_level_profiles(
    results_dir: Path,
    plots_dir: Path,
    coordinates_dir: Path,
    coordinate_map_name: str | None = None,
    prefix: str = "cdhit_msa",
) -> None:
    sites_path = Path(results_dir) / "population_site_metrics_all_regions.csv"
    if not sites_path.exists():
        raise FileNotFoundError(sites_path)

    sites = pd.read_csv(sites_path)
    coordinate_map = _find_coordinate_map(Path(coordinates_dir), coordinate_map_name)
    gene_boundaries = load_gene_boundaries(coordinate_map)
    print(f"[INFO] Using coordinate map for nucleotide-level CD-HIT MSA plots: {coordinate_map}")

    entropy_profile = _site_profile_to_genome_coordinates(sites, gene_boundaries, "global_entropy")
    _plot_stacked_genome_profiles(
        {"CD-HIT MSA global entropy": entropy_profile},
        "global_entropy",
        "Entropy (bits)",
        gene_boundaries,
        Path(plots_dir) / f"{prefix}_nucleotide_entropy_stacked_with_genes.png",
        ylim=(0, DNA_ENTROPY_MAX_BITS),
    )

    if "global_consensus_similarity" not in sites.columns:
        sites["global_consensus_similarity"] = 1.0 - (sites["global_entropy"] / DNA_ENTROPY_MAX_BITS)
    similarity_profile = _site_profile_to_genome_coordinates(sites, gene_boundaries, "global_consensus_similarity")
    _plot_stacked_genome_profiles(
        {"CD-HIT MSA consensus similarity": similarity_profile},
        "global_consensus_similarity",
        "Similarity",
        gene_boundaries,
        Path(plots_dir) / f"{prefix}_nucleotide_similarity_stacked_with_genes.png",
        ylim=(0, 1),
    )

    informativeness_profile = _site_profile_to_genome_coordinates(sites, gene_boundaries, "serotype_informativeness")
    _plot_stacked_genome_profiles(
        {"CD-HIT MSA serotype informativeness": informativeness_profile},
        "serotype_informativeness",
        "Informativeness (bits)",
        gene_boundaries,
        Path(plots_dir) / f"{prefix}_nucleotide_serotype_informativeness_stacked_with_genes.png",
        ylim=(0, DNA_ENTROPY_MAX_BITS),
    )
