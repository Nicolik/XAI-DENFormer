from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .config import REGION_ORDER


DNA_ENTROPY_MAX_BITS = 2.0


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {r.replace("'", ""): i for i, r in enumerate(REGION_ORDER)}
    return df.assign(_order=df["region"].map(order).fillna(999)).sort_values("_order").drop(columns="_order")


def _clean_region(region: str) -> str:
    return str(region).replace("'", "")


def _find_coordinate_map(coordinates_dir: Path, preferred_name: str | None = None) -> Path:
    coordinates_dir = Path(coordinates_dir)
    candidates = []
    if preferred_name:
        candidates.append(coordinates_dir / preferred_name)
    candidates.extend([
        coordinates_dir / "coordinates_dengue_LONGEST.csv",
        coordinates_dir / "coordinates_dengue_DENV1_LONGEST.csv",
        coordinates_dir / "coordinates_dengue_DENV1.csv",
    ])
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(coordinates_dir.glob("coordinates_dengue_*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No coordinate map found in {coordinates_dir}")


def load_gene_boundaries(coordinate_map_path: Path) -> dict[str, tuple[int, int]]:
    df = pd.read_csv(coordinate_map_path)
    boundaries: dict[str, tuple[int, int]] = {}
    for _, row in df.iterrows():
        region = str(row.get("Region") or row.get("Proteina"))
        if region not in REGION_ORDER:
            continue
        start = int(row["Start_nt"])
        end = int(row["End_nt"])
        if region == "5'UTR":
            start = 0
        boundaries[region] = (start, end)

    missing = [region for region in REGION_ORDER if region not in boundaries]
    if missing:
        raise ValueError(f"Missing regions in coordinate map {coordinate_map_path}: {missing}")
    return {region: boundaries[region] for region in REGION_ORDER}


def _site_profile_to_genome_coordinates(
    sites: pd.DataFrame,
    gene_boundaries: dict[str, tuple[int, int]],
    metric_col: str,
) -> pd.DataFrame:
    rows = []
    for region in REGION_ORDER:
        region_sites = sites[sites["region"].map(_clean_region) == _clean_region(region)].copy()
        if region_sites.empty or region not in gene_boundaries:
            continue

        region_sites = region_sites.sort_values("alignment_pos_1based")
        start, end = gene_boundaries[region]
        n_genomic_positions = max(1, end - start + 1)
        n_alignment_sites = len(region_sites)

        # MAFFT columns may include insertions/gaps, so alignment columns are projected
        # onto the corresponding nucleotide interval and duplicate coordinates are averaged.
        projected = np.rint(np.linspace(start, end, n_alignment_sites)).astype(int)
        tmp = pd.DataFrame({
            "genome_pos": projected,
            metric_col: pd.to_numeric(region_sites[metric_col], errors="coerce"),
        }).dropna()
        rows.append(tmp)

    if not rows:
        raise ValueError(f"No site-level values found for metric: {metric_col}")

    out = pd.concat(rows, ignore_index=True)
    out = out.groupby("genome_pos", as_index=False)[metric_col].mean()
    return out.sort_values("genome_pos")


def _plot_stacked_genome_profiles(
    profiles: dict[str, pd.DataFrame],
    metric_col: str,
    ylabel: str,
    gene_boundaries: dict[str, tuple[int, int]],
    output_path: Path,
    ylim: tuple[float, float] | None = None,
    cmap_name: str = "tab20",
    figsize: tuple[float, float] = (12, 3),
    dpi: int = 300,
    region_name: str = "Region",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(profiles.keys())
    fig, axes = plt.subplots(
        len(labels),
        1,
        figsize=(figsize[0], figsize[1] * len(labels)),
        sharex=True,
    )
    if len(labels) == 1:
        axes = [axes]

    colors = cm.get_cmap(cmap_name, len(gene_boundaries))
    region_handles, region_labels = [], []
    xmax = max(end for _, end in gene_boundaries.values())

    for ax, label in zip(axes, labels):
        profile = profiles[label]
        ax.plot(profile["genome_pos"], profile[metric_col], lw=2, color="black")

        for i, (region, (start, end)) in enumerate(gene_boundaries.items()):
            span = ax.axvspan(start, end, color=colors(i), alpha=0.3, label=region)
            if region not in region_labels:
                region_handles.append(span)
                region_labels.append(region)

        ax.set_title(label, fontsize=20, pad=5)
        ax.set_ylabel(ylabel, fontsize=18, labelpad=6)
        ax.set_xlim(0, xmax)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

    axes[-1].set_xlabel("Genomic RNA (Position)", fontsize=18, labelpad=6)
    fig.legend(
        region_handles,
        region_labels,
        loc="center left",
        bbox_to_anchor=(0.915, 0.5),
        ncol=1,
        fontsize=16,
        title=region_name,
        title_fontsize=18,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def plot_region_summary(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "region_divergence_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = _ordered(pd.read_csv(summary_path))

    for metric, ylabel, filename in [
        ("variable_site_fraction", "Variable site fraction", "region_variable_site_fraction.png"),
        ("mean_entropy", "Mean nucleotide entropy", "region_mean_entropy.png"),
        ("mean_consensus_similarity", "Mean consensus similarity", "region_consensus_similarity.png"),
        ("mean_pairwise_identity", "Mean pairwise identity", "region_pairwise_identity.png"),
    ]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(df["region"], df[metric])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("DENV genomic region")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_path}")


def plot_site_entropy(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sites_path = results_dir / "site_divergence_all_regions.csv"
    if not sites_path.exists():
        raise FileNotFoundError(sites_path)
    sites = pd.read_csv(sites_path)
    for region, df in sites.groupby("region"):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["alignment_pos_1based"], df["entropy"], linewidth=1.5)
        ax.set_title(region)
        ax.set_xlabel("Alignment position")
        ax.set_ylabel("Nucleotide entropy")
        fig.tight_layout()
        out_path = output_dir / f"{region}_site_entropy.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[OK] Saved site entropy plots in {output_dir}")


def plot_nucleotide_level_msa_profiles(
    results_dir: Path,
    output_dir: Path,
    coordinates_dir: Path,
    coordinate_map_name: str | None = None,
    prefix: str = "msa",
) -> None:
    output_dir = Path(output_dir)
    sites_path = Path(results_dir) / "site_divergence_all_regions.csv"
    if not sites_path.exists():
        raise FileNotFoundError(sites_path)

    sites = pd.read_csv(sites_path)
    coordinate_map = _find_coordinate_map(Path(coordinates_dir), coordinate_map_name)
    gene_boundaries = load_gene_boundaries(coordinate_map)
    print(f"[INFO] Using coordinate map for nucleotide-level MSA plots: {coordinate_map}")

    entropy_profile = _site_profile_to_genome_coordinates(sites, gene_boundaries, "entropy")
    _plot_stacked_genome_profiles(
        {"MSA nucleotide entropy": entropy_profile},
        "entropy",
        "Entropy (bits)",
        gene_boundaries,
        output_dir / f"{prefix}_nucleotide_entropy_stacked_with_genes.png",
        ylim=(0, DNA_ENTROPY_MAX_BITS),
    )

    if "consensus_similarity" not in sites.columns:
        sites["consensus_similarity"] = 1.0 - (sites["entropy"] / DNA_ENTROPY_MAX_BITS)
    similarity_profile = _site_profile_to_genome_coordinates(sites, gene_boundaries, "consensus_similarity")
    _plot_stacked_genome_profiles(
        {"MSA consensus similarity": similarity_profile},
        "consensus_similarity",
        "Similarity",
        gene_boundaries,
        output_dir / f"{prefix}_nucleotide_similarity_stacked_with_genes.png",
        ylim=(0, 1),
    )
