"""Region-level recap plots for serotype-aware CD-HIT MSA variability.

This module summarizes the per-region outputs produced by
``msa.serotype_variability`` into a small set of paper-oriented
figures. It is intended for outputs such as::

    Data/msa/cdhit/serotype_variability_panel/strategy_consensus/<REGION>/

and writes a compact recap under::

    Data/msa/cdhit/serotype_variability_panel/strategy_consensus/region_recap/

The code uses only pandas/numpy/matplotlib and derives region order from
``msa.config.REGION_ORDER``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from msa.config import REGION_ORDER, SEROTYPES


# Keep DENV colors consistent with dataset-level stats/pie-chart figures
# (dataset.config.SEROTYPE_COLORS).
DENV_COLORS = {
    "DENV1": "#66c2a5",
    "DENV2": "#fc8d62",
    "DENV3": "#8da0cb",
    "DENV4": "#e78ac3",
}

# Paper-oriented sequential palettes.  The low end is softly tinted rather than
# pure white, so the scale reads as blue tones for entropy and red tones for
# divergence without the previous blue-yellow or red-yellow impression.
ENTROPY_CMAP = LinearSegmentedColormap.from_list(
    "paper_entropy_blues", ["#deebf7", "#9ecae1", "#3182bd", "#08519c"]
)
DIVERGENCE_CMAP = LinearSegmentedColormap.from_list(
    "paper_divergence_reds", ["#fee0d2", "#fc9272", "#de2d26", "#99000d"]
)
BAR_COLORS = {
    "between_serotype": "#1f4e79",
    "global_entropy": "#8f8f8f",
}

REGION_ALIASES = {r.replace("'", ""): r for r in REGION_ORDER}
REGION_ALIASES.update({r: r for r in REGION_ORDER})


def clean_region_name(value: object) -> str:
    text = str(value).strip()
    stem = Path(text).stem
    stem = re.sub(r"\.aln$", "", stem)
    stem = stem.replace("_", "'") if stem in {"5_UTR", "3_UTR"} else stem
    if stem in REGION_ALIASES:
        return REGION_ALIASES[stem]
    compact = stem.replace("'", "")
    if compact in REGION_ALIASES:
        return REGION_ALIASES[compact]
    return stem


def region_sort_key(region: object) -> tuple[int, str]:
    clean = clean_region_name(region)
    order = {r: i for i, r in enumerate(REGION_ORDER)}
    return order.get(clean, 999), clean


def ordered_regions(regions: Sequence[object]) -> list[str]:
    unique = {clean_region_name(r) for r in regions}
    return sorted(unique, key=region_sort_key)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA")


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def find_region_dirs(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    out: list[Path] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("region_recap") or child.name.startswith("_recap"):
            continue
        if (child / "serotype_summary.tsv").exists() or (child / "per_position_serotype_metrics.tsv").exists():
            out.append(child)
    return out


def read_region_serotype_summary(region_dir: Path) -> pd.DataFrame:
    region = clean_region_name(region_dir.name)
    summary_path = region_dir / "serotype_summary.tsv"
    if summary_path.exists():
        df = pd.read_csv(summary_path, sep="\t")
        df.insert(0, "region", region)
        return df

    metrics_path = region_dir / "per_position_serotype_metrics.tsv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing serotype summary and per-position metrics in {region_dir}")
    metrics = pd.read_csv(metrics_path, sep="\t")
    metrics = metrics[metrics["serotype"].isin(SEROTYPES)].copy()
    out = (
        metrics.groupby("serotype", as_index=False)
        .agg(
            n_sequences=("n_sequences", "max"),
            mean_entropy_norm=("entropy_norm", "mean"),
            median_entropy_norm=("entropy_norm", "median"),
            mean_conservation_norm=("conservation_norm", "mean"),
            mean_intra_similarity=("intra_similarity", "mean"),
            mean_gap_fraction=("gap_fraction", "mean"),
        )
    )
    out.insert(0, "region", region)
    return out


def read_region_pair_summary(region_dir: Path) -> pd.DataFrame:
    region = clean_region_name(region_dir.name)
    pair_path = region_dir / "pair_summary.tsv"
    if pair_path.exists():
        df = pd.read_csv(pair_path, sep="\t")
        df.insert(0, "region", region)
        return df

    pairs_path = region_dir / "per_position_serotype_pairs.tsv"
    if not pairs_path.exists():
        return pd.DataFrame()
    pairs = pd.read_csv(pairs_path, sep="\t")
    if pairs.empty:
        return pd.DataFrame()
    out = (
        pairs.groupby(["pair", "serotype_a", "serotype_b"], as_index=False)
        .agg(
            mean_distribution_overlap=("distribution_overlap", "mean"),
            median_distribution_overlap=("distribution_overlap", "median"),
            mean_jsd=("jensen_shannon_divergence", "mean"),
            median_jsd=("jensen_shannon_divergence", "median"),
            mean_cosine=("cosine_similarity", "mean"),
            mean_coverage_weighted_overlap=("coverage_weighted_overlap", "mean"),
            mean_gap_fraction_absdiff=("gap_fraction_absdiff", "mean"),
        )
    )
    out.insert(0, "region", region)
    return out


def read_region_aggregate_summary(region_dir: Path, variable_entropy_threshold: float) -> dict[str, object]:
    region = clean_region_name(region_dir.name)
    agg_path = region_dir / "per_position_aggregate_metrics.tsv"
    row: dict[str, object] = {"region": region}
    if not agg_path.exists():
        return row
    agg = pd.read_csv(agg_path, sep="\t")
    numeric_cols = [
        "global_entropy_norm",
        "serotype_entropy_mean",
        "serotype_entropy_range",
        "intra_similarity_mean",
        "pairwise_jsd_mean",
        "pairwise_jsd_max",
        "pairwise_overlap_mean",
        "serotype_specificity_score",
        "global_gap_fraction",
    ]
    for col in numeric_cols:
        if col in agg.columns:
            row[f"mean_{col}"] = pd.to_numeric(agg[col], errors="coerce").mean()
            row[f"median_{col}"] = pd.to_numeric(agg[col], errors="coerce").median()
    if "global_entropy_norm" in agg.columns:
        vals = pd.to_numeric(agg["global_entropy_norm"], errors="coerce")
        row["global_variable_site_fraction"] = float((vals > variable_entropy_threshold).mean())
    row["n_alignment_positions"] = int(len(agg))
    return row


def load_recap_tables(input_dir: Path, variable_entropy_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_dirs = find_region_dirs(input_dir)
    if not region_dirs:
        raise FileNotFoundError(
            f"No per-region serotype_variability outputs found in {input_dir}. "
            "Expected subfolders with serotype_summary.tsv or per_position_serotype_metrics.tsv."
        )

    serotype_rows = [read_region_serotype_summary(path) for path in region_dirs]
    pair_rows = [read_region_pair_summary(path) for path in region_dirs]
    aggregate_rows = [read_region_aggregate_summary(path, variable_entropy_threshold) for path in region_dirs]

    serotype = pd.concat(serotype_rows, ignore_index=True)
    serotype["region"] = serotype["region"].map(clean_region_name)
    serotype["_region_order"] = serotype["region"].map(lambda x: region_sort_key(x)[0])
    serotype["_serotype_order"] = serotype["serotype"].map(lambda x: SEROTYPES.index(x) if x in SEROTYPES else 999)
    serotype = serotype.sort_values(["_region_order", "_serotype_order"]).drop(columns=["_region_order", "_serotype_order"])

    pair = pd.concat([df for df in pair_rows if not df.empty], ignore_index=True) if any(not df.empty for df in pair_rows) else pd.DataFrame()
    if not pair.empty:
        pair["region"] = pair["region"].map(clean_region_name)

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["region"] = aggregate["region"].map(clean_region_name)
    aggregate["_region_order"] = aggregate["region"].map(lambda x: region_sort_key(x)[0])
    aggregate = aggregate.sort_values("_region_order").drop(columns="_region_order")
    return serotype, pair, aggregate


def plot_serotype_entropy_heatmap(serotype: pd.DataFrame, out_base: Path, dpi: int) -> None:
    regions = ordered_regions(serotype["region"])
    serotypes = [s for s in SEROTYPES if s in set(serotype["serotype"])]
    mat = (
        serotype.pivot(index="serotype", columns="region", values="mean_entropy_norm")
        .reindex(index=serotypes, columns=regions)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=max(0.25, np.nanmax(mat) * 1.05), cmap=ENTROPY_CMAP)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(serotypes)))
    ax.set_yticklabels(serotypes, fontsize=12)
    ax.set_title("Within-serotype entropy", fontsize=14)
    ax.set_xlabel("DENV genomic region", fontsize=12)
    ax.set_ylabel("Serotype", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Mean normalized entropy", fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_base, dpi)


def plot_serotype_entropy_grouped_bar(serotype: pd.DataFrame, out_base: Path, dpi: int) -> None:
    regions = ordered_regions(serotype["region"])
    serotypes = [s for s in SEROTYPES if s in set(serotype["serotype"])]
    pivot = serotype.pivot(index="region", columns="serotype", values="mean_entropy_norm").reindex(regions)

    x = np.arange(len(regions))
    width = min(0.18, 0.8 / max(1, len(serotypes)))
    offsets = (np.arange(len(serotypes)) - (len(serotypes) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    for idx, serotype_name in enumerate(serotypes):
        ax.bar(
            x + offsets[idx],
            pivot[serotype_name].to_numpy(float),
            width,
            label=serotype_name,
            color=DENV_COLORS.get(serotype_name),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("DENV genomic region", fontsize=12)
    ax.set_ylabel("Mean normalized entropy", fontsize=12)
    ax.set_title("Within-serotype entropy", fontsize=14)
    ax.legend(frameon=False, ncol=len(serotypes), fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, out_base, dpi)



def plot_serotype_variability_grouped_bars(serotype: pd.DataFrame, pair: pd.DataFrame, out_base: Path, dpi: int) -> None:
    """Save a two-panel grouped-bar summary with one bar per serotype.

    Panel A reports mean within-serotype entropy for each genomic region.
    Panel B reports mean JSD of each serotype against the remaining serotypes
    for each genomic region.  This keeps the region-level summary as a barplot,
    but avoids mixing global aggregate bars with heatmap panels.
    """
    if pair.empty or "mean_jsd" not in pair.columns:
        return

    regions = ordered_regions(serotype["region"])
    serotypes = [s for s in SEROTYPES if s in set(serotype["serotype"])]
    entropy = (
        serotype.pivot(index="region", columns="serotype", values="mean_entropy_norm")
        .reindex(index=regions, columns=serotypes)
    )
    vs_others = pairwise_to_serotype_vs_others(pair, regions).reindex(index=serotypes, columns=regions)

    x = np.arange(len(regions))
    width = min(0.18, 0.78 / max(1, len(serotypes)))
    offsets = (np.arange(len(serotypes)) - (len(serotypes) - 1) / 2) * width

    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.2), sharex=True)
    ax0, ax1 = axes

    for idx, serotype_name in enumerate(serotypes):
        color = DENV_COLORS.get(serotype_name)
        ax0.bar(
            x + offsets[idx],
            entropy[serotype_name].to_numpy(dtype=float),
            width,
            label=serotype_name,
            color=color,
        )
        ax1.bar(
            x + offsets[idx],
            vs_others.loc[serotype_name].to_numpy(dtype=float),
            width,
            label=serotype_name,
            color=color,
        )

    ax0.set_title("A. Within-serotype entropy", fontsize=14, loc="left")
    ax0.set_ylabel("Mean normalized entropy", fontsize=12)
    ax0.grid(axis="y", alpha=0.2)

    ax1.set_title("B. Serotype-vs-others divergence", fontsize=14, loc="left")
    ax1.set_ylabel("Mean JSD", fontsize=12)
    ax1.set_xlabel("DENV genomic region", fontsize=12)
    ax1.grid(axis="y", alpha=0.2)

    for ax in axes:
        ax.set_xlim(-0.5, len(regions) - 0.5)
        ax.tick_params(axis="y", labelsize=11)

    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha="right", fontsize=11)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(serotypes),
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    save_figure(fig, out_base, dpi)


def pairwise_to_serotype_vs_others(pair: pd.DataFrame, regions: Sequence[str]) -> pd.DataFrame:
    """Average pairwise JSD for each serotype against all other serotypes.

    The per-region analysis stores the six DENV pairwise comparisons.  For the
    recap panel this view is easier to read than a 6-row pair matrix: each row
    answers the question "how divergent is this serotype from the remaining
    serotypes in this genomic region?".
    """
    if pair.empty or "mean_jsd" not in pair.columns:
        return pd.DataFrame(index=[s for s in SEROTYPES], columns=list(regions), dtype=float)

    rows: list[dict[str, object]] = []
    for _, row in pair.iterrows():
        region = clean_region_name(row.get("region"))
        value = pd.to_numeric(row.get("mean_jsd"), errors="coerce")
        serotype_a = row.get("serotype_a")
        serotype_b = row.get("serotype_b")
        if pd.isna(serotype_a) or pd.isna(serotype_b):
            pair_name = str(row.get("pair", ""))
            parts = re.split(r"v|-|_vs_", pair_name)
            parts = [part.strip() for part in parts if part.strip()]
            if len(parts) >= 2:
                serotype_a, serotype_b = parts[0], parts[1]
        for serotype_name in (serotype_a, serotype_b):
            if serotype_name in SEROTYPES:
                rows.append({"serotype": serotype_name, "region": region, "mean_jsd": value})

    if not rows:
        return pd.DataFrame(index=[s for s in SEROTYPES], columns=list(regions), dtype=float)

    long_df = pd.DataFrame(rows)
    return (
        long_df.groupby(["serotype", "region"], as_index=False)["mean_jsd"].mean()
        .pivot(index="serotype", columns="region", values="mean_jsd")
        .reindex(index=[s for s in SEROTYPES if s in set(long_df["serotype"])], columns=list(regions))
    )


def plot_pairwise_jsd_heatmap(pair: pd.DataFrame, out_base: Path, dpi: int) -> None:
    if pair.empty or "mean_jsd" not in pair.columns:
        return
    regions = ordered_regions(pair["region"])
    pair_order = sorted(pair["pair"].dropna().unique())
    mat = pair.pivot(index="pair", columns="region", values="mean_jsd").reindex(index=pair_order, columns=regions).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    vmax = max(0.25, np.nanmax(mat) * 1.05) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap=DIVERGENCE_CMAP)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(pair_order)))
    ax.set_yticklabels([str(p).replace("v", "-") for p in pair_order], fontsize=11)
    ax.set_title("Inter-serotype divergence", fontsize=14)
    ax.set_xlabel("DENV genomic region", fontsize=12)
    ax.set_ylabel("Serotype pair", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Mean Jensen-Shannon divergence", fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_base, dpi)


def plot_serotype_vs_others_jsd_heatmap(pair: pd.DataFrame, out_base: Path, dpi: int) -> None:
    if pair.empty or "mean_jsd" not in pair.columns:
        return
    regions = ordered_regions(pair["region"])
    mat_df = pairwise_to_serotype_vs_others(pair, regions)
    mat = mat_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    vmax = max(0.25, np.nanmax(mat) * 1.05) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap=DIVERGENCE_CMAP)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(mat_df.index)))
    ax.set_yticklabels(list(mat_df.index), fontsize=11)
    ax.set_title("Serotype-vs-others divergence", fontsize=14)
    ax.set_xlabel("DENV genomic region", fontsize=12)
    ax.set_ylabel("Serotype", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Mean Jensen-Shannon divergence", fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_base, dpi)


def plot_between_serotype_bar(aggregate: pd.DataFrame, out_base: Path, dpi: int) -> None:
    regions = ordered_regions(aggregate["region"])
    df = aggregate.set_index("region").reindex(regions).reset_index()
    metric = "mean_serotype_specificity_score" if "mean_serotype_specificity_score" in df.columns else "mean_pairwise_jsd_mean"

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    x = np.arange(len(df))
    ax.bar(x, pd.to_numeric(df[metric], errors="coerce"))
    ax.set_xticks(x)
    ax.set_xticklabels(df["region"], rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("DENV genomic region", fontsize=12)
    ax.set_ylabel("Mean between-serotype variability", fontsize=12)
    ax.set_title("Regions with stronger serotype-specific sequence variability", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, out_base, dpi)


def plot_paper_recap(serotype: pd.DataFrame, pair: pd.DataFrame, aggregate: pd.DataFrame, out_base: Path, dpi: int) -> None:
    """Save the compact paper recap without the aggregate barplot panel.

    The detailed region-level grouped bar summary is written separately by
    plot_serotype_variability_grouped_bars().  Keeping this recap to heatmaps
    avoids the previous visual misalignment between a categorical barplot and
    heatmap panels.
    """
    regions = ordered_regions(serotype["region"])
    serotypes = [s for s in SEROTYPES if s in set(serotype["serotype"])]
    entropy_mat = (
        serotype.pivot(index="serotype", columns="region", values="mean_entropy_norm")
        .reindex(index=serotypes, columns=regions)
        .to_numpy(dtype=float)
    )

    has_pairs = not pair.empty and "mean_jsd" in pair.columns
    nrows = 2 if has_pairs else 1
    fig_height = 6.2 if has_pairs else 3.8
    fig, axes = plt.subplots(nrows, 1, figsize=(12.8, fig_height), gridspec_kw={"height_ratios": [1.0, 1.0] if has_pairs else [1.0]})
    if has_pairs:
        ax0, ax1 = axes
    else:
        ax0 = axes
        ax1 = None

    vmax_entropy = max(0.25, np.nanmax(entropy_mat) * 1.05) if np.isfinite(entropy_mat).any() else 1.0
    im0 = ax0.imshow(entropy_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax_entropy, cmap=ENTROPY_CMAP)
    ax0.set_xticks(np.arange(len(regions)))
    ax0.set_xticklabels([] if has_pairs else regions, rotation=45, ha="right", fontsize=11)
    ax0.set_yticks(np.arange(len(serotypes)))
    ax0.set_yticklabels(serotypes, fontsize=12)
    ax0.set_ylabel("Serotype", fontsize=12)
    ax0.set_title("A. Within-serotype entropy", fontsize=14, loc="left")
    cbar0 = fig.colorbar(im0, ax=ax0, pad=0.01)
    cbar0.set_label("Mean normalized entropy", fontsize=11)

    if has_pairs and ax1 is not None:
        vs_others = pairwise_to_serotype_vs_others(pair, regions)
        pair_mat = vs_others.to_numpy(dtype=float)
        vmax_pair = max(0.25, np.nanmax(pair_mat) * 1.05) if np.isfinite(pair_mat).any() else 1.0
        im1 = ax1.imshow(pair_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax_pair, cmap=DIVERGENCE_CMAP)
        ax1.set_xticks(np.arange(len(regions)))
        ax1.set_xticklabels(regions, rotation=45, ha="right", fontsize=11)
        ax1.set_yticks(np.arange(len(vs_others.index)))
        ax1.set_yticklabels(list(vs_others.index), fontsize=11)
        ax1.set_xlabel("DENV genomic region", fontsize=12)
        ax1.set_ylabel("Serotype", fontsize=12)
        ax1.set_title("B. Serotype-vs-others divergence", fontsize=14, loc="left")
        cbar1 = fig.colorbar(im1, ax=ax1, pad=0.01)
        cbar1.set_label("Mean JSD", fontsize=11)
    else:
        ax0.set_xlabel("DENV genomic region", fontsize=12)

    fig.tight_layout()
    save_figure(fig, out_base, dpi)

def write_readme(out_dir: Path) -> None:
    text = """Region-level serotype variability recap
=========================================

This folder summarizes the per-region outputs from msa.serotype_variability.
It is intentionally less detailed than the per-region folders and is meant to provide
one or two paper-ready figures.

Core figure:
- paper_region_serotype_entropy_recap.[png/pdf]: compact heatmap panel showing mean entropy by serotype/region and serotype-vs-others distribution divergence.
- paper_region_serotype_variability_grouped_bars.[png/pdf]: two-panel grouped barplot with four serotype bars per region, for intra-serotype entropy and serotype-vs-others divergence.

Additional figures:
- region_serotype_entropy_heatmap.[png/pdf]
- region_serotype_entropy_grouped_bar.[png/pdf]
- region_pairwise_jsd_heatmap.[png/pdf]
- region_serotype_vs_others_jsd_heatmap.[png/pdf]
- region_between_serotype_variability_bar.[png/pdf]

Core tables:
- region_serotype_entropy_summary.tsv
- region_pairwise_divergence_summary.tsv
- region_aggregate_variability_summary.tsv
"""
    (out_dir / "README_region_recap.txt").write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create simple region-level recap plots from serotype_variability outputs.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing one subfolder per genomic region.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for recap plots and tables.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--variable-entropy-threshold", type=float, default=0.0, help="Threshold on normalized global entropy for variable-site fraction.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    serotype, pair, aggregate = load_recap_tables(args.input_dir, args.variable_entropy_threshold)
    save_table(serotype, args.out_dir / "region_serotype_entropy_summary.tsv")
    if not pair.empty:
        save_table(pair, args.out_dir / "region_pairwise_divergence_summary.tsv")
    save_table(aggregate, args.out_dir / "region_aggregate_variability_summary.tsv")

    plot_serotype_entropy_heatmap(serotype, args.out_dir / "region_serotype_entropy_heatmap", args.dpi)
    plot_serotype_entropy_grouped_bar(serotype, args.out_dir / "region_serotype_entropy_grouped_bar", args.dpi)
    plot_between_serotype_bar(aggregate, args.out_dir / "region_between_serotype_variability_bar", args.dpi)
    if not pair.empty:
        plot_serotype_variability_grouped_bars(serotype, pair, args.out_dir / "paper_region_serotype_variability_grouped_bars", args.dpi)
        plot_pairwise_jsd_heatmap(pair, args.out_dir / "region_pairwise_jsd_heatmap", args.dpi)
        plot_serotype_vs_others_jsd_heatmap(pair, args.out_dir / "region_serotype_vs_others_jsd_heatmap", args.dpi)
    plot_paper_recap(serotype, pair, aggregate, args.out_dir / "paper_region_serotype_entropy_recap", args.dpi)
    write_readme(args.out_dir)

    print(f"[OK] Region-level serotype variability recap written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
