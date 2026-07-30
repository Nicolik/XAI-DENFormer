from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

import paths
from cdhit.config import identity_label, k_folds
from dataset import config


BAR_PANEL_SPECS = [
    (
        "B",
        "CD-HIT cluster-aware folds",
        lambda stats_dir: stats_dir / f"dengue_cdhit_distribution_k={k_folds}_identity={identity_label}.csv",
        0,
    ),
    (
        "C",
        "Geographical distribution by continent",
        lambda stats_dir: stats_dir / "dengue_geographical_distribution_by_continent.csv",
        35,
    ),
    (
        "D",
        "Temporal distribution",
        lambda stats_dir: stats_dir / f"dengue_temporal_distribution_c{config.CUTOFF_YEAR}_b{config.BIN_LENGTH}.csv",
        35,
    ),
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required panel input: {path}\n"
            "Run the stats scripts first to generate the CSV files."
        )


def make_serotype_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=11,
            label=serotype,
        )
        for serotype, color in zip(config.SEROTYPES, config.SEROTYPE_COLORS)
    ]


def save_standalone_legend(output_path: Path) -> None:
    handles = make_serotype_handles()
    fig = plt.figure(figsize=(5.6, 0.75))
    fig.legend(
        handles=handles,
        labels=config.SEROTYPES,
        loc="center",
        ncol=len(config.SEROTYPES),
        frameon=False,
        fontsize=config.STATS_PANEL_LEGEND_FONTSIZE,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved standalone legend to: {output_path}")


def load_pivot(csv_path: Path) -> pd.DataFrame:
    require_file(csv_path)
    pivot = pd.read_csv(csv_path, sep=";", index_col=0)
    pivot = pivot.reindex(columns=config.SEROTYPES, fill_value=0)
    return pivot


def load_serotype_counts(stats_dir: Path) -> pd.Series:
    counts_path = stats_dir / "dengue_serotype_counts.csv"
    require_file(counts_path)
    counts = pd.read_csv(counts_path, sep=";", index_col=0).iloc[:, 0]
    return counts.reindex(config.SEROTYPES, fill_value=0).astype(int)


def validate_panel_totals(canonical_counts: pd.Series, bar_pivots) -> None:
    failures = []
    for panel_label, title, pivot, _ in bar_pivots:
        totals = pivot.sum(axis=0).reindex(config.SEROTYPES, fill_value=0).astype(int)
        if not totals.equals(canonical_counts):
            comparison = pd.DataFrame({
                "canonical": canonical_counts,
                f"panel_{panel_label}": totals,
            })
            comparison["difference"] = comparison.iloc[:, 1] - comparison.iloc[:, 0]
            failures.append(f"Panel {panel_label} ({title}):\n{comparison}")

    if failures:
        raise ValueError(
            "Figure 2 serotype totals are inconsistent. Regenerate the affected inputs.\n\n"
            + "\n\n".join(failures)
        )
    print("Figure 2 A-D serotype consistency check: PASSED")


def autopct_factory(counts: pd.Series):
    total = counts.sum()

    def autopct_format(pct):
        absolute = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({absolute})"

    return autopct_format


def draw_pie(ax, counts: pd.Series) -> None:
    _, _, autotexts = ax.pie(
        counts,
        labels=None,
        colors=config.SEROTYPE_COLORS,
        autopct=autopct_factory(counts),
        startangle=90,
        pctdistance=0.68,
        radius=0.88,
        center=(0.45, 0.0),
        textprops={
            "fontsize": config.STATS_PANEL_PIE_TEXT_FONTSIZE,
            "fontweight": "bold",
            "color": "white",
        },
    )
    for text in autotexts:
        text.set_color("white")

    if len(autotexts) > 0:
        smallest_idx = int(counts.to_numpy().argmin())
        x, y = autotexts[smallest_idx].get_position()
        autotexts[smallest_idx].set_position((x, y + config.STATS_PANEL_PIE_SMALL_SLICE_Y_SHIFT))

    ax.legend(
        handles=make_serotype_handles(),
        labels=config.SEROTYPES,
        loc="center left",
        bbox_to_anchor=(0.02, 0.50),
        frameon=False,
        fontsize=config.STATS_PANEL_LEGEND_FONTSIZE,
        borderaxespad=0.0,
        handletextpad=0.45,
        labelspacing=0.90,
    )
    ax.set_xlim(-1.85, 1.45)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(
        "A. Serotype distribution",
        fontsize=config.STATS_PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        pad=10,
    )


def draw_stacked_bar(ax, pivot: pd.DataFrame, panel_label: str, title: str, rotation: int, shared_ylim: float) -> None:
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=config.SEROTYPE_COLORS,
        width=config.STATS_BAR_WIDTH,
        legend=False,
    )
    ax.set_title(
        f"{panel_label}. {title}",
        fontsize=config.STATS_PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Genomes", fontsize=config.STATS_PANEL_AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0, shared_ylim)
    ax.tick_params(axis="both", labelsize=config.STATS_PANEL_TICK_FONTSIZE)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def main() -> None:
    stats_dir = paths.stats_dir
    output_path = stats_dir / "dengue_stats_2x2_panel.png"
    legend_path = stats_dir / "dengue_stats_serotype_legend.png"

    counts = load_serotype_counts(stats_dir)
    bar_pivots = []
    for panel_label, title, csv_path_fn, rotation in BAR_PANEL_SPECS:
        pivot = load_pivot(csv_path_fn(stats_dir))
        bar_pivots.append((panel_label, title, pivot, rotation))

    validate_panel_totals(counts, bar_pivots)

    max_bar_total = max(float(pivot.sum(axis=1).max()) for _, _, pivot, _ in bar_pivots)
    shared_ylim = max_bar_total * 1.10

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.0))

    draw_pie(axes[0, 0], counts)
    for ax, (panel_label, title, pivot, rotation) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], bar_pivots):
        draw_stacked_bar(ax, pivot, panel_label, title, rotation, shared_ylim)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.07, wspace=0.25, hspace=0.32)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2x2 stats panel to: {output_path}")

    save_standalone_legend(legend_path)


if __name__ == "__main__":
    main()
