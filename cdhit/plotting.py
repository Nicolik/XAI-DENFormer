from pathlib import Path

import matplotlib.pyplot as plt

from cdhit.config import identity, show_titles
from dataset import config as dataset_config


STATS_BAR_WIDTH = 0.75
STATS_BAR_SLOT_INCHES = 0.95
STATS_BAR_MARGIN_INCHES = 2.3
STATS_BAR_MIN_FIGWIDTH = 6.5
STATS_BAR_HEIGHT = 6.0


serotype_colors = {
    "DENV1": "#66c2a5",
    "DENV2": "#fc8d62",
    "DENV3": "#8da0cb",
    "DENV4": "#e78ac3",
}

continent_order = [
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Oceania",
    "Antarctica",
]

continent_colors = {
    "Africa": "#1b9e77",
    "Asia": "#d95f02",
    "Europe": "#7570b3",
    "North America": "#e7298a",
    "South America": "#66a61e",
    "Oceania": "#e6ab02",
    "Antarctica": "#a6761d",
}

year_bin_order = [
    "<2000",
    "2000-2009",
    "2010-2019",
    "2020+",
]

year_colors = {
    "<2000": "#8dd3c7",
    "2000-2009": "#ffffb3",
    "2010-2019": "#bebada",
    "2020+": "#fb8072",
}


def plot_all_sequences_vs_representatives(summary_rows: list[dict], out_png: Path) -> None:
    serotypes = ["DENV1", "DENV2", "DENV3", "DENV4"]

    original = {row["serotype"]: row["n_input_sequences"] for row in summary_rows}
    representatives = {row["serotype"]: row["n_representative_sequences"] for row in summary_rows}

    x = list(range(len(serotypes)))
    width = 0.35

    colors = {
        "All sequences": "#bdbdbd",
        "CD-HIT representatives": "#3182bd",
    }

    plt.figure(figsize=(9, 6))

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        [original.get(s, 0) for s in serotypes],
        width=width,
        label="All sequences",
        color=colors["All sequences"],
    )

    bars2 = plt.bar(
        [i + width / 2 for i in x],
        [representatives.get(s, 0) for s in serotypes],
        width=width,
        label="CD-HIT representatives",
        color=colors["CD-HIT representatives"],
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            value = int(bar.get_height())
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xticks(x, serotypes)
    plt.xlabel("Serotype")
    plt.ylabel("Number of sequences")
    plt.title(f"All sequences vs CD-HIT representatives at {identity:.2f} identity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def _stats_bar_figsize(n_bars: int) -> tuple[float, float]:
    width = max(
        STATS_BAR_MIN_FIGWIDTH,
        STATS_BAR_MARGIN_INCHES + n_bars * STATS_BAR_SLOT_INCHES,
    )
    return width, STATS_BAR_HEIGHT


def save_pivot_and_plot(
    df,
    index_col,
    columns_col,
    values_col,
    row_order,
    col_order,
    color_map,
    csv_path: Path,
    plot_path: Path,
    title: str,
    xlabel: str,
    no_legend_plot_path: Path | None = None,
) -> None:
    plot_df = df[~df[columns_col].eq("Unknown")].copy()

    pivot = plot_df.pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc="count",
        fill_value=0,
    )

    existing_rows = [x for x in row_order if x in pivot.index]
    extra_rows = [x for x in pivot.index if x not in existing_rows]
    pivot = pivot.reindex(existing_rows + extra_rows)

    existing_cols = [x for x in col_order if x in pivot.columns]
    pivot = pivot.reindex(columns=existing_cols, fill_value=0)

    pivot.to_csv(csv_path, sep=";")

    colors = [color_map[x] for x in pivot.columns]

    def draw(output_path: Path, show_legend: bool) -> None:
        ax = pivot.plot(
            kind="bar",
            stacked=True,
            figsize=_stats_bar_figsize(len(pivot.index)),
            width=STATS_BAR_WIDTH,
            color=colors,
        )

        ax.set_ylabel("Genomes", fontsize=dataset_config.STATS_AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=dataset_config.STATS_TICK_FONTSIZE)

        if show_titles:
            ax.set_xlabel(xlabel, fontsize=dataset_config.STATS_AXIS_LABEL_FONTSIZE)
            ax.set_title(title, fontsize=dataset_config.STATS_TITLE_FONTSIZE)
        else:
            ax.set_xlabel("")
            ax.set_title("")

        if show_legend:
            legend = ax.legend(title="Serotype" if show_titles else None, fontsize=dataset_config.STATS_LEGEND_FONTSIZE)
            if legend is not None and not show_titles:
                legend.set_title(None)
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    draw(plot_path, show_legend=True)
    if no_legend_plot_path is not None:
        draw(no_legend_plot_path, show_legend=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")
    if no_legend_plot_path is not None:
        print(f"Saved: {no_legend_plot_path}")
