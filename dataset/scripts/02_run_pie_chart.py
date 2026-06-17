import shutil

import matplotlib.pyplot as plt

import paths
from dataset import config
from dataset.utils import ensure_dir, load_fasta_records


def save_pie_chart(df, plot_path):
    counts = df["Serotype"].value_counts().reindex(config.SEROTYPES, fill_value=0)

    def autopct_format(pct):
        total = counts.sum()
        absolute = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({absolute})"

    ensure_dir(plot_path.parent)
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, autotexts = ax.pie(
        counts,
        labels=None,
        colors=config.SEROTYPE_COLORS,
        autopct=autopct_format,
        startangle=90,
        pctdistance=0.68,
    )

    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
        text.set_fontsize(config.STATS_PANEL_PIE_TEXT_FONTSIZE)

    ax.set_title("Serotype distribution of Dengue virus genomes" if config.SHOW_TITLES else "")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {plot_path}")


def main():
    df = load_fasta_records(paths.genomes_dir)
    print(f"Loaded records: {len(df)}")

    counts = df["Serotype"].value_counts().reindex(config.SEROTYPES, fill_value=0)
    print("\nSerotype counts:")
    print(counts)

    counts_path = paths.stats_dir / "dengue_serotype_counts.csv"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts.rename("count").to_csv(counts_path, sep=";", header=True)
    print(f"Saved serotype counts to: {counts_path}")

    plot_path = paths.stats_dir / "dengue_serotype_pie_chart.png"
    no_legend_plot_path = paths.stats_dir / "dengue_serotype_pie_chart_no_legend.png"
    save_pie_chart(df, plot_path)
    shutil.copy(plot_path, no_legend_plot_path)
    print(f"Saved plot to: {no_legend_plot_path}")


if __name__ == "__main__":
    main()
