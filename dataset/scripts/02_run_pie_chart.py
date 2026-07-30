import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from dataset import config
from dataset.utils import ensure_dir, load_fasta_records


def load_encoded_counts():
    labels_path = paths.embeddings_dir / "label_matrix.txt"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Encoded label matrix not found: {labels_path}\n"
            "Run python -m dataset.scripts.01_run_ohe first."
        )

    labels = np.atleast_1d(np.loadtxt(labels_path, dtype=np.int64))
    invalid = sorted(set(labels.tolist()) - set(range(len(config.SEROTYPES))))
    if invalid:
        raise ValueError(f"Unexpected class labels in {labels_path}: {invalid}")

    return (
        pd.Series(labels)
        .value_counts()
        .reindex(range(len(config.SEROTYPES)), fill_value=0)
        .rename(index=dict(enumerate(config.SEROTYPES)))
        .astype(int)
    )


def validate_counts(fasta_counts):
    encoded_counts = load_encoded_counts()
    fasta_counts = fasta_counts.astype(int)
    if not fasta_counts.equals(encoded_counts):
        comparison = pd.DataFrame({
            "fasta_assignment": fasta_counts,
            "encoded_label_matrix": encoded_counts,
        })
        comparison["difference"] = comparison["fasta_assignment"] - comparison["encoded_label_matrix"]
        raise ValueError(
            "Serotype-count mismatch between source FASTA assignments and label_matrix.txt:\n"
            + comparison.to_string()
        )
    print("Serotype consistency check against label_matrix.txt: PASSED")


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

    conflicts = int(df["SerotypeHeaderConflict"].sum())
    print(f"Header/file serotype conflicts retained as diagnostics: {conflicts}")

    counts = df["Serotype"].value_counts().reindex(config.SEROTYPES, fill_value=0)
    validate_counts(counts)

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
