import shutil

import paths
from cdhit.config import desired_serotypes, identity_label, k_folds
from cdhit.metadata import (
    annotate_fold_rows,
    build_metadata_from_fasta,
    fold_sort_key,
    load_fold_membership,
    save_fold_summary,
    save_missing_ids,
    warn_unknowns,
)
from cdhit.plotting import (
    continent_colors,
    continent_order,
    save_pivot_and_plot,
    serotype_colors,
    year_bin_order,
    year_colors,
)
from paths import cdhit_dir, genomes_dir, as_path


def main() -> None:
    fasta_root = genomes_dir
    cdhit_root = cdhit_dir
    kfold_dir = cdhit_root / f"kfold_{k_folds}_identity_{identity_label}"

    fold_file = kfold_dir / "sequence_fold_membership.tsv"
    out_dir = kfold_dir / "fold_metadata_annotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"FASTA input root: {fasta_root}")
    print(f"Fold membership:  {fold_file}")
    print(f"Output dir:       {out_dir}")

    metadata = build_metadata_from_fasta(fasta_root)
    fold_rows = load_fold_membership(fold_file)

    df, missing_ids = annotate_fold_rows(fold_rows, metadata)

    warn_unknowns(df)

    annotated_path = out_dir / "fold_sequence_metadata_annotation.tsv"
    df.to_csv(annotated_path, sep="\t", index=False)
    print(f"Saved: {annotated_path}")

    save_fold_summary(
        df,
        out_dir / "fold_summary_serotype_continent_year.csv",
    )

    save_missing_ids(
        missing_ids,
        out_dir / "missing_metadata_sequence_ids.txt",
    )

    folds_order = sorted(df["fold"].unique(), key=fold_sort_key)

    plot_fold_path = out_dir / "01_fold_distribution_by_serotype.png"
    csv_fold_path = out_dir / "01_fold_distribution_by_serotype.csv"

    save_pivot_and_plot(
        df=df,
        index_col="fold",
        columns_col="serotype",
        values_col="sequence_id",
        row_order=folds_order,
        col_order=desired_serotypes,
        color_map=serotype_colors,
        csv_path=csv_fold_path,
        plot_path=plot_fold_path,
        title="Serotype distribution across cluster-aware folds",
        xlabel="Fold",
        no_legend_plot_path=out_dir / "01_fold_distribution_by_serotype_no_legend.png",
    )

    plot_stat_path = paths.stats_dir / f"dengue_cdhit_distribution_k={k_folds}_identity={identity_label}.png"
    plot_stat_no_legend_path = paths.stats_dir / f"dengue_cdhit_distribution_k={k_folds}_identity={identity_label}_no_legend.png"
    csv_stat_path = paths.stats_dir / f"dengue_cdhit_distribution_k={k_folds}_identity={identity_label}.csv"
    paths.stats_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying {plot_fold_path} to {plot_stat_path}")
    shutil.copy(plot_fold_path, plot_stat_path)
    print(f"Copying {out_dir / '01_fold_distribution_by_serotype_no_legend.png'} to {plot_stat_no_legend_path}")
    shutil.copy(out_dir / "01_fold_distribution_by_serotype_no_legend.png", plot_stat_no_legend_path)
    print(f"Copying {csv_fold_path} to {csv_stat_path}")
    shutil.copy(csv_fold_path, csv_stat_path)

    save_pivot_and_plot(
        df=df,
        index_col="fold",
        columns_col="continent",
        values_col="sequence_id",
        row_order=folds_order,
        col_order=continent_order,
        color_map=continent_colors,
        csv_path=out_dir / "02_fold_distribution_by_continent.csv",
        plot_path=out_dir / "02_fold_distribution_by_continent.png",
        title="Continent distribution across cluster-aware folds",
        xlabel="Fold",
    )

    save_pivot_and_plot(
        df=df,
        index_col="fold",
        columns_col="year_bin",
        values_col="sequence_id",
        row_order=folds_order,
        col_order=year_bin_order,
        color_map=year_colors,
        csv_path=out_dir / "03_fold_distribution_by_year_bin.csv",
        plot_path=out_dir / "03_fold_distribution_by_year_bin.png",
        title="Temporal distribution across cluster-aware folds",
        xlabel="Fold",
    )

    print("\nDone.")
    print(f"Annotated sequences: {len(df)}")
    print(f"Missing metadata sequence IDs: {len(set(missing_ids))}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
