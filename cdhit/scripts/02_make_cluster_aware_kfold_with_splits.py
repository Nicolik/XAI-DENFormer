from cdhit.config import generate_splits, identity_label, k_folds, split_sep
from cdhit.io import read_clusters, save_sequence_fold, save_split_summary, save_training_split
from cdhit.parsing import build_dataset_index_map
from cdhit.splits import assign_clusters, build_cluster_aware_training_splits
from paths import cdhit_dir, genomes_dir, splits_dir, as_path


def main() -> None:
    root = cdhit_dir
    out = root / f"kfold_{k_folds}_identity_{identity_label}"
    out.mkdir(exist_ok=True)

    clusters = read_clusters(root)
    folds = assign_clusters(clusters, k_folds)

    save_sequence_fold(
        folds,
        out / "sequence_fold_membership.tsv",
    )

    if generate_splits:
        dataset_root = genomes_dir
        seq_to_index, all_fasta_ids = build_dataset_index_map(dataset_root)

        split_rows = build_cluster_aware_training_splits(
            folds=folds,
            seq_to_index=seq_to_index,
            all_fasta_ids=all_fasta_ids,
        )

        split_out_dir = splits_dir
        split_out = split_out_dir / (
            f"dengue_cdhit_cluster_aware_kfold_{k_folds}_identity_{identity_label}_splits.csv"
        )

        save_training_split(split_rows, split_out, split_sep=split_sep)
        save_split_summary(split_rows, split_out)

    print("Done k-fold (cluster-aware)")


if __name__ == "__main__":
    main()
