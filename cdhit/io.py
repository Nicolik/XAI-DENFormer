import csv
from pathlib import Path


def save_cluster_membership_tsv(clusters: list[dict], out_tsv: Path, serotype: str) -> None:
    with open(out_tsv, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "serotype",
            "cluster_id",
            "cluster_size",
            "representative",
            "sequence_id",
            "is_representative",
        ])

        for cluster in clusters:
            for seq_id in cluster["members"]:
                writer.writerow([
                    serotype,
                    cluster["cluster_id"],
                    cluster["size"],
                    cluster["representative"],
                    seq_id,
                    int(seq_id == cluster["representative"]),
                ])


def save_cluster_summary_tsv(clusters: list[dict], out_tsv: Path, serotype: str) -> None:
    with open(out_tsv, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["serotype", "cluster_id", "cluster_size", "representative"])

        for cluster in clusters:
            writer.writerow([
                serotype,
                cluster["cluster_id"],
                cluster["size"],
                cluster["representative"],
            ])


def save_global_summary(rows: list[dict], out_tsv: Path) -> None:
    fieldnames = [
        "serotype",
        "input_fasta",
        "identity",
        "n_input_sequences",
        "n_cdhit_clusters",
        "n_representative_sequences",
        "n_removed_if_only_representatives",
        "fraction_removed_if_only_representatives",
        "singleton_clusters",
        "multi_sequence_clusters",
        "max_cluster_size",
        "mean_cluster_size",
    ]

    with open(out_tsv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_clusters(root_dir: Path) -> list[dict]:
    clusters = []
    files_tsv = sorted(root_dir.glob("DENV[1-4]/*_cluster_membership.tsv"))

    for file_tsv in files_tsv:
        grouped = {}

        with open(file_tsv, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")

            for row in reader:
                key = (row["serotype"], row["cluster_id"])

                if key not in grouped:
                    grouped[key] = {
                        "serotype": row["serotype"],
                        "cluster_id": row["cluster_id"],
                        "cluster_size": int(row["cluster_size"]),
                        "members": [],
                    }

                grouped[key]["members"].append(row["sequence_id"])

        clusters.extend(grouped.values())

    return clusters


def save_sequence_fold(folds: list[dict], out_file: Path) -> None:
    with open(out_file, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["fold", "serotype", "cluster_id", "sequence_id"])

        for i, fold in enumerate(folds):
            for cluster in fold["clusters"]:
                for seq in cluster["members"]:
                    writer.writerow([
                        f"fold_{i + 1}",
                        cluster["serotype"],
                        cluster["cluster_id"],
                        seq,
                    ])


def save_training_split(rows: list[dict], out_file: Path, split_sep: str = ",") -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "fold",
        "index",
        "split",
        "group",
        "serotype",
        "cluster_id",
        "sequence_id",
    ]

    with open(out_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=split_sep)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved split file to: {out_file}")


def save_split_summary(rows: list[dict], out_file: Path) -> None:
    from collections import defaultdict

    summary = defaultdict(int)

    for row in rows:
        key = (row["fold"], row["split"], row["serotype"])
        summary[key] += 1

    summary_file = out_file.with_suffix(".summary.csv")

    with open(summary_file, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["fold", "split", "serotype", "n"])

        for fold, split, serotype in sorted(summary):
            writer.writerow([fold, split, serotype, summary[(fold, split, serotype)]])

    print(f"Saved split summary to: {summary_file}")
