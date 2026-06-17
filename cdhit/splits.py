import random
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from cdhit.config import seed, val_size
from cdhit.parsing import find_dataset_index


def assign_clusters(clusters: list[dict], k: int) -> list[dict]:
    random.seed(seed)

    folds = [
        {"clusters": [], "n_seq": defaultdict(int)}
        for _ in range(k)
    ]

    by_serotype = defaultdict(list)
    for cluster in clusters:
        by_serotype[cluster["serotype"]].append(cluster)

    for serotype, cluster_list in by_serotype.items():
        cluster_list.sort(key=lambda x: x["cluster_size"], reverse=True)

        for cluster in cluster_list:
            best = min(range(k), key=lambda i: folds[i]["n_seq"][serotype])
            folds[best]["clusters"].append(cluster)
            folds[best]["n_seq"][serotype] += cluster["cluster_size"]

    return folds


def build_sequence_index(folds: list[dict], seq_to_index: dict, all_fasta_ids: list[str]) -> list[dict]:
    rows = []
    missing = []

    for fold_idx, fold in enumerate(folds):
        fold_name = f"fold_{fold_idx + 1}"

        for cluster in fold["clusters"]:
            for seq in cluster["members"]:
                dataset_index = find_dataset_index(
                    seq=seq,
                    seq_to_index=seq_to_index,
                    all_fasta_ids=all_fasta_ids,
                )

                if dataset_index is None:
                    missing.append(seq)
                    continue

                rows.append({
                    "index": dataset_index,
                    "sequence_id": seq,
                    "serotype": cluster["serotype"],
                    "cluster_id": cluster["cluster_id"],
                    "cdhit_fold": fold_name,
                })

    if missing:
        print(f"\nWARNING: {len(missing)} CD-HIT sequences were not found in the original FASTA index.")
        print("First 20 missing sequence ids:")
        for seq in missing[:20]:
            print(seq)

        raise ValueError("Some CD-HIT sequence IDs could not be mapped to dataset indices.")

    return rows


def choose_validation_indices_from_sequences(sequence_rows: list[dict], test_fold_name: str) -> set[int]:
    df = pd.DataFrame(sequence_rows)
    train_pool = df[df["cdhit_fold"] != test_fold_name].copy()

    if val_size <= 0 or len(train_pool) <= 1:
        return set()

    if val_size < 1:
        val_count = int(round(len(train_pool) * val_size))
    else:
        val_count = int(val_size)

    val_count = max(1, min(val_count, len(train_pool) - 1))

    y = train_pool["serotype"]
    counts = y.value_counts()

    can_stratify = (
        len(counts) > 1
        and counts.min() >= 2
        and val_count >= len(counts)
        and (len(train_pool) - val_count) >= len(counts)
    )

    _, val_df = train_test_split(
        train_pool,
        test_size=val_count,
        random_state=seed,
        shuffle=True,
        stratify=y if can_stratify else None,
    )

    return set(val_df["index"].astype(int).tolist())


def build_cluster_aware_training_splits(folds: list[dict], seq_to_index: dict, all_fasta_ids: list[str]) -> list[dict]:
    sequence_rows = build_sequence_index(
        folds=folds,
        seq_to_index=seq_to_index,
        all_fasta_ids=all_fasta_ids,
    )

    rows = []

    for test_fold_idx, _ in enumerate(folds):
        test_fold_name = f"fold_{test_fold_idx + 1}"

        val_indices = choose_validation_indices_from_sequences(
            sequence_rows=sequence_rows,
            test_fold_name=test_fold_name,
        )

        for row in sequence_rows:
            if row["cdhit_fold"] == test_fold_name:
                split = "test"
            elif int(row["index"]) in val_indices:
                split = "val"
            else:
                split = "train"

            rows.append({
                "fold": test_fold_name,
                "index": row["index"],
                "split": split,
                "group": row["cdhit_fold"],
                "serotype": row["serotype"],
                "cluster_id": row["cluster_id"],
                "sequence_id": row["sequence_id"],
            })

    return rows
