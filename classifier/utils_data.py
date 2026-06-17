import os
import numpy as np
import pandas as pd
import h5py


def get_dataset(input_dir, k_type):
    file_path = input_dir / f"dataset_{k_type}.h5"
    with h5py.File(file_path, "r") as f:
        samples = f["embeddings"][:]  # (n_seq, max_len, emb_dim)

    print("Samples shape (tot):", samples.shape)

    labels_path = input_dir / "label_matrix.txt"
    with open(labels_path, "r") as f:
        targets = np.array([int(line.strip()) for line in f])  # ogni riga = una label
    print("Targets shape (tot):", targets.shape)

    return samples, targets


def split_data_from_csv(split_csv):
    """
    Read a split CSV and return train/val/test indices.
    The Dataset will use these indices to access samples/targets lazily.
    """
    split_info = pd.read_csv(split_csv)

    train_idx = split_info.loc[split_info["subset"] == "train", "idx"].values
    val_idx   = split_info.loc[split_info["subset"] == "val", "idx"].values
    test_idx  = split_info.loc[split_info["subset"] == "test", "idx"].values

    return train_idx, val_idx, test_idx


def dataset_shape(dataset):
    # Assume dataset[i][0] is (L, E) tensor/array, dataset[i][1] is target
    x, _ = dataset[0]
    L, E = x.shape
    N = len(dataset)
    return N, L, E


def collect_shapes(datasets, names, output_dir=None):
    """
    Collect shapes of datasets and save as CSV (if output_dir provided).

    Parameters
    ----------
    datasets : list
        List of dataset objects (must support __getitem__ and __len__).
    names : list
        List of subset names (e.g., ["train", "validation", "test"]).
    output_dir : str, optional
        Directory to save 'dataset_shapes.csv'. If None, CSV is not saved.

    Returns
    -------
    pd.DataFrame
        DataFrame with subset, N, L, E.
    """

    rows = []
    for ds, name in zip(datasets, names):
        rows.append([name, *dataset_shape(ds)])

    # Add "all" (concatenation)
    if len(datasets) > 1:
        all_indices = np.concatenate([ds.indices for ds in datasets if hasattr(ds, "indices")])
        all_dataset = datasets[0].__class__(datasets[0].samples, datasets[0].targets, indices=all_indices)
        rows.append(["all", *dataset_shape(all_dataset)])

    df = pd.DataFrame(rows, columns=["subset", "N (samples)", "L (sequence)", "E (embedding)"])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "dataset_shapes.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved dataset shapes to {out_path}")

    return df
