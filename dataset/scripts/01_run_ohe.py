import os
import time
import numpy as np
import h5py
from tqdm import tqdm

from paths import genomes_dir, embeddings_dir
from dataset.utils import fasta_to_onehot_embeddings


LABS = ["DENV1", "DENV2", "DENV3", "DENV4"]
SEROTYPE_TO_LABEL = {lab: i for i, lab in enumerate(LABS)}
EMBEDDING_DIM = 4


def main():
    start_time = time.time()

    os.makedirs(embeddings_dir, exist_ok=True)

    log_path = os.path.join(embeddings_dir, "sequence_counts.txt")

    all_embeddings = []
    all_labels = []
    all_ids = []
    max_len = 0

    print("\n==== BUILD OHE EMBEDDINGS ====\n")

    with open(log_path, "w") as log_file:

        for lab in LABS:
            fasta_path = os.path.join(genomes_dir, f"{lab}_merged_meta.fasta")

            if not os.path.exists(fasta_path):
                print(f"WARNING: file not found: {fasta_path}")
                continue

            print(f"\nProcessing {lab}")
            seq_vecs, seq_ids = fasta_to_onehot_embeddings(fasta_path)

            if not seq_vecs:
                print(f"WARNING: no sequences found for {lab}")
                continue

            max_len = max(max_len, max(v.shape[0] for v in seq_vecs))

            all_embeddings.append((lab, seq_vecs, seq_ids))
            all_labels.extend([SEROTYPE_TO_LABEL[lab]] * len(seq_vecs))
            all_ids.extend(seq_ids)

            log_file.write(f"{lab}: {len(seq_vecs)} sequences\n")
            print(f"{lab}: {len(seq_vecs)} sequences")

    total_seq = sum(len(seq_vecs) for _, seq_vecs, _ in all_embeddings)

    print(f"\nTotal sequences: {total_seq}")
    print(f"Max sequence length: {max_len}")

    # =========================
    # SAVE HDF5
    # =========================
    out_path = os.path.join(embeddings_dir, "dataset_ohe.h5")

    print("\n==== WRITING HDF5 ====\n")

    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset(
            "embeddings",
            shape=(total_seq, max_len, EMBEDDING_DIM),
            dtype=np.float32,
            compression="gzip"
        )

        idx = 0

        for lab, seq_vecs, _ in all_embeddings:
            print(f"Writing {lab}...")

            batch = np.zeros((len(seq_vecs), max_len, EMBEDDING_DIM), dtype=np.float32)

            for j, arr in enumerate(tqdm(seq_vecs, desc=f"{lab}", unit="seq")):
                batch[j, :arr.shape[0], :] = arr

            dset[idx: idx + len(seq_vecs)] = batch
            idx += len(seq_vecs)

    print(f"\nSaved embeddings to: {out_path}")

    # =========================
    # SAVE LABELS + IDS
    # =========================
    labels_path = os.path.join(embeddings_dir, "label_matrix.txt")
    np.savetxt(labels_path, np.array(all_labels, dtype=np.int32), fmt="%d")
    print(f"Saved labels to: {labels_path}")

    ids_path = os.path.join(embeddings_dir, "seq_ids_ohe.txt")
    with open(ids_path, "w") as f:
        for sid in all_ids:
            f.write(sid + "\n")
    print(f"Saved sequence IDs to: {ids_path}")

    # =========================
    # TIME
    # =========================
    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    print(f"\nDone in {m} min {s} sec")


if __name__ == "__main__":
    main()
