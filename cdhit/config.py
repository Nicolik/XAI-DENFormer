identity = 0.95

threads = 4
memory_mb = 16000

fasta_extensions = [".fasta", ".fa", ".fna"]

k_folds = 5
seed = 42

generate_splits = True
val_size = 0.10
split_sep = ","

# Used in output filenames. Keep it aligned with identity unless you intentionally want a custom label.
identity_label = f"{identity:.2f}"

fasta_files_by_serotype = {
    "DENV1_merged_meta.fasta": "DENV1",
    "DENV2_merged_meta.fasta": "DENV2",
    "DENV3_merged_meta.fasta": "DENV3",
    "DENV4_merged_meta.fasta": "DENV4",
}

desired_serotypes = ["DENV1", "DENV2", "DENV3", "DENV4"]
show_titles = False
