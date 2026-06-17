import subprocess

from cdhit.config import fasta_extensions, identity, memory_mb, threads
from cdhit.io import (
    save_cluster_membership_tsv,
    save_cluster_summary_tsv,
    save_global_summary,
)
from cdhit.parsing import (
    choose_word_size,
    count_fasta_records,
    infer_serotype_name,
    parse_clstr,
)
from cdhit.plotting import plot_all_sequences_vs_representatives
from paths import cdhit_dir, genomes_dir, as_path


def run_command_live(cmd: list[str]) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")


def summarize_clusters(fasta_file, clusters: list[dict], serotype: str) -> dict:
    n_input_sequences = count_fasta_records(fasta_file)
    n_clusters = len(clusters)
    n_representatives = n_clusters
    n_removed = n_input_sequences - n_representatives

    singleton_clusters = sum(1 for c in clusters if c["size"] == 1)
    multi_sequence_clusters = sum(1 for c in clusters if c["size"] > 1)
    max_cluster_size = max([c["size"] for c in clusters], default=0)
    mean_cluster_size = sum(c["size"] for c in clusters) / len(clusters) if clusters else 0.0
    fraction_removed = n_removed / n_input_sequences if n_input_sequences else 0.0

    return {
        "serotype": serotype,
        "input_fasta": fasta_file.name,
        "identity": identity,
        "n_input_sequences": n_input_sequences,
        "n_cdhit_clusters": n_clusters,
        "n_representative_sequences": n_representatives,
        "n_removed_if_only_representatives": n_removed,
        "fraction_removed_if_only_representatives": fraction_removed,
        "singleton_clusters": singleton_clusters,
        "multi_sequence_clusters": multi_sequence_clusters,
        "max_cluster_size": max_cluster_size,
        "mean_cluster_size": mean_cluster_size,
    }


def print_summary(row: dict) -> None:
    print(f"\nSummary for {row['serotype']}")
    print(f"  Input sequences:                 {row['n_input_sequences']}")
    print(f"  CD-HIT clusters:                 {row['n_cdhit_clusters']}")
    print(f"  Representative sequences:        {row['n_representative_sequences']}")
    print(f"  Removed if only representatives: {row['n_removed_if_only_representatives']}")
    print(f"  Fraction removed:                {row['fraction_removed_if_only_representatives']:.4f}")
    print(f"  Singleton clusters:              {row['singleton_clusters']}")
    print(f"  Multi-sequence clusters:         {row['multi_sequence_clusters']}")
    print(f"  Max cluster size:                {row['max_cluster_size']}")
    print(f"  Mean cluster size:               {row['mean_cluster_size']:.2f}")


def main() -> None:
    input_dir = genomes_dir
    out_root = cdhit_dir
    out_root.mkdir(parents=True, exist_ok=True)

    fasta_files = []
    for ext in fasta_extensions:
        fasta_files.extend(input_dir.glob(f"*{ext}"))
    fasta_files = sorted(fasta_files)

    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in: {input_dir}")

    word_size = choose_word_size(identity)
    summary_rows = []

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {out_root}")
    print(f"CD-HIT identity: {identity}")
    print(f"CD-HIT word size: {word_size}")

    for fasta_file in fasta_files:
        serotype = infer_serotype_name(fasta_file.name)
        serotype_out_dir = out_root / serotype
        serotype_out_dir.mkdir(parents=True, exist_ok=True)

        cdhit_output_fasta = serotype_out_dir / f"{serotype}_cdhit.fasta"
        clstr_path = cdhit_output_fasta.with_suffix(cdhit_output_fasta.suffix + ".clstr")

        print("\n" + "=" * 80)
        print(f"Processing: {fasta_file.name}")
        print(f"Detected serotype: {serotype}")
        print("=" * 80)

        cmd = [
            "cd-hit-est",
            "-i", str(fasta_file),
            "-o", str(cdhit_output_fasta),
            "-c", str(identity),
            "-n", word_size,
            "-T", str(threads),
            "-M", str(memory_mb),
            "-d", "0",
        ]

        run_command_live(cmd)

        if not clstr_path.exists():
            raise FileNotFoundError(f"CD-HIT cluster file not found: {clstr_path}")

        clusters = parse_clstr(clstr_path)

        save_cluster_membership_tsv(
            clusters,
            serotype_out_dir / f"{serotype}_cluster_membership.tsv",
            serotype,
        )
        save_cluster_summary_tsv(
            clusters,
            serotype_out_dir / f"{serotype}_cluster_summary.tsv",
            serotype,
        )

        row = summarize_clusters(fasta_file, clusters, serotype)
        summary_rows.append(row)
        print_summary(row)

    summary_rows = sorted(summary_rows, key=lambda x: x["serotype"])

    save_global_summary(summary_rows, out_root / "cdhit_summary_by_serotype.tsv")
    plot_all_sequences_vs_representatives(
        summary_rows,
        out_root / "all_sequences_vs_cdhit_representatives.png",
    )

    print(f"\nOutputs written to: {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()
