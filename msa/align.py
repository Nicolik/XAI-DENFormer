from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run_mafft(
    input_dir: Path,
    output_dir: Path,
    mafft_bin: str = "mafft",
    threads: int = 1,
) -> None:
    if shutil.which(mafft_bin) is None:
        raise RuntimeError(
            f"MAFFT executable not found: {mafft_bin}. Install it with e.g. "
            "'conda install -c bioconda mafft' or run this script on the cluster "
            "where MAFFT is available."
        )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(input_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {input_dir}")

    thread_arg = str(threads if threads and threads > 0 else -1)

    for in_path in fasta_files:
        if sum(1 for line in in_path.open(encoding="utf-8") if line.startswith(">")) < 2:
            print(f"[WARN] Skipping {in_path}: fewer than 2 sequences")
            continue
        out_path = output_dir / f"{in_path.stem}.aln.fasta"
        cmd = [mafft_bin, "--auto", "--thread", thread_arg, str(in_path)]
        print("[RUN]", " ".join(cmd), ">", out_path)
        with out_path.open("w", encoding="utf-8") as out_f:
            subprocess.run(cmd, stdout=out_f, check=True)
        print(f"[OK] {out_path}")


def run_mafft_for_directory(
    input_dir: Path,
    output_dir: Path,
    mafft_bin: str = "mafft",
    threads: int = 1,
) -> None:
    run_mafft(input_dir=input_dir, output_dir=output_dir, mafft_bin=mafft_bin, threads=threads)
