from __future__ import annotations

import argparse

import paths
from msa.align import run_mafft_for_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAFFT on CD-HIT-derived regional FASTA files.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mafft-bin", type=str, default="mafft")
    args = parser.parse_args()

    cdhit_msa_dir = paths.data_dir / "msa" / "cdhit"
    run_mafft_for_directory(
        cdhit_msa_dir / "region_fastas",
        cdhit_msa_dir / "alignments",
        mafft_bin=args.mafft_bin,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
