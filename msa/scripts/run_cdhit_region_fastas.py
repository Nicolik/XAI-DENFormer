from __future__ import annotations

import argparse
from pathlib import Path

import paths
from msa.refseq import export_refseq_coordinates
from msa.cdhit import write_existing_cdhit_region_fastas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract region FASTA files from existing CD-HIT representative FASTA files."
    )
    parser.add_argument("--input-dir", type=Path, default=paths.cdhit_dir)
    parser.add_argument("--max-per-serotype", type=int, default=0, help="0 means use all representatives.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-region-length-fraction", type=float, default=0.65)
    args = parser.parse_args()

    cdhit_msa_dir = paths.data_dir / "msa" / "cdhit"
    region_fasta_dir = cdhit_msa_dir / "region_fastas"
    map_dir = paths.msa_refseq_map_dir

    max_per_serotype = args.max_per_serotype if args.max_per_serotype > 0 else None

    export_refseq_coordinates(paths.refseq_dir, map_dir)
    write_existing_cdhit_region_fastas(
        cdhit_dir=args.input_dir,
        refseq_dir=paths.refseq_dir,
        coordinates_dir=map_dir,
        output_dir=region_fasta_dir,
        max_per_serotype=max_per_serotype,
        seed=args.seed,
        min_region_length_fraction=args.min_region_length_fraction,
    )


if __name__ == "__main__":
    main()
