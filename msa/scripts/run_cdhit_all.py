from __future__ import annotations

import argparse
from pathlib import Path

import paths
from msa.refseq import export_refseq_coordinates
from msa.cdhit import write_existing_cdhit_region_fastas
from msa.align import run_mafft_for_directory
from msa.population_divergence import analyze_population_alignments
from msa.population_plotting import plot_population_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run region-aware MSA analysis using already generated CD-HIT representatives "
            "from Data/cdhit/NCBI+GISAID-META/DENV*/DENV*_cdhit.fasta."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=paths.cdhit_dir)
    parser.add_argument("--output-dir", type=Path, default=paths.data_dir / "msa" / "cdhit")
    parser.add_argument("--max-per-serotype", type=int, default=0, help="0 means use all CD-HIT representatives.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mafft-bin", type=str, default="mafft")
    parser.add_argument("--max-pairs-per-group", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-region-length-fraction", type=float, default=0.65)
    args = parser.parse_args(argv)

    cdhit_msa_dir = args.output_dir
    map_dir = paths.msa_refseq_map_dir
    region_fasta_dir = cdhit_msa_dir / "region_fastas"
    alignment_dir = cdhit_msa_dir / "alignments"
    results_dir = cdhit_msa_dir / "results"
    plots_dir = cdhit_msa_dir / "plots"

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
    run_mafft_for_directory(
        region_fasta_dir,
        alignment_dir,
        mafft_bin=args.mafft_bin,
        threads=args.threads,
    )
    analyze_population_alignments(
        alignment_dir,
        results_dir,
        max_pairs_per_group=args.max_pairs_per_group,
    )
    plot_population_summary(results_dir, plots_dir)


if __name__ == "__main__":
    main()
