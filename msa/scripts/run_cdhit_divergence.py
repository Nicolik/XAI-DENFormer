from __future__ import annotations

import argparse

import paths
from msa.population_divergence import analyze_population_alignments
from msa.population_plotting import (
    plot_population_nucleotide_level_profiles,
    plot_population_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CD-HIT population-level MSA divergence.")
    parser.add_argument("--max-pairs-per-group", type=int, default=25000)
    args = parser.parse_args()

    cdhit_msa_dir = paths.data_dir / "msa" / "cdhit"
    results_dir = cdhit_msa_dir / "results"
    plots_dir = cdhit_msa_dir / "plots"

    analyze_population_alignments(
        cdhit_msa_dir / "alignments",
        results_dir,
        max_pairs_per_group=args.max_pairs_per_group,
    )
    plot_population_summary(results_dir, plots_dir)
    plot_population_nucleotide_level_profiles(
        results_dir=results_dir,
        plots_dir=plots_dir,
        coordinates_dir=paths.msa_refseq_map_dir,
        prefix="cdhit_msa",
    )


if __name__ == "__main__":
    main()
