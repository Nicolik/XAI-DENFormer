import paths
from msa.divergence import analyze_alignments
from msa.plotting import (
    plot_nucleotide_level_msa_profiles,
    plot_region_summary,
    plot_site_entropy,
)


def main():
    analyze_alignments(
        alignment_dir=paths.msa_alignment_dir,
        output_dir=paths.msa_results_dir,
    )
    plot_region_summary(
        results_dir=paths.msa_results_dir,
        output_dir=paths.msa_plots_dir,
    )
    plot_site_entropy(
        results_dir=paths.msa_results_dir,
        output_dir=paths.msa_plots_dir,
    )
    plot_nucleotide_level_msa_profiles(
        results_dir=paths.msa_results_dir,
        output_dir=paths.msa_plots_dir,
        coordinates_dir=paths.msa_refseq_map_dir,
        prefix="msa",
    )


if __name__ == "__main__":
    main()
