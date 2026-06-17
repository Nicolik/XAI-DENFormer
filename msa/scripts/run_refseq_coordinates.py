from pathlib import Path
import argparse
import paths
from msa.refseq import export_longest_coordinates, export_refseq_coordinates


def main():
    parser = argparse.ArgumentParser(description="Generate RefSeq and longest-genome coordinate maps.")
    parser.add_argument("--skip-longest", action="store_true", help="Only generate RefSeq coordinate maps.")
    args = parser.parse_args()

    export_refseq_coordinates(
        refseq_dir=paths.refseq_dir,
        output_dir=paths.msa_refseq_map_dir,
    )
    if not args.skip_longest:
        export_longest_coordinates(
            genomes_dir=paths.genomes_dir,
            refseq_dir=paths.refseq_dir,
            output_dir=paths.msa_refseq_map_dir,
        )


if __name__ == "__main__":
    main()
