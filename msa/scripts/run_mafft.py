import paths
from msa.align import run_mafft


def main():
    run_mafft(
        input_dir=paths.msa_region_fasta_dir,
        output_dir=paths.msa_alignment_dir,
    )


if __name__ == "__main__":
    main()
