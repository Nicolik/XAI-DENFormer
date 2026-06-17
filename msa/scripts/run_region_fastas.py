import paths
from msa.refseq import write_region_fastas


def main():
    write_region_fastas(
        refseq_dir=paths.refseq_dir,
        coordinates_dir=paths.msa_refseq_map_dir,
        output_dir=paths.msa_region_fasta_dir,
    )


if __name__ == "__main__":
    main()
