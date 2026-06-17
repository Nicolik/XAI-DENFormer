from msa.refseq import export_longest_coordinates, export_refseq_coordinates, write_region_fastas
import paths


def main():
    export_refseq_coordinates(paths.refseq_dir, paths.msa_refseq_map_dir)
    export_longest_coordinates(paths.genomes_dir, paths.refseq_dir, paths.msa_refseq_map_dir)
    write_region_fastas(paths.refseq_dir, paths.msa_refseq_map_dir, paths.msa_region_fasta_dir)


if __name__ == "__main__":
    main()
