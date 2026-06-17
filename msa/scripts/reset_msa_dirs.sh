#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DENFORMER_DATA_DIR:-data}"
MSA_DIR="$DATA_DIR/msa"

rm -rf "$MSA_DIR/refseq/alignments"
rm -rf "$MSA_DIR/refseq/results"
rm -rf "$MSA_DIR/refseq/plots"

mkdir -p "$MSA_DIR/refseq/alignments"
mkdir -p "$MSA_DIR/refseq/results"
mkdir -p "$MSA_DIR/refseq/plots"
