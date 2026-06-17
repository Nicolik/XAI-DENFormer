"""Run the MSA analysis workflow.

This module provides a portable Python entry point that derives project paths
from ``paths.py``.

Examples
--------
Run the complete CD-HIT MSA + serotype variability workflow::

    python -m msa.run

Reuse existing alignments and only regenerate the serotype variability panels::

    python -m msa.run --skip-cdhit-all

Regenerate only the region-level recap from existing per-region outputs::

    python -m msa.run --skip-cdhit-all --skip-serotype-variability
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import paths
from msa.scripts.run_cdhit_all import main as run_cdhit_all_main
from msa.serotype_variability import main as run_serotype_variability_main
from msa.serotype_variability_recap import main as run_serotype_variability_recap_main


def _alignment_files(alignment_dir: Path) -> list[Path]:
    return sorted(alignment_dir.glob("*.aln.fasta"))


def _run_cdhit_all(args: argparse.Namespace, input_dir: Path) -> None:
    argv = [
        "--input-dir", str(input_dir),
        "--output-dir", str(args.msa_dir),
        "--max-per-serotype", str(args.max_per_serotype),
        "--threads", str(args.threads),
        "--mafft-bin", args.mafft_bin,
        "--max-pairs-per-group", str(args.max_pairs_per_group),
        "--min-region-length-fraction", str(args.min_region_length_fraction),
    ]
    run_cdhit_all_main(argv)


def _run_serotype_variability_for_alignments(
    alignments: Iterable[Path],
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for alignment_path in alignments:
        region = alignment_path.name.removesuffix(".aln.fasta")
        out_dir = output_root / region
        print(f"[RUN] Serotype-aware MSA variability for region: {region}")
        run_serotype_variability_main([
            "--msa", str(alignment_path),
            "--out-dir", str(out_dir),
            "--window", str(args.window),
            "--dpi", str(args.dpi),
            "--top-n", str(args.top_n),
            "--alphabet", args.alphabet,
        ])


def _run_serotype_recap(input_dir: Path, output_dir: Path, dpi: int) -> None:
    print("[RUN] Region-level serotype variability recap")
    run_serotype_variability_recap_main([
        "--input-dir", str(input_dir),
        "--out-dir", str(output_dir),
        "--dpi", str(dpi),
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MSA analysis workflow.")
    parser.add_argument("--input-dir", type=Path, default=Path(paths.cdhit_dir),
                        help="Directory containing CD-HIT input FASTA files.")
    parser.add_argument("--msa-dir", type=Path, default=Path(paths.data_dir) / "msa" / "cdhit",
                        help="Root directory for MSA outputs.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mafft-bin", default="mafft")
    parser.add_argument("--max-per-serotype", type=int, default=0)
    parser.add_argument("--max-pairs-per-group", type=int, default=25000)
    parser.add_argument("--min-region-length-fraction", type=float, default=0.65)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--alphabet", default="auto", choices=("auto", "nt", "aa"))
    parser.add_argument("--skip-cdhit-all", action="store_true",
                        help="Reuse existing MSA alignments instead of rebuilding them.")
    parser.add_argument("--skip-serotype-variability", action="store_true",
                        help="Skip per-region serotype variability panels.")
    parser.add_argument("--skip-serotype-recap", action="store_true",
                        help="Skip the region-level serotype variability recap.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    msa_dir = args.msa_dir
    alignments_dir = msa_dir / "alignments"
    serotype_panel_dir = msa_dir / "serotype_variability_panel" / "strategy_consensus"
    serotype_recap_dir = serotype_panel_dir / "region_recap"

    print(f"[INFO] Data dir: {paths.data_dir}")
    print(f"[INFO] CD-HIT input dir: {args.input_dir}")
    print(f"[INFO] CD-HIT MSA dir: {msa_dir}")

    if not args.skip_cdhit_all:
        _run_cdhit_all(args, args.input_dir)

    if not args.skip_serotype_variability:
        if not alignments_dir.is_dir():
            raise FileNotFoundError(
                f"Missing alignment directory: {alignments_dir}. "
                "Run without --skip-cdhit-all first, or check paths.py."
            )
        alignment_files = _alignment_files(alignments_dir)
        if not alignment_files:
            raise FileNotFoundError(f"No .aln.fasta files found in {alignments_dir}")
        _run_serotype_variability_for_alignments(alignment_files, serotype_panel_dir, args)
        print(f"[OK] Serotype variability outputs written under: {serotype_panel_dir}")

    if not args.skip_serotype_recap:
        if not serotype_panel_dir.is_dir():
            raise FileNotFoundError(
                f"Missing serotype panel directory: {serotype_panel_dir}. "
                "Run without --skip-serotype-variability first."
            )
        _run_serotype_recap(serotype_panel_dir, serotype_recap_dir, args.dpi)
        print(f"[OK] Region-level serotype recap written under: {serotype_recap_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
