from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

BASES = ["A", "C", "G", "T"]


def shannon_entropy(chars: list[str]) -> float:
    chars = [c.upper() for c in chars if c.upper() in BASES]
    if not chars:
        return 0.0
    total = len(chars)
    h = 0.0
    for b in BASES:
        p = chars.count(b) / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def consensus_similarity(chars: list[str]) -> float:
    chars = [c.upper() for c in chars if c.upper() in BASES]
    if not chars:
        return 0.0
    counts = [chars.count(b) for b in BASES]
    return float(max(counts) / len(chars))


def pairwise_identity(seq_a: str, seq_b: str) -> float:
    comparable = 0
    matches = 0
    for a, b in zip(seq_a.upper(), seq_b.upper()):
        if a == "-" or b == "-":
            continue
        comparable += 1
        matches += int(a == b)
    return matches / comparable if comparable else np.nan


def analyze_alignment(aln_path: Path, region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = list(SeqIO.parse(aln_path, "fasta"))
    if len(records) < 2:
        raise ValueError(f"Alignment must contain at least 2 sequences: {aln_path}")

    names = [r.id.split("|")[0] for r in records]
    seqs = [str(r.seq).upper() for r in records]
    aln_len = len(seqs[0])
    if any(len(s) != aln_len for s in seqs):
        raise ValueError(f"Alignment has inconsistent lengths: {aln_path}")

    site_rows = []
    ungapped_pos = {name: 0 for name in names}
    for col_idx in range(aln_len):
        column = [s[col_idx] for s in seqs]
        for name, char in zip(names, column):
            if char != "-":
                ungapped_pos[name] += 1
        nongap = [c for c in column if c != "-"]
        variable = len(set(c for c in nongap if c in BASES)) > 1
        site_rows.append({
            "region": region,
            "alignment_pos_1based": col_idx + 1,
            "entropy": shannon_entropy(column),
            "consensus_similarity": consensus_similarity(column),
            "variable_site": int(variable),
            "gap_fraction": float(sum(c == "-" for c in column) / len(column)),
            **{f"{name}_base": char for name, char in zip(names, column)},
            **{f"{name}_ungapped_pos": ungapped_pos[name] if column[i] != "-" else np.nan for i, name in enumerate(names)},
        })

    pair_rows = []
    for i, name_i in enumerate(names):
        for j in range(i + 1, len(names)):
            name_j = names[j]
            pair_rows.append({
                "region": region,
                "pair": f"{name_i}_vs_{name_j}",
                "identity": pairwise_identity(seqs[i], seqs[j]),
                "divergence": 1.0 - pairwise_identity(seqs[i], seqs[j]),
            })

    return pd.DataFrame(site_rows), pd.DataFrame(pair_rows)


def analyze_alignments(alignment_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aln_files = sorted(alignment_dir.glob("*.aln.fasta"))
    if not aln_files:
        raise FileNotFoundError(f"No .aln.fasta files found in {alignment_dir}")

    all_sites = []
    all_pairs = []
    for aln_path in aln_files:
        region = aln_path.name.replace(".aln.fasta", "")
        site_df, pair_df = analyze_alignment(aln_path, region)
        site_df.to_csv(output_dir / f"{region}_site_divergence.csv", index=False)
        all_sites.append(site_df)
        all_pairs.append(pair_df)

    sites = pd.concat(all_sites, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)

    summary = sites.groupby("region", as_index=False).agg(
        alignment_length=("alignment_pos_1based", "size"),
        mean_entropy=("entropy", "mean"),
        max_entropy=("entropy", "max"),
        mean_consensus_similarity=("consensus_similarity", "mean"),
        variable_sites=("variable_site", "sum"),
        mean_gap_fraction=("gap_fraction", "mean"),
    )
    summary["variable_site_fraction"] = summary["variable_sites"] / summary["alignment_length"]

    pair_summary = pairs.groupby("region", as_index=False).agg(
        mean_pairwise_identity=("identity", "mean"),
        min_pairwise_identity=("identity", "min"),
        max_pairwise_divergence=("divergence", "max"),
    )

    summary = summary.merge(pair_summary, on="region", how="left")
    summary.to_csv(output_dir / "region_divergence_summary.csv", index=False)
    pairs.to_csv(output_dir / "pairwise_identity_by_region.csv", index=False)
    sites.to_csv(output_dir / "site_divergence_all_regions.csv", index=False)

    print(f"[OK] Saved divergence outputs in {output_dir}")
