from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO

from .config import REGION_ORDER, SEROTYPES

BASES = ["A", "C", "G", "T"]


def shannon_entropy(chars: Iterable[str]) -> float:
    vals = [c.upper() for c in chars if c.upper() in BASES]
    if not vals:
        return 0.0
    total = len(vals)
    h = 0.0
    for b in BASES:
        p = vals.count(b) / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def consensus_similarity(chars: Iterable[str]) -> float:
    vals = [c.upper() for c in chars if c.upper() in BASES]
    if not vals:
        return 0.0
    counts = [vals.count(b) for b in BASES]
    return float(max(counts) / len(vals))


def pairwise_identity(seq_a: str, seq_b: str) -> float:
    comparable = 0
    matches = 0
    for a, b in zip(seq_a.upper(), seq_b.upper()):
        if a == "-" or b == "-":
            continue
        if a not in BASES or b not in BASES:
            continue
        comparable += 1
        matches += int(a == b)
    return matches / comparable if comparable else np.nan


def parse_serotype(record_id: str) -> str:
    denv = record_id.split("|")[0]
    if denv not in SEROTYPES:
        raise ValueError(f"Could not parse serotype from record id: {record_id}")
    return denv


def mean_pairwise_identity(
    seqs_a: List[str],
    seqs_b: List[str] | None = None,
    max_pairs: int = 25000,
    seed: int = 13,
) -> float:
    rng = random.Random(seed)
    if seqs_b is None:
        pairs = list(itertools.combinations(range(len(seqs_a)), 2))
        if not pairs:
            return np.nan
        if len(pairs) > max_pairs:
            pairs = rng.sample(pairs, max_pairs)
        values = [pairwise_identity(seqs_a[i], seqs_a[j]) for i, j in pairs]
    else:
        pairs = [(i, j) for i in range(len(seqs_a)) for j in range(len(seqs_b))]
        if not pairs:
            return np.nan
        if len(pairs) > max_pairs:
            pairs = rng.sample(pairs, max_pairs)
        values = [pairwise_identity(seqs_a[i], seqs_b[j]) for i, j in pairs]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def analyze_population_alignment(
    aln_path: Path,
    region: str,
    max_pairs_per_group: int = 25000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = list(SeqIO.parse(aln_path, "fasta"))
    if len(records) < 2:
        raise ValueError(f"Alignment must contain at least 2 sequences: {aln_path}")

    seqs = [str(r.seq).upper() for r in records]
    aln_len = len(seqs[0])
    if any(len(s) != aln_len for s in seqs):
        raise ValueError(f"Alignment has inconsistent lengths: {aln_path}")

    serotypes = [parse_serotype(r.id) for r in records]
    seqs_by_serotype: Dict[str, List[str]] = {s: [] for s in SEROTYPES}
    for denv, seq in zip(serotypes, seqs):
        seqs_by_serotype[denv].append(seq)

    site_rows = []
    for col_idx in range(aln_len):
        column = [s[col_idx] for s in seqs]
        global_entropy = shannon_entropy(column)
        global_similarity = consensus_similarity(column)
        within_entropies = []
        within_similarities = []
        within_variable_flags = []
        per_serotype_major_base = {}
        per_serotype_major_freq = {}

        for denv in SEROTYPES:
            chars = [seq[col_idx] for seq in seqs_by_serotype[denv]]
            valid_chars = [c for c in chars if c in BASES]
            h = shannon_entropy(valid_chars)
            within_entropies.append(h)
            within_similarities.append(consensus_similarity(valid_chars))
            within_variable_flags.append(int(len(set(valid_chars)) > 1))
            if valid_chars:
                counts = {b: valid_chars.count(b) for b in BASES}
                major_base = max(counts, key=counts.get)
                per_serotype_major_base[f"{denv}_major_base"] = major_base
                per_serotype_major_freq[f"{denv}_major_freq"] = counts[major_base] / len(valid_chars)
            else:
                per_serotype_major_base[f"{denv}_major_base"] = np.nan
                per_serotype_major_freq[f"{denv}_major_freq"] = np.nan

        within_entropy_mean = float(np.mean(within_entropies))
        within_similarity_mean = float(np.mean(within_similarities))
        serotype_informativeness = global_entropy - within_entropy_mean
        valid_global = [c for c in column if c in BASES]
        site_rows.append({
            "region": region,
            "alignment_pos_1based": col_idx + 1,
            "n_sequences": len(records),
            "global_entropy": global_entropy,
            "global_consensus_similarity": global_similarity,
            "within_serotype_entropy_mean": within_entropy_mean,
            "within_serotype_consensus_similarity_mean": within_similarity_mean,
            "serotype_informativeness": serotype_informativeness,
            "global_variable_site": int(len(set(valid_global)) > 1),
            "within_variable_site_mean": float(np.mean(within_variable_flags)),
            "gap_fraction": float(sum(c == "-" for c in column) / len(column)),
            **per_serotype_major_base,
            **per_serotype_major_freq,
        })

    pair_rows = []
    for denv in SEROTYPES:
        identity = mean_pairwise_identity(
            seqs_by_serotype[denv],
            None,
            max_pairs=max_pairs_per_group,
        )
        pair_rows.append({
            "region": region,
            "comparison_type": "within_serotype",
            "comparison": denv,
            "identity": identity,
        })

    for i, denv_a in enumerate(SEROTYPES):
        for denv_b in SEROTYPES[i + 1:]:
            identity = mean_pairwise_identity(
                seqs_by_serotype[denv_a],
                seqs_by_serotype[denv_b],
                max_pairs=max_pairs_per_group,
            )
            pair_rows.append({
                "region": region,
                "comparison_type": "between_serotype",
                "comparison": f"{denv_a}_vs_{denv_b}",
                "identity": identity,
            })

    return pd.DataFrame(site_rows), pd.DataFrame(pair_rows)


def analyze_population_alignments(
    alignment_dir: Path,
    output_dir: Path,
    max_pairs_per_group: int = 25000,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aln_files = sorted(Path(alignment_dir).glob("*.aln.fasta"))
    if not aln_files:
        raise FileNotFoundError(f"No .aln.fasta files found in {alignment_dir}")

    all_sites = []
    all_pairs = []
    for aln_path in aln_files:
        region = aln_path.name.replace(".aln.fasta", "")
        site_df, pair_df = analyze_population_alignment(
            aln_path,
            region,
            max_pairs_per_group=max_pairs_per_group,
        )
        site_df.to_csv(output_dir / f"{region}_population_site_metrics.csv", index=False)
        all_sites.append(site_df)
        all_pairs.append(pair_df)
        print(f"[OK] analyzed {region}: {len(site_df)} aligned sites")

    sites = pd.concat(all_sites, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)

    summary = sites.groupby("region", as_index=False).agg(
        alignment_length=("alignment_pos_1based", "size"),
        mean_global_entropy=("global_entropy", "mean"),
        mean_global_consensus_similarity=("global_consensus_similarity", "mean"),
        mean_within_serotype_entropy=("within_serotype_entropy_mean", "mean"),
        mean_within_serotype_consensus_similarity=("within_serotype_consensus_similarity_mean", "mean"),
        mean_serotype_informativeness=("serotype_informativeness", "mean"),
        max_serotype_informativeness=("serotype_informativeness", "max"),
        global_variable_sites=("global_variable_site", "sum"),
        mean_within_variable_site_fraction=("within_variable_site_mean", "mean"),
        mean_gap_fraction=("gap_fraction", "mean"),
    )
    summary["global_variable_site_fraction"] = summary["global_variable_sites"] / summary["alignment_length"]

    pair_summary = pairs.groupby(["region", "comparison_type"], as_index=False).agg(
        mean_identity=("identity", "mean"),
        min_identity=("identity", "min"),
        max_identity=("identity", "max"),
    )
    pair_wide = pair_summary.pivot(index="region", columns="comparison_type", values="mean_identity").reset_index()
    pair_wide = pair_wide.rename(columns={
        "within_serotype": "mean_within_serotype_identity",
        "between_serotype": "mean_between_serotype_identity",
    })
    summary = summary.merge(pair_wide, on="region", how="left")
    summary["identity_gap_within_minus_between"] = (
        summary["mean_within_serotype_identity"] - summary["mean_between_serotype_identity"]
    )

    order = {region.replace("'", ""): i for i, region in enumerate(REGION_ORDER)}
    summary["_order"] = summary["region"].map(order).fillna(999)
    summary = summary.sort_values("_order").drop(columns="_order")

    sites.to_csv(output_dir / "population_site_metrics_all_regions.csv", index=False)
    pairs.to_csv(output_dir / "population_pairwise_identity_by_region.csv", index=False)
    pair_summary.to_csv(output_dir / "population_pairwise_identity_summary.csv", index=False)
    summary.to_csv(output_dir / "population_region_summary.csv", index=False)
    print(f"[OK] Saved CD-HIT population MSA outputs in {output_dir}")
