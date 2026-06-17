"""Serotype-aware variability plots for aligned dengue MSAs.

Run it as a module on one shared aligned FASTA, e.g. the pipeline uses:

python -m msa.serotype_variability \
  --msa "${DATA_DIR}/msa/cdhit/alignments/E.aln.fasta" \
  --out-dir "${DATA_DIR}/msa/cdhit/serotype_variability_panel/strategy_consensus/E" \
  --window 21

The module intentionally avoids Biopython and seaborn so it can run in a
minimal project environment with numpy/pandas/matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


FASTA_EXTENSIONS = {".fa", ".fasta", ".faa", ".fas", ".aln", ".fna", ".msa"}
GAP_CHARS = {"-", "."}
AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
NT_ALPHABET = tuple("ACGT")
NT_IUPAC = set("ACGTUNRYSWKMBDHV")
ATTENTION_VALUE_CANDIDATES = (
    "attention_mean",
    "mean_attention",
    "attention",
    "attn_mean",
    "score",
    "importance",
    "denformer_mean",
    "mean",
)
POSITION_CANDIDATES = ("position", "pos", "msa_pos", "ref_pos", "site", "idx", "index")


@dataclass(frozen=True)
class SequenceRecord:
    header: str
    sequence: str
    source_file: str
    serotype: str


@dataclass(frozen=True)
class AnalysisBundle:
    records: List[SequenceRecord]
    serotypes: List[str]
    alphabet: Tuple[str, ...]
    msa_length: int
    position_table: pd.DataFrame
    metrics: pd.DataFrame
    pairs: pd.DataFrame
    aggregate: pd.DataFrame
    summary_serotype: pd.DataFrame
    summary_pair: pd.DataFrame
    distributions: Dict[str, np.ndarray]


def natural_serotype_key(label: str) -> Tuple[int, str]:
    m = re.search(r"([1-4])", label)
    if m:
        return int(m.group(1)), label
    return 99, label


def normalize_serotype(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if not text:
        return None
    m = re.search(r"(?:DENV|SEROTYPE|TYPE|ST)?[-_ :=]*([1-4])(?=$|[^A-Z0-9])", text, flags=re.IGNORECASE)
    if m:
        return f"DENV{m.group(1)}"
    if text in {"1", "2", "3", "4"}:
        return f"DENV{text}"
    return None


def infer_serotype(text: str, custom_regex: Optional[str] = None) -> Optional[str]:
    if custom_regex:
        m = re.search(custom_regex, text, flags=re.IGNORECASE)
        if m:
            value = m.group(1) if m.groups() else m.group(0)
            norm = normalize_serotype(value)
            if norm:
                return norm
    patterns = [
        r"\bDENV[-_ ]?([1-4])(?=$|[^A-Z0-9])",
        r"\bDENGUE[-_ ]?VIRUS[-_ ]?([1-4])(?=$|[^A-Z0-9])",
        r"\bSEROTYPE[:=_ -]*([1-4])(?=$|[^A-Z0-9])",
        r"\bSERO[:=_ -]*([1-4])(?=$|[^A-Z0-9])",
        r"\bTYPE[:=_ -]*([1-4])(?=$|[^A-Z0-9])",
        r"\bST[:=_ -]*([1-4])(?=$|[^A-Z0-9])",
        r"(?:^|[^A-Z0-9])D([1-4])(?:[^A-Z0-9]|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return f"DENV{m.group(1)}"
    return None


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks).upper()))
                header = line[1:].strip()
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence data before FASTA header at line {line_number}")
                chunks.append(re.sub(r"\s+", "", line))
        if header is not None:
            records.append((header, "".join(chunks).upper()))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def find_fasta_files(msa_dir: Path) -> List[Path]:
    files = [p for p in msa_dir.rglob("*") if p.is_file() and p.suffix.lower() in FASTA_EXTENSIONS]
    return sorted(files)


def load_records(
    msa_paths: Sequence[Path],
    serotype_regex: Optional[str] = None,
    drop_unknown: bool = True,
) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    unknown_headers: List[str] = []
    for path in msa_paths:
        file_serotype = infer_serotype(path.name, serotype_regex)
        for header, sequence in parse_fasta(path):
            serotype = infer_serotype(header, serotype_regex) or file_serotype
            if serotype is None:
                unknown_headers.append(f"{path.name}::{header[:120]}")
                if drop_unknown:
                    continue
                serotype = "UNKNOWN"
            records.append(
                SequenceRecord(
                    header=header,
                    sequence=sequence,
                    source_file=str(path),
                    serotype=serotype,
                )
            )
    if not records:
        msg = "No usable records after serotype inference. "
        if unknown_headers:
            msg += "Examples without serotype: " + "; ".join(unknown_headers[:5])
        raise ValueError(msg)

    lengths = pd.Series([len(r.sequence) for r in records]).value_counts().sort_index()
    if len(lengths) > 1:
        detail = ", ".join(f"L={int(k)}:n={int(v)}" for k, v in lengths.items())
        raise ValueError(
            "MSA records do not all have the same aligned length. "
            "Inter-serotype position-wise plots require one shared alignment coordinate system. "
            f"Observed lengths: {detail}"
        )
    return records


def detect_alphabet(records: Sequence[SequenceRecord], requested: str) -> Tuple[str, ...]:
    requested = requested.lower()
    if requested == "aa":
        return AA_ALPHABET
    if requested == "nt":
        return NT_ALPHABET

    observed = set()
    sample_limit = min(len(records), 5000)
    for record in records[:sample_limit]:
        observed.update(ch for ch in record.sequence.upper() if ch not in GAP_CHARS and ch != " ")
    observed.discard("*")

    if requested == "observed":
        clean = sorted(ch for ch in observed if len(ch) == 1 and ch.isprintable())
        if not clean:
            raise ValueError("No non-gap symbols detected in MSA")
        return tuple(clean)

    if requested != "auto":
        raise ValueError(f"Unknown alphabet mode: {requested}")

    # If everything is compatible with nucleotide IUPAC and very few amino-acid-only
    # symbols occur, use A/C/G/T. Ambiguous nucleotide symbols are counted as unknown
    # unless --alphabet observed is requested.
    if observed and observed.issubset(NT_IUPAC):
        return NT_ALPHABET
    return AA_ALPHABET


def build_position_table(
    msa_length: int,
    records: Sequence[SequenceRecord],
    reference_regex: Optional[str] = None,
) -> pd.DataFrame:
    table = pd.DataFrame({"msa_pos": np.arange(1, msa_length + 1, dtype=int)})
    table["ref_pos"] = np.nan
    table["ref_symbol"] = ""
    if not reference_regex:
        return table

    ref_record = None
    for record in records:
        if re.search(reference_regex, record.header, flags=re.IGNORECASE):
            ref_record = record
            break
    if ref_record is None:
        raise ValueError(f"No reference sequence header matched --reference-regex {reference_regex!r}")

    ref_positions: List[object] = []
    ref_symbols: List[str] = []
    cursor = 0
    for symbol in ref_record.sequence:
        if symbol in GAP_CHARS:
            ref_positions.append(np.nan)
            ref_symbols.append(symbol)
        else:
            cursor += 1
            ref_positions.append(cursor)
            ref_symbols.append(symbol)
    table["ref_pos"] = ref_positions
    table["ref_symbol"] = ref_symbols
    return table


def records_to_byte_array(records: Sequence[SequenceRecord]) -> np.ndarray:
    if not records:
        raise ValueError("No records supplied")
    length = len(records[0].sequence)
    blob = "".join(r.sequence for r in records).encode("ascii", errors="replace")
    return np.frombuffer(blob, dtype="S1").reshape(len(records), length)


def column_distributions_for_group(
    records: Sequence[SequenceRecord],
    alphabet: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    arr = records_to_byte_array(records)
    n_seq, msa_length = arr.shape
    alphabet_bytes = [a.encode("ascii") for a in alphabet]
    counts = np.zeros((msa_length, len(alphabet)), dtype=np.float64)
    for idx, symbol in enumerate(alphabet_bytes):
        if symbol == b"T":
            counts[:, idx] = ((arr == b"T") | (arr == b"U")).sum(axis=0)
        else:
            counts[:, idx] = (arr == symbol).sum(axis=0)

    gap_count = np.zeros(msa_length, dtype=np.float64)
    for gap in GAP_CHARS:
        gap_count += (arr == gap.encode("ascii")).sum(axis=0)

    valid_count = counts.sum(axis=1)
    unknown_count = n_seq - valid_count - gap_count
    unknown_count = np.maximum(unknown_count, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / valid_count[:, None]
    p[~np.isfinite(p)] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    entropy = -np.sum(np.where(p > 0, p * logp, 0.0), axis=1)
    max_entropy = math.log2(len(alphabet)) if len(alphabet) > 1 else 1.0
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy
    intra_similarity = np.sum(p * p, axis=1)
    consensus_idx = np.argmax(counts, axis=1)
    consensus_count = counts[np.arange(msa_length), consensus_idx]
    consensus_fraction = np.divide(
        consensus_count,
        valid_count,
        out=np.zeros_like(consensus_count),
        where=valid_count > 0,
    )
    consensus_symbol = np.array(alphabet, dtype=object)[consensus_idx]
    consensus_symbol[valid_count == 0] = ""

    df = pd.DataFrame(
        {
            "n_sequences": n_seq,
            "valid_count": valid_count.astype(int),
            "gap_count": gap_count.astype(int),
            "unknown_count": unknown_count.astype(int),
            "valid_fraction": valid_count / float(n_seq),
            "gap_fraction": gap_count / float(n_seq),
            "unknown_fraction": unknown_count / float(n_seq),
            "entropy_bits": entropy,
            "entropy_norm": entropy_norm,
            "conservation_norm": 1.0 - entropy_norm,
            "intra_similarity": intra_similarity,
            "consensus": consensus_symbol,
            "consensus_fraction": consensus_fraction,
            "effective_symbols": np.power(2.0, entropy),
        }
    )
    return p, df


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0)
    out = 0.5 * np.sum(kl_pm, axis=1) + 0.5 * np.sum(kl_qm, axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def cosine_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    num = np.sum(p * q, axis=1)
    den = np.sqrt(np.sum(p * p, axis=1)) * np.sqrt(np.sum(q * q, axis=1))
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def distribution_overlap_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(np.minimum(p, q), axis=1)


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    arr = pd.Series(values, dtype="float64")
    if window <= 1:
        return arr.to_numpy()
    min_periods = max(1, min(window, int(math.ceil(window / 3))))
    return arr.rolling(window=window, center=True, min_periods=min_periods).mean().to_numpy()


def x_coordinate(position_table: pd.DataFrame, coordinate: str) -> Tuple[np.ndarray, np.ndarray, str]:
    coordinate = coordinate.lower()
    if coordinate == "msa":
        return position_table["msa_pos"].to_numpy(dtype=float), np.ones(len(position_table), dtype=bool), "MSA position"
    if coordinate == "ref":
        valid = position_table["ref_pos"].notna().to_numpy()
        if not valid.any():
            raise ValueError("--x-coordinate ref requires --reference-regex and at least one non-gap reference site")
        return position_table["ref_pos"].to_numpy(dtype=float), valid, "Reference position"
    raise ValueError("--x-coordinate must be 'msa' or 'ref'")


def analyze_msa(
    records: Sequence[SequenceRecord],
    alphabet: Sequence[str],
    reference_regex: Optional[str] = None,
) -> AnalysisBundle:
    msa_length = len(records[0].sequence)
    position_table = build_position_table(msa_length, records, reference_regex)
    serotypes = sorted({r.serotype for r in records}, key=natural_serotype_key)

    distributions: Dict[str, np.ndarray] = {}
    metric_frames: List[pd.DataFrame] = []

    for serotype in serotypes:
        subset = [r for r in records if r.serotype == serotype]
        p, group_df = column_distributions_for_group(subset, alphabet)
        distributions[serotype] = p
        group_df = pd.concat([position_table.copy(), group_df], axis=1)
        group_df.insert(0, "serotype", serotype)
        metric_frames.append(group_df)

    p_global, global_df = column_distributions_for_group(records, alphabet)
    distributions["GLOBAL"] = p_global
    global_df = pd.concat([position_table.copy(), global_df], axis=1)
    global_df.insert(0, "serotype", "GLOBAL")
    metric_frames.append(global_df)

    metrics = pd.concat(metric_frames, ignore_index=True)

    pair_frames: List[pd.DataFrame] = []
    for s1, s2 in itertools.combinations(serotypes, 2):
        p1 = distributions[s1]
        p2 = distributions[s2]
        m1 = metrics[metrics["serotype"] == s1].reset_index(drop=True)
        m2 = metrics[metrics["serotype"] == s2].reset_index(drop=True)
        overlap = distribution_overlap_rows(p1, p2)
        jsd = js_divergence_rows(p1, p2)
        cosine = cosine_rows(p1, p2)
        min_valid = np.minimum(m1["valid_fraction"].to_numpy(float), m2["valid_fraction"].to_numpy(float))
        pair_df = position_table.copy()
        pair_df.insert(0, "pair", f"{s1}v{s2}")
        pair_df["serotype_a"] = s1
        pair_df["serotype_b"] = s2
        pair_df["distribution_overlap"] = overlap
        pair_df["jensen_shannon_divergence"] = jsd
        pair_df["cosine_similarity"] = cosine
        pair_df["min_valid_fraction"] = min_valid
        pair_df["coverage_weighted_overlap"] = overlap * min_valid
        pair_df["gap_fraction_absdiff"] = np.abs(m1["gap_fraction"].to_numpy(float) - m2["gap_fraction"].to_numpy(float))
        pair_frames.append(pair_df)

    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()

    global_metrics = metrics[metrics["serotype"] == "GLOBAL"].reset_index(drop=True)
    if not pairs.empty:
        pair_group = pairs.groupby("msa_pos", sort=True)
        pair_jsd_mean = pair_group["jensen_shannon_divergence"].mean().to_numpy()
        pair_jsd_max = pair_group["jensen_shannon_divergence"].max().to_numpy()
        pair_overlap_mean = pair_group["distribution_overlap"].mean().to_numpy()
    else:
        pair_jsd_mean = np.full(msa_length, np.nan)
        pair_jsd_max = np.full(msa_length, np.nan)
        pair_overlap_mean = np.full(msa_length, np.nan)

    serotype_metric_only = metrics[metrics["serotype"].isin(serotypes)]
    entropy_by_pos = serotype_metric_only.pivot(index="msa_pos", columns="serotype", values="entropy_norm")
    intra_by_pos = serotype_metric_only.pivot(index="msa_pos", columns="serotype", values="intra_similarity")
    gap_by_pos = serotype_metric_only.pivot(index="msa_pos", columns="serotype", values="gap_fraction")

    aggregate = position_table.copy()
    aggregate["global_entropy_norm"] = global_metrics["entropy_norm"].to_numpy(float)
    aggregate["global_conservation_norm"] = global_metrics["conservation_norm"].to_numpy(float)
    aggregate["global_gap_fraction"] = global_metrics["gap_fraction"].to_numpy(float)
    aggregate["serotype_entropy_mean"] = entropy_by_pos.mean(axis=1).to_numpy(float)
    aggregate["serotype_entropy_max"] = entropy_by_pos.max(axis=1).to_numpy(float)
    aggregate["serotype_entropy_min"] = entropy_by_pos.min(axis=1).to_numpy(float)
    aggregate["serotype_entropy_range"] = (entropy_by_pos.max(axis=1) - entropy_by_pos.min(axis=1)).to_numpy(float)
    aggregate["intra_similarity_mean"] = intra_by_pos.mean(axis=1).to_numpy(float)
    aggregate["gap_fraction_mean"] = gap_by_pos.mean(axis=1).to_numpy(float)
    aggregate["pairwise_jsd_mean"] = pair_jsd_mean
    aggregate["pairwise_jsd_max"] = pair_jsd_max
    aggregate["pairwise_overlap_mean"] = pair_overlap_mean
    aggregate["serotype_specificity_score"] = pair_jsd_mean * (1.0 - aggregate["global_gap_fraction"].to_numpy(float))

    summary_serotype = (
        metrics[metrics["serotype"].isin(serotypes)]
        .groupby("serotype", as_index=False)
        .agg(
            n_sequences=("n_sequences", "max"),
            mean_entropy_norm=("entropy_norm", "mean"),
            median_entropy_norm=("entropy_norm", "median"),
            mean_conservation_norm=("conservation_norm", "mean"),
            mean_intra_similarity=("intra_similarity", "mean"),
            mean_gap_fraction=("gap_fraction", "mean"),
        )
        .sort_values("serotype", key=lambda s: s.map(natural_serotype_key))
    )

    if not pairs.empty:
        summary_pair = (
            pairs.groupby(["pair", "serotype_a", "serotype_b"], as_index=False)
            .agg(
                mean_distribution_overlap=("distribution_overlap", "mean"),
                median_distribution_overlap=("distribution_overlap", "median"),
                mean_jsd=("jensen_shannon_divergence", "mean"),
                median_jsd=("jensen_shannon_divergence", "median"),
                mean_cosine=("cosine_similarity", "mean"),
                mean_coverage_weighted_overlap=("coverage_weighted_overlap", "mean"),
                mean_gap_fraction_absdiff=("gap_fraction_absdiff", "mean"),
            )
            .sort_values("pair")
        )
    else:
        summary_pair = pd.DataFrame()

    return AnalysisBundle(
        records=list(records),
        serotypes=serotypes,
        alphabet=tuple(alphabet),
        msa_length=msa_length,
        position_table=position_table,
        metrics=metrics,
        pairs=pairs,
        aggregate=aggregate,
        summary_serotype=summary_serotype,
        summary_pair=summary_pair,
        distributions=distributions,
    )


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA")


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_entropy_lines(bundle: AnalysisBundle, out_base: Path, window: int, x_coord: str, dpi: int) -> None:
    x, valid, xlabel = x_coordinate(bundle.position_table, x_coord)
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    for serotype in bundle.serotypes:
        sub = bundle.metrics[bundle.metrics["serotype"] == serotype].sort_values("msa_pos")
        y = rolling_mean(sub["entropy_norm"].to_numpy(float), window)
        ax.plot(x[valid], y[valid], linewidth=1.2, label=serotype)
    global_sub = bundle.metrics[bundle.metrics["serotype"] == "GLOBAL"].sort_values("msa_pos")
    y_global = rolling_mean(global_sub["entropy_norm"].to_numpy(float), window)
    ax.plot(x[valid], y_global[valid], linewidth=2.0, linestyle="--", label="GLOBAL")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Shannon entropy, normalized (rolling={window})")
    ax.set_title("MSA column entropy by serotype")
    ax.legend(ncol=min(5, len(bundle.serotypes) + 1), frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    save_figure(fig, out_base, dpi)


def _heatmap_matrix(
    df: pd.DataFrame,
    row_col: str,
    value_col: str,
    row_order: Sequence[str],
    valid: np.ndarray,
) -> np.ndarray:
    pivot = df.pivot(index=row_col, columns="msa_pos", values=value_col)
    pivot = pivot.reindex(row_order)
    mat = pivot.to_numpy(dtype=float)
    return mat[:, valid]


def plot_serotype_heatmap(
    bundle: AnalysisBundle,
    metric: str,
    out_base: Path,
    x_coord: str,
    dpi: int,
    title: str,
    cbar_label: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    x, valid, xlabel = x_coordinate(bundle.position_table, x_coord)
    df = bundle.metrics[bundle.metrics["serotype"].isin(bundle.serotypes)]
    mat = _heatmap_matrix(df, "serotype", metric, bundle.serotypes, valid)
    fig_height = max(2.4, 0.45 * len(bundle.serotypes) + 1.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    extent = [np.nanmin(x[valid]), np.nanmax(x[valid]), len(bundle.serotypes) - 0.5, -0.5]
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_yticks(range(len(bundle.serotypes)))
    ax.set_yticklabels(bundle.serotypes)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label)
    save_figure(fig, out_base, dpi)


def plot_pair_heatmap(
    bundle: AnalysisBundle,
    metric: str,
    out_base: Path,
    x_coord: str,
    dpi: int,
    title: str,
    cbar_label: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "magma",
) -> None:
    if bundle.pairs.empty:
        return
    x, valid, xlabel = x_coordinate(bundle.position_table, x_coord)
    row_order = sorted(bundle.pairs["pair"].unique())
    mat = _heatmap_matrix(bundle.pairs, "pair", metric, row_order, valid)
    fig_height = max(3.0, 0.42 * len(row_order) + 1.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    extent = [np.nanmin(x[valid]), np.nanmax(x[valid]), len(row_order) - 0.5, -0.5]
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label)
    save_figure(fig, out_base, dpi)


def minmax_normalize(y: Sequence[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return out
    lo = np.nanmin(arr[valid])
    hi = np.nanmax(arr[valid])
    if hi <= lo:
        out[valid] = 0.0
    else:
        out[valid] = (arr[valid] - lo) / (hi - lo)
    return out


def plot_paper_panel(
    bundle: AnalysisBundle,
    out_base: Path,
    window: int,
    x_coord: str,
    dpi: int,
    attention: Optional[pd.DataFrame] = None,
    attention_label: str = "attention",
) -> None:
    x, valid, xlabel = x_coordinate(bundle.position_table, x_coord)
    agg = bundle.aggregate.sort_values("msa_pos").reset_index(drop=True)
    metrics_sero = bundle.metrics[bundle.metrics["serotype"].isin(bundle.serotypes)]
    entropy_mat = _heatmap_matrix(metrics_sero, "serotype", "entropy_norm", bundle.serotypes, valid)
    intra_mat = _heatmap_matrix(metrics_sero, "serotype", "intra_similarity", bundle.serotypes, valid)

    if bundle.pairs.empty:
        pair_order: List[str] = []
        jsd_mat = np.empty((0, valid.sum()))
    else:
        pair_order = sorted(bundle.pairs["pair"].unique())
        jsd_mat = _heatmap_matrix(bundle.pairs, "pair", "jensen_shannon_divergence", pair_order, valid)

    fig_height = 9.8 if attention is None else 10.4
    fig = plt.figure(figsize=(14.0, fig_height))
    gs = GridSpec(
        nrows=4,
        ncols=1,
        height_ratios=[1.35, 1.15, max(1.2, 0.23 * max(1, len(pair_order)) + 0.8), 1.15],
        hspace=0.42,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    spec = rolling_mean(agg["serotype_specificity_score"].to_numpy(float), window)
    ent = rolling_mean(agg["global_entropy_norm"].to_numpy(float), window)
    sim = rolling_mean(agg["intra_similarity_mean"].to_numpy(float), window)
    ax0.plot(x[valid], minmax_normalize(spec)[valid], linewidth=1.5, label="serotype specificity (mean JSD, norm.)")
    ax0.plot(x[valid], minmax_normalize(ent)[valid], linewidth=1.5, linestyle="--", label="global entropy (norm.)")
    ax0.plot(x[valid], minmax_normalize(sim)[valid], linewidth=1.2, linestyle=":", label="intra-serotype similarity (norm.)")
    if attention is not None and not attention.empty:
        att = attention.sort_values("msa_pos").set_index("msa_pos").reindex(agg["msa_pos"])
        att_y = rolling_mean(att["attention_value"].to_numpy(float), window)
        ax0.plot(x[valid], minmax_normalize(att_y)[valid], linewidth=1.8, alpha=0.85, label=f"{attention_label} (norm.)")
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_xlim(np.nanmin(x[valid]), np.nanmax(x[valid]))
    ax0.set_ylabel("normalized track")
    ax0.set_title("A. Aggregate MSA variability tracks")
    ax0.grid(alpha=0.2)
    ax0.legend(ncol=2, fontsize=8, frameon=False, loc="upper right")

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    extent1 = [np.nanmin(x[valid]), np.nanmax(x[valid]), len(bundle.serotypes) - 0.5, -0.5]
    im1 = ax1.imshow(entropy_mat, aspect="auto", interpolation="nearest", extent=extent1, vmin=0, vmax=1, cmap="viridis")
    ax1.set_yticks(range(len(bundle.serotypes)))
    ax1.set_yticklabels(bundle.serotypes)
    ax1.set_ylabel("serotype")
    ax1.set_title("B. Normalized Shannon entropy by serotype")
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.01)
    cbar1.set_label("entropy")

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    if jsd_mat.size:
        extent2 = [np.nanmin(x[valid]), np.nanmax(x[valid]), len(pair_order) - 0.5, -0.5]
        im2 = ax2.imshow(jsd_mat, aspect="auto", interpolation="nearest", extent=extent2, vmin=0, vmax=1, cmap="magma")
        ax2.set_yticks(range(len(pair_order)))
        ax2.set_yticklabels(pair_order)
        cbar2 = fig.colorbar(im2, ax=ax2, pad=0.01)
        cbar2.set_label("JSD")
    else:
        ax2.text(0.5, 0.5, "No serotype pairs available", transform=ax2.transAxes, ha="center", va="center")
        ax2.set_yticks([])
    ax2.set_ylabel("pair")
    ax2.set_title("C. Inter-serotype distribution divergence")

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    im3 = ax3.imshow(intra_mat, aspect="auto", interpolation="nearest", extent=extent1, vmin=0, vmax=1, cmap="cividis")
    ax3.set_yticks(range(len(bundle.serotypes)))
    ax3.set_yticklabels(bundle.serotypes)
    ax3.set_ylabel("serotype")
    ax3.set_xlabel(xlabel)
    ax3.set_title("D. Intra-serotype residue similarity")
    cbar3 = fig.colorbar(im3, ax=ax3, pad=0.01)
    cbar3.set_label("similarity")

    for ax in [ax0, ax1, ax2]:
        plt.setp(ax.get_xticklabels(), visible=False)
    save_figure(fig, out_base, dpi)


def rankdata_average(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    ranks = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return ranks
    xv = x[valid]
    order = np.argsort(xv, kind="mergesort")
    sorted_x = xv[order]
    sorted_ranks = np.empty_like(sorted_x, dtype=float)
    i = 0
    n = len(sorted_x)
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        # 1-based average rank.
        sorted_ranks[i:j] = 0.5 * (i + 1 + j)
        i = j
    rv = np.empty_like(sorted_ranks)
    rv[order] = sorted_ranks
    ranks[valid] = rv
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    av = a[valid] - np.mean(a[valid])
    bv = b[valid] - np.mean(b[valid])
    den = math.sqrt(float(np.sum(av * av) * np.sum(bv * bv)))
    if den == 0:
        return float("nan")
    return float(np.sum(av * bv) / den)


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    # Sniff a small sample when suffix is not informative.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    return pd.read_csv(path, sep=dialect.delimiter)


def choose_column(columns: Sequence[str], requested: Optional[str], candidates: Sequence[str], numeric_df: Optional[pd.DataFrame] = None) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested column {requested!r} not found. Available columns: {list(columns)}")
        return requested
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    for c in columns:
        lc = c.lower()
        if any(candidate in lc for candidate in candidates):
            return c
    if numeric_df is not None:
        numeric_cols = list(numeric_df.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            return numeric_cols[0]
    raise ValueError(f"Could not infer a column from available columns: {list(columns)}")


def load_attention(
    attention_table: Optional[Path],
    bundle: AnalysisBundle,
    position_col: Optional[str] = None,
    value_col: Optional[str] = None,
    coordinate: str = "msa",
    position_base: int = 1,
) -> Tuple[Optional[pd.DataFrame], str]:
    if attention_table is None:
        return None, "attention"
    df = read_table_auto(attention_table)
    pos_col = choose_column(df.columns, position_col, POSITION_CANDIDATES, df)
    val_col = choose_column(df.columns, value_col, ATTENTION_VALUE_CANDIDATES, df.drop(columns=[pos_col], errors="ignore"))
    tmp = df[[pos_col, val_col]].copy()
    tmp.columns = ["input_pos", "attention_value"]
    tmp["input_pos"] = pd.to_numeric(tmp["input_pos"], errors="coerce")
    tmp["attention_value"] = pd.to_numeric(tmp["attention_value"], errors="coerce")
    tmp = tmp.dropna(subset=["input_pos", "attention_value"])
    tmp["input_pos"] = tmp["input_pos"].round().astype(int)
    if position_base == 0:
        tmp["input_pos"] = tmp["input_pos"] + 1

    if coordinate == "msa":
        tmp = tmp.rename(columns={"input_pos": "msa_pos"})
    elif coordinate == "ref":
        mapping = bundle.position_table.dropna(subset=["ref_pos"])[["msa_pos", "ref_pos"]].copy()
        mapping["ref_pos"] = mapping["ref_pos"].astype(int)
        tmp = tmp.rename(columns={"input_pos": "ref_pos"}).merge(mapping, on="ref_pos", how="inner")
    else:
        raise ValueError("attention coordinate must be msa or ref")

    tmp = tmp.groupby("msa_pos", as_index=False)["attention_value"].mean()
    label = Path(attention_table).stem
    return tmp, label


def save_attention_comparison(bundle: AnalysisBundle, attention: Optional[pd.DataFrame], out_dir: Path, dpi: int, label: str) -> None:
    if attention is None or attention.empty:
        return
    joined = bundle.aggregate.merge(attention, on="msa_pos", how="inner")
    if joined.empty:
        return
    metrics = [
        "global_entropy_norm",
        "global_conservation_norm",
        "serotype_entropy_mean",
        "serotype_entropy_range",
        "intra_similarity_mean",
        "pairwise_jsd_mean",
        "pairwise_jsd_max",
        "pairwise_overlap_mean",
        "serotype_specificity_score",
        "global_gap_fraction",
    ]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "attention_label": label,
                "msa_metric": metric,
                "n_positions": int(joined[["attention_value", metric]].dropna().shape[0]),
                "pearson_r": pearson_corr(joined["attention_value"], joined[metric]),
                "spearman_rho": spearman_corr(joined["attention_value"], joined[metric]),
            }
        )
    save_table(pd.DataFrame(rows), out_dir / "attention_vs_msa_metric_correlations.tsv")
    save_table(joined, out_dir / "attention_vs_msa_position_join.tsv")

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    x = joined["serotype_specificity_score"].to_numpy(float)
    y = joined["attention_value"].to_numpy(float)
    ax.scatter(x, y, s=10, alpha=0.55)
    rho = spearman_corr(x, y)
    ax.set_xlabel("serotype specificity score")
    ax.set_ylabel(label)
    ax.set_title(f"Attention vs MSA serotype specificity (Spearman={rho:.3f})")
    ax.grid(alpha=0.2)
    save_figure(fig, out_dir / "attention_vs_serotype_specificity_scatter", dpi)


def consensus_string_for_position(bundle: AnalysisBundle, msa_pos: int) -> str:
    rows = bundle.metrics[(bundle.metrics["msa_pos"] == msa_pos) & (bundle.metrics["serotype"].isin(bundle.serotypes))]
    parts = []
    for _, row in rows.sort_values("serotype", key=lambda s: s.map(natural_serotype_key)).iterrows():
        symbol = row["consensus"] if isinstance(row["consensus"], str) and row["consensus"] else "?"
        parts.append(f"{row['serotype']}:{symbol}({row['consensus_fraction']:.2f})")
    return ";".join(parts)


def build_top_positions(bundle: AnalysisBundle, top_n: int) -> pd.DataFrame:
    agg = bundle.aggregate.copy()
    candidates = agg.sort_values("serotype_specificity_score", ascending=False).head(top_n).copy()
    candidates["rank_specificity"] = np.arange(1, len(candidates) + 1)
    candidates["consensus_by_serotype"] = [consensus_string_for_position(bundle, int(pos)) for pos in candidates["msa_pos"]]
    return candidates[
        [
            "rank_specificity",
            "msa_pos",
            "ref_pos",
            "serotype_specificity_score",
            "pairwise_jsd_mean",
            "pairwise_jsd_max",
            "serotype_entropy_mean",
            "serotype_entropy_range",
            "global_entropy_norm",
            "intra_similarity_mean",
            "global_gap_fraction",
            "consensus_by_serotype",
        ]
    ]


def build_top_pair_positions(bundle: AnalysisBundle, top_n: int) -> pd.DataFrame:
    if bundle.pairs.empty:
        return pd.DataFrame()
    rows = []
    for pair, sub in bundle.pairs.groupby("pair", sort=True):
        top = sub.sort_values("jensen_shannon_divergence", ascending=False).head(top_n).copy()
        top["rank_pair_jsd"] = np.arange(1, len(top) + 1)
        top["consensus_by_serotype"] = [consensus_string_for_position(bundle, int(pos)) for pos in top["msa_pos"]]
        rows.append(top)
    out = pd.concat(rows, ignore_index=True)
    columns = [
        "pair",
        "rank_pair_jsd",
        "msa_pos",
        "ref_pos",
        "serotype_a",
        "serotype_b",
        "jensen_shannon_divergence",
        "distribution_overlap",
        "cosine_similarity",
        "coverage_weighted_overlap",
        "gap_fraction_absdiff",
        "consensus_by_serotype",
    ]
    return out[columns]


def write_run_report(bundle: AnalysisBundle, out_dir: Path, args: argparse.Namespace, warnings_list: Sequence[str]) -> None:
    report = {
        "msa_length": int(bundle.msa_length),
        "n_records": int(len(bundle.records)),
        "serotypes": bundle.serotypes,
        "n_by_serotype": {s: int(sum(1 for r in bundle.records if r.serotype == s)) for s in bundle.serotypes},
        "alphabet": "".join(bundle.alphabet),
        "window": int(args.window),
        "x_coordinate": args.x_coordinate,
        "warnings": list(warnings_list),
        "input_paths": [str(p) for p in resolve_input_paths(args)],
    }
    with (out_dir / "run_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (out_dir / "README_outputs.txt").open("w", encoding="utf-8") as handle:
        handle.write("MSA serotype variability outputs\n")
        handle.write("================================\n\n")
        handle.write("Core tables:\n")
        handle.write("- per_position_serotype_metrics.tsv: entropy, gaps, consensus and intra-serotype similarity for each serotype and MSA column.\n")
        handle.write("- per_position_serotype_pairs.tsv: inter-serotype overlap, cosine similarity and Jensen-Shannon divergence per MSA column.\n")
        handle.write("- per_position_aggregate_metrics.tsv: global tracks and aggregate serotype-specificity score.\n")
        handle.write("- top_serotype_specific_positions.tsv: highest aggregate serotype-specific positions with consensus residues per serotype.\n")
        handle.write("- top_pairwise_divergent_positions.tsv: highest divergent positions for each pair such as DENV1vDENV2.\n\n")
        handle.write("Core figures:\n")
        handle.write("- entropy_by_serotype_lines.[png/pdf]\n")
        handle.write("- entropy_by_serotype_heatmap.[png/pdf]\n")
        handle.write("- inter_serotype_jsd_heatmap.[png/pdf]\n")
        handle.write("- intra_serotype_similarity_heatmap.[png/pdf]\n")
        handle.write("- paper_msa_variability_panel.[png/pdf]\n\n")
        if warnings_list:
            handle.write("Warnings:\n")
            for item in warnings_list:
                handle.write(f"- {item}\n")


def resolve_input_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.msa:
        paths.extend(Path(p) for p in args.msa)
    if args.msa_dir:
        paths.extend(find_fasta_files(Path(args.msa_dir)))
    return sorted(set(paths))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate serotype-aware entropy/similarity plots from a shared aligned FASTA MSA."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--msa", nargs="+", help="One or more aligned FASTA files to combine. Records must share the same aligned length.")
    input_group.add_argument("--msa-dir", help="Directory containing aligned FASTA files. Files are combined if all records share the same aligned length.")
    parser.add_argument("--out-dir", required=True, help="Output directory for tables and figures.")
    parser.add_argument("--serotype-regex", default=None, help="Optional regex to extract serotype from FASTA headers or filenames. Use one capture group for 1/2/3/4 or DENV1..DENV4.")
    parser.add_argument("--keep-unknown", action="store_true", help="Keep records whose serotype cannot be inferred as UNKNOWN. Default: drop them.")
    parser.add_argument("--alphabet", choices=["auto", "aa", "nt", "observed"], default="auto", help="Alphabet used for entropy. Default auto-detects nt vs aa.")
    parser.add_argument("--reference-regex", default=None, help="Regex identifying a reference sequence header for MSA-to-reference coordinate mapping.")
    parser.add_argument("--x-coordinate", choices=["msa", "ref"], default="msa", help="Position coordinate to use in figures. ref requires --reference-regex.")
    parser.add_argument("--window", type=int, default=21, help="Centered rolling window for line tracks. Use 1 for unsmoothed tracks.")
    parser.add_argument("--top-n", type=int, default=200, help="Number of top serotype-specific positions to export.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument("--attention-table", default=None, help="Optional attention aggregate CSV/TSV to compare/overlay.")
    parser.add_argument("--attention-position-col", default=None, help="Column in attention table containing positions.")
    parser.add_argument("--attention-value-col", default=None, help="Column in attention table containing attention/importance score.")
    parser.add_argument("--attention-coordinate", choices=["msa", "ref"], default="msa", help="Coordinate system used by --attention-table positions.")
    parser.add_argument("--attention-position-base", type=int, choices=[0, 1], default=1, help="Use 0 if attention-table positions are zero-based. Default: 1-based positions.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings_list: List[str] = []

    input_paths = resolve_input_paths(args)
    if not input_paths:
        raise ValueError("No FASTA files found. Check --msa or --msa-dir.")

    records = load_records(input_paths, serotype_regex=args.serotype_regex, drop_unknown=not args.keep_unknown)
    serotypes = sorted({r.serotype for r in records}, key=natural_serotype_key)
    if len(serotypes) < 2:
        warnings_list.append("Fewer than two serotypes were detected; pairwise inter-serotype plots will be skipped.")
    if "UNKNOWN" in serotypes:
        warnings_list.append("Some records were kept as UNKNOWN; interpret pairwise serotype metrics carefully.")

    alphabet = detect_alphabet(records, args.alphabet)
    bundle = analyze_msa(records, alphabet, reference_regex=args.reference_regex)

    save_table(bundle.position_table, out_dir / "msa_position_mapping.tsv")
    save_table(bundle.metrics, out_dir / "per_position_serotype_metrics.tsv")
    if not bundle.pairs.empty:
        save_table(bundle.pairs, out_dir / "per_position_serotype_pairs.tsv")
    save_table(bundle.aggregate, out_dir / "per_position_aggregate_metrics.tsv")
    save_table(bundle.summary_serotype, out_dir / "serotype_summary.tsv")
    if not bundle.summary_pair.empty:
        save_table(bundle.summary_pair, out_dir / "pair_summary.tsv")
    save_table(build_top_positions(bundle, args.top_n), out_dir / "top_serotype_specific_positions.tsv")
    top_pair_positions = build_top_pair_positions(bundle, args.top_n)
    if not top_pair_positions.empty:
        save_table(top_pair_positions, out_dir / "top_pairwise_divergent_positions.tsv")

    attention_df, attention_label = load_attention(
        Path(args.attention_table) if args.attention_table else None,
        bundle,
        position_col=args.attention_position_col,
        value_col=args.attention_value_col,
        coordinate=args.attention_coordinate,
        position_base=args.attention_position_base,
    )

    plot_entropy_lines(bundle, out_dir / "entropy_by_serotype_lines", args.window, args.x_coordinate, args.dpi)
    plot_serotype_heatmap(
        bundle,
        "entropy_norm",
        out_dir / "entropy_by_serotype_heatmap",
        args.x_coordinate,
        args.dpi,
        title="MSA normalized entropy by serotype",
        cbar_label="entropy",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plot_serotype_heatmap(
        bundle,
        "intra_similarity",
        out_dir / "intra_serotype_similarity_heatmap",
        args.x_coordinate,
        args.dpi,
        title="MSA intra-serotype residue similarity",
        cbar_label="similarity",
        vmin=0,
        vmax=1,
        cmap="cividis",
    )
    plot_serotype_heatmap(
        bundle,
        "gap_fraction",
        out_dir / "gap_fraction_by_serotype_heatmap",
        args.x_coordinate,
        args.dpi,
        title="MSA gap fraction by serotype",
        cbar_label="gap fraction",
        vmin=0,
        vmax=1,
        cmap="Greys",
    )
    plot_pair_heatmap(
        bundle,
        "jensen_shannon_divergence",
        out_dir / "inter_serotype_jsd_heatmap",
        args.x_coordinate,
        args.dpi,
        title="Inter-serotype Jensen-Shannon divergence by position",
        cbar_label="JSD",
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    plot_pair_heatmap(
        bundle,
        "distribution_overlap",
        out_dir / "inter_serotype_overlap_heatmap",
        args.x_coordinate,
        args.dpi,
        title="Inter-serotype residue-distribution overlap by position",
        cbar_label="overlap",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plot_paper_panel(
        bundle,
        out_dir / "paper_msa_variability_panel",
        args.window,
        args.x_coordinate,
        args.dpi,
        attention=attention_df,
        attention_label=attention_label,
    )
    save_attention_comparison(bundle, attention_df, out_dir, args.dpi, attention_label)
    write_run_report(bundle, out_dir, args, warnings_list)

    print(f"Wrote MSA serotype variability outputs to: {out_dir}")
    print(f"Detected serotypes: {', '.join(bundle.serotypes)}")
    print(f"MSA length: {bundle.msa_length}; records: {len(bundle.records)}; alphabet: {''.join(bundle.alphabet)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
