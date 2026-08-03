"""Prediction-level statistical comparisons for DENFormer performance results.

The module deliberately avoids inferential tests on the small number of outer folds.
Instead, it aligns out-of-fold predictions for the same genomes and performs paired
comparisons while using CD-HIT clusters as the resampling/permutation unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import ast
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest



MODEL_DISPLAY_NAMES = {
    "denformer_first": "DENFormer-first",
    "denformer_mean": "DENFormer-mean",
    "denformer_max": "DENFormer-max",
    "longformer": "Longformer",
    "performer": "Performer",
    "ffnn": "FFNN",
    "logreg": "Logistic Regression",
}

PROTOCOL_DISPLAY_NAMES = {
    "ohe_cdhit_e100": "CD-HIT cluster-aware",
    "ohe_continent_e100": "Geographical",
    "ohe_timebin_e100": "Temporal",
}

DEFAULT_PROTOCOLS = tuple(PROTOCOL_DISPLAY_NAMES)
DEFAULT_MODELS = tuple(MODEL_DISPLAY_NAMES)

BENCHMARK_PAIRS = (
    ("denformer_mean", "longformer"),
    ("denformer_mean", "performer"),
    ("denformer_mean", "ffnn"),
    ("denformer_mean", "logreg"),
)

ABLATION_PAIRS = (
    ("denformer_mean", "denformer_first"),
    ("denformer_mean", "denformer_max"),
    ("denformer_first", "denformer_max"),
)


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_reps: int = 10_000
    permutation_reps: int = 100_000
    seed: int = 42
    confidence_level: float = 0.95
    bootstrap_batch_size: int = 128
    permutation_batch_size: int = 512


@dataclass
class AnalysisOutputs:
    primary_accuracy: pd.DataFrame
    macro_f1: pd.DataFrame
    per_class_f1: pd.DataFrame
    model_summary: pd.DataFrame
    qc_alignment: pd.DataFrame
    parameters: pd.DataFrame


def display_model_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def display_protocol_name(protocol: str) -> str:
    return PROTOCOL_DISPLAY_NAMES.get(protocol, protocol)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise error correction with NaN preservation."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if valid_idx.size == 0:
        return adjusted

    order = valid_idx[np.argsort(values[valid_idx])]
    m = order.size
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _normalize_serotype(value: object) -> str:
    text = str(value).strip().upper().replace("DENV-", "").replace("DENV", "")
    try:
        numeric = int(float(text))
    except (TypeError, ValueError):
        return str(value).strip()
    return str(numeric)


def load_cluster_map(cdhit_split_file: Path | str) -> pd.DataFrame:
    """Load one CD-HIT cluster identifier per global dataset index.

    The CD-HIT split CSV repeats every index once for each outer fold. The mapping
    itself must be invariant across those repeated rows.
    """
    path = Path(cdhit_split_file)
    if not path.exists():
        raise FileNotFoundError(f"CD-HIT split file not found: {path}")

    df = pd.read_csv(path)
    required = {"index", "cluster_id", "serotype"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CD-HIT split file is missing required columns {sorted(missing)}: {path}"
        )

    work = df[["index", "cluster_id", "serotype"]].copy()
    work["index"] = work["index"].astype(int)
    work["serotype_norm"] = work["serotype"].map(_normalize_serotype)
    work["cluster_key"] = (
        "DENV" + work["serotype_norm"].astype(str)
        + "::cluster_" + work["cluster_id"].astype(str)
    )

    conflicts = (
        work.groupby("index")["cluster_key"].nunique().loc[lambda s: s > 1]
    )
    if not conflicts.empty:
        preview = conflicts.head(10).to_dict()
        raise ValueError(f"Inconsistent CD-HIT cluster mapping for indices: {preview}")

    mapping = (
        work.sort_values("index")
        .drop_duplicates("index")[["index", "cluster_key", "serotype_norm"]]
        .reset_index(drop=True)
    )
    return mapping


def _prediction_files(metrics_dir: Path) -> list[Path]:
    return sorted(metrics_dir.glob("split_*/predictions_test.npz"))


def _safe_fold_name(value: object) -> str:
    text = re.sub(r'[<>:"/\\|?*]', '_', str(value))
    return text.replace(' ', '_')


def _parse_index_cell(value: object) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, float) and value.is_integer():
        return [int(value)]
    text = str(value).strip()
    if not text:
        return []
    if text[0] in '[(':
        return [int(x) for x in ast.literal_eval(text)]
    for sep in [',', ';', ' ']:
        if sep in text:
            normalized = text.replace(';', ',').replace(' ', ',')
            return [int(x) for x in normalized.split(',') if x]
    return [int(text)]


def _test_indices_by_fold(split_file: Path | str | None) -> dict[str, np.ndarray]:
    """Return test indices keyed by the sanitized fold directory suffix.

    Supports the long CSV split format used by this project
    (``index, split, fold``) and the wide format containing ``test_idx``.
    """
    if split_file is None:
        return {}
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    fold_col = lower.get('fold') or lower.get('fold_id') or lower.get('cv_fold')
    split_col = lower.get('split') or lower.get('set') or lower.get('partition')
    index_col = (
        lower.get('index') or lower.get('idx') or
        lower.get('sample_idx') or lower.get('sample_index')
    )

    lookup: dict[str, np.ndarray] = {}
    if split_col and index_col:
        work = df.copy()
        work['_split_norm'] = work[split_col].astype(str).str.strip().str.lower()
        work = work[work['_split_norm'].isin({'test', 'te'})]
        work['_fold_norm'] = work[fold_col].astype(str) if fold_col else 'holdout'
        for fold_id, fold_df in work.groupby('_fold_norm', sort=False):
            values: list[int] = []
            for value in fold_df[index_col]:
                values.extend(_parse_index_cell(value))
            lookup[_safe_fold_name(fold_id)] = np.asarray(values, dtype=np.int64)
        return lookup

    test_col = lower.get('test_idx') or lower.get('test_indices')
    if test_col:
        for pos, row in df.iterrows():
            fold_id = str(row[fold_col]) if fold_col else ('holdout' if len(df) == 1 else str(pos))
            lookup[_safe_fold_name(fold_id)] = np.asarray(
                _parse_index_cell(row[test_col]), dtype=np.int64
            )
        return lookup

    raise ValueError(
        f"Unsupported split file format for reconstructing test indices: {path}. "
        "Expected long columns (index, split[, fold]) or a test_idx column."
    )


def load_oof_predictions(
    logs_dir: Path | str,
    model: str,
    protocol: str,
    split_file: Path | str | None = None,
) -> pd.DataFrame:
    """Load and concatenate held-out predictions across all partitions.

    Current inference outputs store the global dataset indices directly. Legacy
    prediction files may contain only labels/predictions/probabilities. For those
    files, indices are reconstructed from the corresponding precomputed split.
    This is valid because inference uses the test indices in their stored order
    with ``shuffle=False``.
    """
    metrics_dir = Path(logs_dir) / model / protocol / "metrics"
    files = _prediction_files(metrics_dir)
    if not files:
        raise FileNotFoundError(
            f"No predictions_test.npz files found for model={model}, "
            f"protocol={protocol} under {metrics_dir}"
        )

    fold_test_indices = _test_indices_by_fold(split_file)
    frames: list[pd.DataFrame] = []
    for path in files:
        fold = path.parent.name.removeprefix("split_")
        with np.load(path) as data:
            required = {"labels", "preds"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            labels = np.asarray(data["labels"], dtype=np.int64)
            preds = np.asarray(data["preds"], dtype=np.int64)
            if "indices" in data.files:
                indices = np.asarray(data["indices"], dtype=np.int64)
                index_source = "prediction_file"
            else:
                if not fold_test_indices:
                    raise ValueError(
                        f"{path} has no 'indices' array and no split file was supplied "
                        f"for protocol={protocol}."
                    )
                if fold not in fold_test_indices:
                    available = sorted(fold_test_indices)
                    raise ValueError(
                        f"Cannot reconstruct indices for {path}: fold {fold!r} was not "
                        f"found in {split_file}. Available folds: {available}"
                    )
                indices = fold_test_indices[fold]
                index_source = "split_file_fallback"

        if not (len(indices) == len(labels) == len(preds)):
            raise ValueError(
                f"Length mismatch in {path}: indices={len(indices)}, "
                f"labels={len(labels)}, preds={len(preds)}. "
                f"Index source: {index_source}."
            )

        frames.append(
            pd.DataFrame(
                {
                    "index": indices,
                    "y_true": labels,
                    "y_pred": preds,
                    "fold": fold,
                    "index_source": index_source,
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    if out["index"].duplicated().any():
        duplicates = out.loc[out["index"].duplicated(keep=False), "index"].head(20).tolist()
        raise ValueError(
            f"Duplicate held-out indices for model={model}, protocol={protocol}: {duplicates}"
        )
    return out.sort_values("index").reset_index(drop=True)


def align_pair_predictions(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    cluster_map: pd.DataFrame,
    model_a: str,
    model_b: str,
    protocol: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Strictly align two models on the same held-out genomes."""
    indices_a = set(pred_a["index"].astype(int))
    indices_b = set(pred_b["index"].astype(int))
    only_a = sorted(indices_a - indices_b)
    only_b = sorted(indices_b - indices_a)
    if only_a or only_b:
        raise ValueError(
            f"Prediction index mismatch for {protocol}: {model_a} vs {model_b}; "
            f"only_a={only_a[:10]}, only_b={only_b[:10]}"
        )

    merged = pred_a.merge(
        pred_b,
        on="index",
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    label_mismatch = int((merged["y_true_a"] != merged["y_true_b"]).sum())
    fold_mismatch = int((merged["fold_a"] != merged["fold_b"]).sum())
    if label_mismatch:
        raise ValueError(
            f"Ground-truth mismatch for {protocol}: {model_a} vs {model_b} "
            f"({label_mismatch} rows)"
        )
    if fold_mismatch:
        raise ValueError(
            f"Held-out partition mismatch for {protocol}: {model_a} vs {model_b} "
            f"({fold_mismatch} rows)"
        )

    merged = merged.merge(cluster_map, on="index", how="left", validate="many_to_one")
    missing_clusters = int(merged["cluster_key"].isna().sum())
    if missing_clusters:
        missing_idx = merged.loc[merged["cluster_key"].isna(), "index"].head(20).tolist()
        raise ValueError(
            f"Missing CD-HIT cluster mapping for {missing_clusters} held-out genomes: {missing_idx}"
        )

    out = pd.DataFrame(
        {
            "index": merged["index"].astype(int),
            "fold": merged["fold_a"].astype(str),
            "cluster_key": merged["cluster_key"].astype(str),
            "y_true": merged["y_true_a"].astype(int),
            "pred_a": merged["y_pred_a"].astype(int),
            "pred_b": merged["y_pred_b"].astype(int),
        }
    )
    qc = {
        "protocol": protocol,
        "protocol_display": display_protocol_name(protocol),
        "model_a": model_a,
        "model_b": model_b,
        "n_model_a": len(pred_a),
        "n_model_b": len(pred_b),
        "n_aligned": len(out),
        "n_unique_indices": int(out["index"].nunique()),
        "n_unique_clusters": int(out["cluster_key"].nunique()),
        "n_partitions": int(out["fold"].nunique()),
        "index_source_model_a": ", ".join(sorted(pred_a["index_source"].astype(str).unique())),
        "index_source_model_b": ", ".join(sorted(pred_b["index_source"].astype(str).unique())),
        "label_mismatches": label_mismatch,
        "fold_mismatches": fold_mismatch,
        "missing_cluster_ids": missing_clusters,
        "alignment_status": "PASS",
    }
    return out, qc


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> tuple[float, float, np.ndarray]:
    total = float(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else np.nan
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    denominator = 2.0 * tp + fp + fn
    per_class = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator > 0,
    )
    macro_f1 = float(np.mean(per_class)) if per_class.size else np.nan
    return accuracy, macro_f1, per_class


def _metrics_from_confusion_batch(cms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    totals = cms.sum(axis=(1, 2)).astype(float)
    traces = np.trace(cms, axis1=1, axis2=2).astype(float)
    accuracy = np.divide(traces, totals, out=np.full_like(traces, np.nan), where=totals > 0)

    tp = np.diagonal(cms, axis1=1, axis2=2).astype(float)
    fp = cms.sum(axis=1).astype(float) - tp
    fn = cms.sum(axis=2).astype(float) - tp
    denominator = 2.0 * tp + fp + fn
    per_class = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator > 0,
    )
    macro_f1 = per_class.mean(axis=1)
    return accuracy, macro_f1, per_class


def exact_mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float | int]:
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(min(a_only, b_only), n=discordant, p=0.5, alternative="two-sided").pvalue
        )
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant_total": discordant,
        "mcnemar_exact_p": p_value,
    }


def _cluster_confusion_contributions(
    aligned: pd.DataFrame,
    n_classes: int,
) -> tuple[list[str], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    cluster_names: list[str] = []
    cluster_labels: list[int] = []
    cm_a_rows: list[np.ndarray] = []
    cm_b_rows: list[np.ndarray] = []

    for cluster_key, group in aligned.groupby("cluster_key", sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        labels = np.unique(y_true)
        if labels.size != 1:
            raise ValueError(f"CD-HIT cluster {cluster_key} contains multiple true labels: {labels}")
        cluster_names.append(str(cluster_key))
        cluster_labels.append(int(labels[0]))
        cm_a_rows.append(
            _confusion_matrix(y_true, group["pred_a"].to_numpy(dtype=int), n_classes).reshape(-1)
        )
        cm_b_rows.append(
            _confusion_matrix(y_true, group["pred_b"].to_numpy(dtype=int), n_classes).reshape(-1)
        )

    cm_a = np.vstack(cm_a_rows).astype(np.int64)
    cm_b = np.vstack(cm_b_rows).astype(np.int64)
    strata: dict[int, np.ndarray] = {}
    cluster_labels_array = np.asarray(cluster_labels, dtype=int)
    for label in sorted(np.unique(cluster_labels_array)):
        strata[int(label)] = np.flatnonzero(cluster_labels_array == label)
    return cluster_names, strata, cm_a, cm_b


def cluster_bootstrap_metrics(
    aligned: pd.DataFrame,
    config: AnalysisConfig,
    n_classes: int,
    seed_offset: int = 0,
) -> dict[str, object]:
    """Paired cluster bootstrap, stratified by true serotype.

    Whole CD-HIT clusters are sampled with replacement. All genomes belonging to a
    sampled cluster are retained, preserving within-cluster dependence and the
    paired predictions from both models.
    """
    _, strata, cm_a_flat, cm_b_flat = _cluster_confusion_contributions(aligned, n_classes)
    reps = int(config.bootstrap_reps)
    if reps < 1:
        raise ValueError("bootstrap_reps must be positive")

    rng = np.random.default_rng(config.seed + seed_offset)
    delta_accuracy = np.empty(reps, dtype=float)
    delta_macro_f1 = np.empty(reps, dtype=float)
    delta_per_class = np.empty((reps, n_classes), dtype=float)
    accuracy_a = np.empty(reps, dtype=float)
    accuracy_b = np.empty(reps, dtype=float)

    batch_size = max(1, int(config.bootstrap_batch_size))
    cursor = 0
    while cursor < reps:
        current = min(batch_size, reps - cursor)
        boot_a = np.zeros((current, n_classes * n_classes), dtype=np.int64)
        boot_b = np.zeros_like(boot_a)

        for idx in strata.values():
            k = len(idx)
            if k == 0:
                continue
            weights = rng.multinomial(
                n=k,
                pvals=np.full(k, 1.0 / k, dtype=float),
                size=current,
            )
            boot_a += weights @ cm_a_flat[idx]
            boot_b += weights @ cm_b_flat[idx]

        cms_a = boot_a.reshape(current, n_classes, n_classes)
        cms_b = boot_b.reshape(current, n_classes, n_classes)
        acc_a, macro_a, class_a = _metrics_from_confusion_batch(cms_a)
        acc_b, macro_b, class_b = _metrics_from_confusion_batch(cms_b)

        sl = slice(cursor, cursor + current)
        accuracy_a[sl] = acc_a
        accuracy_b[sl] = acc_b
        delta_accuracy[sl] = acc_a - acc_b
        delta_macro_f1[sl] = macro_a - macro_b
        delta_per_class[sl] = class_a - class_b
        cursor += current

    alpha = 1.0 - config.confidence_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    def interval(values: np.ndarray) -> tuple[float, float]:
        low, high = np.nanpercentile(values, [lower_q, upper_q])
        return float(low), float(high)

    result: dict[str, object] = {
        "bootstrap_reps": reps,
        "bootstrap_confidence_level": config.confidence_level,
        "delta_accuracy_ci": interval(delta_accuracy),
        "delta_macro_f1_ci": interval(delta_macro_f1),
        "delta_per_class_ci": [interval(delta_per_class[:, i]) for i in range(n_classes)],
        "accuracy_a_ci": interval(accuracy_a),
        "accuracy_b_ci": interval(accuracy_b),
        "delta_accuracy_bootstrap_median": float(np.nanmedian(delta_accuracy)),
        "delta_macro_f1_bootstrap_median": float(np.nanmedian(delta_macro_f1)),
    }
    return result


def cluster_sign_flip_permutation(
    aligned: pd.DataFrame,
    config: AnalysisConfig,
    seed_offset: int = 0,
) -> dict[str, float | int | str]:
    """Cluster-level paired sign-flip test for the accuracy difference."""
    cluster_diff = (
        aligned.assign(
            correct_a=(aligned["pred_a"] == aligned["y_true"]).astype(int),
            correct_b=(aligned["pred_b"] == aligned["y_true"]).astype(int),
        )
        .assign(correct_diff=lambda df: df["correct_a"] - df["correct_b"])
        .groupby("cluster_key", sort=True)["correct_diff"]
        .sum()
        .to_numpy(dtype=np.int64)
    )
    observed_sum = int(cluster_diff.sum())
    observed_abs = abs(observed_sum)
    n_clusters = int(cluster_diff.size)

    if observed_abs == 0:
        return {
            "cluster_permutation_p": 1.0,
            "cluster_permutation_reps": 0,
            "cluster_permutation_method": "degenerate_zero_difference",
            "cluster_accuracy_difference_count": observed_sum,
        }

    if n_clusters <= 20:
        n_exact = 1 << n_clusters
        exceed = 0
        for mask in range(n_exact):
            signs = np.ones(n_clusters, dtype=np.int8)
            bits = ((mask >> np.arange(n_clusters)) & 1).astype(bool)
            signs[bits] = -1
            value = abs(int(np.dot(signs, cluster_diff)))
            exceed += int(value >= observed_abs)
        p_value = exceed / n_exact
        return {
            "cluster_permutation_p": float(p_value),
            "cluster_permutation_reps": int(n_exact),
            "cluster_permutation_method": "exact_cluster_sign_flip",
            "cluster_accuracy_difference_count": observed_sum,
        }

    reps = int(config.permutation_reps)
    if reps < 1:
        raise ValueError("permutation_reps must be positive")
    rng = np.random.default_rng(config.seed + seed_offset)
    batch_size = max(1, int(config.permutation_batch_size))
    exceed = 0
    completed = 0
    while completed < reps:
        current = min(batch_size, reps - completed)
        signs = rng.integers(0, 2, size=(current, n_clusters), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs(signs @ cluster_diff)
        exceed += int(np.sum(permuted >= observed_abs))
        completed += current

    p_value = (exceed + 1.0) / (reps + 1.0)
    return {
        "cluster_permutation_p": float(p_value),
        "cluster_permutation_reps": reps,
        "cluster_permutation_method": "monte_carlo_cluster_sign_flip",
        "cluster_accuracy_difference_count": observed_sum,
    }


def _observed_metrics(aligned: pd.DataFrame, n_classes: int) -> dict[str, object]:
    y_true = aligned["y_true"].to_numpy(dtype=int)
    pred_a = aligned["pred_a"].to_numpy(dtype=int)
    pred_b = aligned["pred_b"].to_numpy(dtype=int)
    cm_a = _confusion_matrix(y_true, pred_a, n_classes)
    cm_b = _confusion_matrix(y_true, pred_b, n_classes)
    acc_a, macro_a, class_a = metrics_from_confusion(cm_a)
    acc_b, macro_b, class_b = metrics_from_confusion(cm_b)
    return {
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "delta_accuracy": acc_a - acc_b,
        "macro_f1_a": macro_a,
        "macro_f1_b": macro_b,
        "delta_macro_f1": macro_a - macro_b,
        "per_class_f1_a": class_a,
        "per_class_f1_b": class_b,
        "delta_per_class_f1": class_a - class_b,
    }


def compare_pair(
    aligned: pd.DataFrame,
    protocol: str,
    family: str,
    model_a: str,
    model_b: str,
    config: AnalysisConfig,
    n_classes: int,
    seed_offset: int,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    observed = _observed_metrics(aligned, n_classes)
    correct_a = aligned["pred_a"].to_numpy(dtype=int) == aligned["y_true"].to_numpy(dtype=int)
    correct_b = aligned["pred_b"].to_numpy(dtype=int) == aligned["y_true"].to_numpy(dtype=int)
    mcnemar = exact_mcnemar(correct_a, correct_b)
    bootstrap = cluster_bootstrap_metrics(
        aligned,
        config=config,
        n_classes=n_classes,
        seed_offset=seed_offset,
    )
    permutation = cluster_sign_flip_permutation(
        aligned,
        config=config,
        seed_offset=10_000 + seed_offset,
    )

    acc_ci_low, acc_ci_high = bootstrap["delta_accuracy_ci"]
    primary = {
        "family": family,
        "protocol": protocol,
        "protocol_display": display_protocol_name(protocol),
        "model_a": model_a,
        "model_a_display": display_model_name(model_a),
        "model_b": model_b,
        "model_b_display": display_model_name(model_b),
        "comparison": f"{display_model_name(model_a)} vs {display_model_name(model_b)}",
        "n_genomes": int(len(aligned)),
        "n_cdhit_clusters": int(aligned["cluster_key"].nunique()),
        "n_partitions": int(aligned["fold"].nunique()),
        "accuracy_a": observed["accuracy_a"],
        "accuracy_b": observed["accuracy_b"],
        "delta_accuracy": observed["delta_accuracy"],
        "delta_accuracy_pp": 100.0 * observed["delta_accuracy"],
        "delta_accuracy_ci_low": acc_ci_low,
        "delta_accuracy_ci_high": acc_ci_high,
        "delta_accuracy_ci_low_pp": 100.0 * acc_ci_low,
        "delta_accuracy_ci_high_pp": 100.0 * acc_ci_high,
        "accuracy_a_ci_low": bootstrap["accuracy_a_ci"][0],
        "accuracy_a_ci_high": bootstrap["accuracy_a_ci"][1],
        "accuracy_b_ci_low": bootstrap["accuracy_b_ci"][0],
        "accuracy_b_ci_high": bootstrap["accuracy_b_ci"][1],
        **mcnemar,
        **permutation,
        "bootstrap_reps": config.bootstrap_reps,
        "confidence_level": config.confidence_level,
    }

    macro_ci_low, macro_ci_high = bootstrap["delta_macro_f1_ci"]
    macro = {
        "family": family,
        "protocol": protocol,
        "protocol_display": display_protocol_name(protocol),
        "model_a": model_a,
        "model_a_display": display_model_name(model_a),
        "model_b": model_b,
        "model_b_display": display_model_name(model_b),
        "comparison": primary["comparison"],
        "n_genomes": primary["n_genomes"],
        "n_cdhit_clusters": primary["n_cdhit_clusters"],
        "macro_f1_a": observed["macro_f1_a"],
        "macro_f1_b": observed["macro_f1_b"],
        "delta_macro_f1": observed["delta_macro_f1"],
        "delta_macro_f1_pp": 100.0 * observed["delta_macro_f1"],
        "delta_macro_f1_ci_low": macro_ci_low,
        "delta_macro_f1_ci_high": macro_ci_high,
        "delta_macro_f1_ci_low_pp": 100.0 * macro_ci_low,
        "delta_macro_f1_ci_high_pp": 100.0 * macro_ci_high,
        "bootstrap_reps": config.bootstrap_reps,
        "confidence_level": config.confidence_level,
    }

    per_class: list[dict[str, object]] = []
    class_a = np.asarray(observed["per_class_f1_a"], dtype=float)
    class_b = np.asarray(observed["per_class_f1_b"], dtype=float)
    class_delta = np.asarray(observed["delta_per_class_f1"], dtype=float)
    class_ci = bootstrap["delta_per_class_ci"]
    for class_idx in range(n_classes):
        low, high = class_ci[class_idx]
        per_class.append(
            {
                "family": family,
                "protocol": protocol,
                "protocol_display": display_protocol_name(protocol),
                "model_a": model_a,
                "model_a_display": display_model_name(model_a),
                "model_b": model_b,
                "model_b_display": display_model_name(model_b),
                "comparison": primary["comparison"],
                "class_index": class_idx,
                "serotype": f"DENV{class_idx + 1}",
                "f1_a": class_a[class_idx],
                "f1_b": class_b[class_idx],
                "delta_f1": class_delta[class_idx],
                "delta_f1_pp": 100.0 * class_delta[class_idx],
                "delta_f1_ci_low": low,
                "delta_f1_ci_high": high,
                "delta_f1_ci_low_pp": 100.0 * low,
                "delta_f1_ci_high_pp": 100.0 * high,
                "bootstrap_reps": config.bootstrap_reps,
                "confidence_level": config.confidence_level,
            }
        )

    return primary, macro, per_class


def _interpret_primary(row: pd.Series) -> str:
    delta = float(row["delta_accuracy"])
    low = float(row["delta_accuracy_ci_low"])
    high = float(row["delta_accuracy_ci_high"])
    p = float(row["cluster_permutation_p_holm"])
    if p < 0.05 and low > 0:
        return "Model A higher under this protocol"
    if p < 0.05 and high < 0:
        return "Model B higher under this protocol"
    if math.isclose(delta, 0.0, abs_tol=1e-15) and low <= 0 <= high:
        return "No observed accuracy difference"
    return "No statistically supported accuracy difference"


def _build_model_summary(
    predictions: Mapping[tuple[str, str], pd.DataFrame],
    cluster_map: pd.DataFrame,
    protocols: Sequence[str],
    models: Sequence[str],
    n_classes: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol in protocols:
        for model in models:
            pred = predictions[(model, protocol)].merge(
                cluster_map[["index", "cluster_key"]],
                on="index",
                how="left",
                validate="one_to_one",
            )
            if pred["cluster_key"].isna().any():
                raise ValueError(f"Missing clusters in model summary: {model}, {protocol}")
            cm = _confusion_matrix(
                pred["y_true"].to_numpy(dtype=int),
                pred["y_pred"].to_numpy(dtype=int),
                n_classes,
            )
            accuracy, macro_f1, per_class = metrics_from_confusion(cm)
            row: dict[str, object] = {
                "protocol": protocol,
                "protocol_display": display_protocol_name(protocol),
                "model": model,
                "model_display": display_model_name(model),
                "n_genomes": len(pred),
                "n_cdhit_clusters": int(pred["cluster_key"].nunique()),
                "n_partitions": int(pred["fold"].nunique()),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
            for class_idx, value in enumerate(per_class):
                row[f"f1_denv{class_idx + 1}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(
    logs_dir: Path | str,
    cdhit_split_file: Path | str,
    config: AnalysisConfig = AnalysisConfig(),
    protocols: Sequence[str] = DEFAULT_PROTOCOLS,
    models: Sequence[str] = DEFAULT_MODELS,
    n_classes: int = 4,
    protocol_split_files: Mapping[str, Path | str] | None = None,
) -> AnalysisOutputs:
    cluster_map = load_cluster_map(cdhit_split_file)
    split_files = dict(protocol_split_files or {})
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for protocol in protocols:
        for model in models:
            predictions[(model, protocol)] = load_oof_predictions(
                logs_dir, model, protocol, split_file=split_files.get(protocol)
            )

    primary_rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    planned = [
        ("benchmark", pair) for pair in BENCHMARK_PAIRS
    ] + [
        ("ablation", pair) for pair in ABLATION_PAIRS
    ]

    seed_offset = 0
    for protocol in protocols:
        for family, (model_a, model_b) in planned:
            aligned, qc = align_pair_predictions(
                predictions[(model_a, protocol)],
                predictions[(model_b, protocol)],
                cluster_map,
                model_a=model_a,
                model_b=model_b,
                protocol=protocol,
            )
            qc["family"] = family
            qc_rows.append(qc)
            primary, macro, per_class = compare_pair(
                aligned=aligned,
                protocol=protocol,
                family=family,
                model_a=model_a,
                model_b=model_b,
                config=config,
                n_classes=n_classes,
                seed_offset=seed_offset,
            )
            primary_rows.append(primary)
            macro_rows.append(macro)
            class_rows.extend(per_class)
            seed_offset += 1

    primary_df = pd.DataFrame(primary_rows)
    for family in primary_df["family"].unique():
        mask = primary_df["family"].eq(family)
        primary_df.loc[mask, "cluster_permutation_p_holm"] = holm_adjust(
            primary_df.loc[mask, "cluster_permutation_p"].to_numpy(dtype=float)
        )
        primary_df.loc[mask, "mcnemar_exact_p_holm"] = holm_adjust(
            primary_df.loc[mask, "mcnemar_exact_p"].to_numpy(dtype=float)
        )
    primary_df["interpretation"] = primary_df.apply(_interpret_primary, axis=1)
    primary_df["ci_excludes_zero"] = (
        (primary_df["delta_accuracy_ci_low"] > 0)
        | (primary_df["delta_accuracy_ci_high"] < 0)
    )

    model_summary = _build_model_summary(
        predictions=predictions,
        cluster_map=cluster_map,
        protocols=protocols,
        models=models,
        n_classes=n_classes,
    )

    parameters = pd.DataFrame(
        [
            {"parameter": "analysis_unit", "value": "out-of-fold held-out genome predictions"},
            {"parameter": "pairing_key", "value": "global dataset index"},
            {"parameter": "legacy_index_recovery", "value": "when absent from prediction files, indices are reconstructed from ordered test_idx values in the corresponding split CSV"},
            {"parameter": "dependence_unit", "value": "serotype-specific CD-HIT cluster"},
            {"parameter": "primary_endpoint", "value": "accuracy difference (model A minus model B)"},
            {"parameter": "primary_test", "value": "cluster-level paired sign-flip permutation"},
            {"parameter": "sensitivity_test", "value": "exact two-sided McNemar"},
            {"parameter": "uncertainty", "value": "serotype-stratified paired CD-HIT cluster bootstrap"},
            {"parameter": "multiple_testing", "value": "Holm correction within benchmark and ablation families"},
            {"parameter": "bootstrap_reps", "value": config.bootstrap_reps},
            {"parameter": "permutation_reps", "value": config.permutation_reps},
            {"parameter": "confidence_level", "value": config.confidence_level},
            {"parameter": "random_seed", "value": config.seed},
            {"parameter": "training_seed_scope", "value": "single fixed training seed; no seed-to-seed inference"},
        ]
    )

    sort_cols = ["family", "protocol", "model_a", "model_b"]
    primary_df = primary_df.sort_values(sort_cols).reset_index(drop=True)
    macro_df = pd.DataFrame(macro_rows).sort_values(sort_cols).reset_index(drop=True)
    per_class_df = pd.DataFrame(class_rows).sort_values(sort_cols + ["class_index"]).reset_index(drop=True)
    qc_df = pd.DataFrame(qc_rows).sort_values(sort_cols).reset_index(drop=True)

    return AnalysisOutputs(
        primary_accuracy=primary_df,
        macro_f1=macro_df,
        per_class_f1=per_class_df,
        model_summary=model_summary,
        qc_alignment=qc_df,
        parameters=parameters,
    )
