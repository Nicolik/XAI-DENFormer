#!/usr/bin/env python3
"""Run the fold-safe version of the original DENFormer metrics report.

This file is a drop-in replacement for classifier/workflow/reports/make_metrics.py.
It retrieves the previously committed original implementation from the local Git
repository, applies the fold-identity correction in memory, validates the patched
source, and executes it with the original command-line interface.

No repository file is modified at runtime. Fold names are always taken from the
``fold`` field stored in inference_summary.json rather than reconstructed from
list position.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_RELATIVE_PATH = "classifier/workflow/reports/make_metrics.py"
CURRENT_FILE = Path(__file__).resolve()


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _find_repo_root() -> Path:
    result = subprocess.run(
        ["git", "-C", str(CURRENT_FILE.parent), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "This replacement must be run inside the DENFormer Git repository. "
            f"Git error: {result.stderr.strip()}"
        )
    return Path(result.stdout.strip()).resolve()


def _looks_like_reviewed_original(source: str) -> bool:
    required = (
        'TEMPORAL_FOLD_ORDER = ["< 2006"',
        'CONTINENT_FOLD_ORDER = ["Africa"',
        "def extract_metric_values(summary_json):",
        "def build_fold_values_df(metric_values_by_model, models):",
        "def display_fold_label(experiment_name, fold_index):",
        'metric_values = extract_metric_values(summary_json)',
    )
    return all(marker in source for marker in required)


def _load_original_from_git(repo_root: Path) -> tuple[str, str]:
    # Before the replacement is staged, the Git index still contains the original.
    candidates: list[tuple[str, list[str]]] = [
        ("Git index", ["show", f":{REPO_RELATIVE_PATH}"]),
        ("HEAD", ["show", f"HEAD:{REPO_RELATIVE_PATH}"]),
        ("HEAD parent", ["show", f"HEAD^:{REPO_RELATIVE_PATH}"]),
        ("origin/master", ["show", f"origin/master:{REPO_RELATIVE_PATH}"]),
        ("origin/main", ["show", f"origin/main:{REPO_RELATIVE_PATH}"]),
    ]

    seen_commands: set[tuple[str, ...]] = set()
    for label, command in candidates:
        key = tuple(command)
        if key in seen_commands:
            continue
        seen_commands.add(key)
        result = _git(repo_root, *command)
        if result.returncode == 0 and _looks_like_reviewed_original(result.stdout):
            return result.stdout, label

    # Search the complete local history of this path. This also works after the
    # replacement has been committed, provided the clone contains its parent history.
    history = _git(repo_root, "rev-list", "--all", "--", REPO_RELATIVE_PATH)
    if history.returncode == 0:
        for commit in history.stdout.splitlines():
            result = _git(repo_root, "show", f"{commit}:{REPO_RELATIVE_PATH}")
            if result.returncode == 0 and _looks_like_reviewed_original(result.stdout):
                return result.stdout, f"commit {commit[:12]}"

    raise RuntimeError(
        "Could not locate the reviewed original make_metrics.py in the Git index "
        "or local history. Restore the repository version of the file, then extract "
        "this replacement again before running it."
    )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot apply {label}: expected one matching block, found {count}. "
            "The local repository version differs from the reviewed source."
        )
    return text.replace(old, new, 1)


def _apply_fold_identity_fix(source: str) -> str:
    patched = source

    patched = _replace_once(
        patched,
        '''TEMPORAL_FOLD_ORDER = ["< 2006", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]
CONTINENT_FOLD_ORDER = ["Africa", "Asia", "Europe", "Oceania", "North America", "South America"]
''',
        '''FOLD_IDS_KEY = "_fold_ids"
''',
        "removal of position-based fold-name lists",
    )

    patched = _replace_once(
        patched,
        '''def extract_metric_values(summary_json):
    values = {metric: [] for metric in METRICS}

    for fold in summary_json:
        test = fold["test"]
        report = test["classification_report"]

        if "Accuracy" in values:
            values["Accuracy"].append(test["accuracy"])

        for metric_name, report_label in SEROTYPE_LABELS.items():
            if metric_name not in values:
                continue

            if report_label not in report:
                # Some evaluation splits may not contain all serotypes in a fold/test set.
                # Keep the explicit numeric mapping fixed, but store NaN for missing
                # serotypes so summaries and statistical tests can ignore them safely.
                values[metric_name].append(np.nan)
                continue

            values[metric_name].append(report[report_label]["f1-score"])

        # To re-enable aggregate metrics, uncomment the corresponding entries
        # in METRICS and uncomment the block below.
        #
        # macro = report["macro avg"]
        # weighted = report["weighted avg"]
        #
        # values["macro_precision"].append(macro["precision"])
        # values["macro_recall"].append(macro["recall"])
        # values["macro_f1"].append(macro["f1-score"])
        #
        # values["weighted_precision"].append(weighted["precision"])
        # values["weighted_recall"].append(weighted["recall"])
        # values["weighted_f1"].append(weighted["f1-score"])

    return values
''',
        '''def normalize_fold_id(fold_id):
    if fold_id is None:
        raise ValueError("Missing fold identifier in inference_summary.json")
    value = str(fold_id).strip()
    if not value:
        raise ValueError("Empty fold identifier in inference_summary.json")
    return value


def fold_sort_key(experiment_name, fold_id):
    """Sort real fold identifiers without inferring their identity by position."""
    value = normalize_fold_id(fold_id)
    compact = value.replace(" ", "")

    if experiment_name == "ohe_timebin_e100":
        if compact.startswith("<"):
            try:
                return (0, int(compact[1:]), value.casefold())
            except ValueError:
                return (0, -1, value.casefold())
        try:
            start_year = int(compact.split("-", 1)[0])
            return (1, start_year, value.casefold())
        except ValueError:
            return (2, value.casefold())

    if experiment_name == "ohe_cdhit_e100":
        numeric = compact
        for prefix in ("fold_", "Fold_", "fold", "Fold"):
            if numeric.startswith(prefix):
                numeric = numeric[len(prefix):]
                break
        numeric = numeric.lstrip("_- ")
        if numeric.isdigit():
            return (0, int(numeric))

    return (0, value.casefold())


def display_fold_label(experiment_name, fold_id):
    """Format the real fold identifier for manuscript-ready outputs."""
    value = normalize_fold_id(fold_id)
    compact = value.replace(" ", "")

    if experiment_name == "ohe_timebin_e100" and compact.startswith("<"):
        return f"< {compact[1:]}"

    if experiment_name == "ohe_cdhit_e100":
        numeric = compact
        for prefix in ("fold_", "Fold_", "fold", "Fold"):
            if numeric.startswith(prefix):
                numeric = numeric[len(prefix):]
                break
        numeric = numeric.lstrip("_- ")
        if numeric.isdigit():
            return f"Fold {int(numeric)}"

    return value


def validate_metric_fold_lengths(metric_values, context):
    fold_ids = metric_values.get(FOLD_IDS_KEY, [])
    expected = len(fold_ids)
    if len(set(fold_ids)) != expected:
        raise ValueError(f"Duplicate fold identifiers in {context}: {fold_ids}")

    for metric in METRICS:
        actual = len(metric_values.get(metric, []))
        if actual != expected:
            raise ValueError(
                f"Fold/metric length mismatch in {context}: "
                f"{expected} fold identifiers but {actual} values for {metric}."
            )


def validate_model_fold_alignment(metric_values_by_model, models, context):
    reference_model = None
    reference_ids = None
    for model in models:
        metric_values = metric_values_by_model.get(model)
        if metric_values is None:
            continue
        validate_metric_fold_lengths(metric_values, f"{context} / {model}")
        fold_ids = tuple(metric_values[FOLD_IDS_KEY])
        if reference_ids is None:
            reference_model = model
            reference_ids = fold_ids
        elif fold_ids != reference_ids:
            raise ValueError(
                f"Fold alignment mismatch in {context}: {model} has {fold_ids}, "
                f"whereas {reference_model} has {reference_ids}."
            )


def extract_metric_values(summary_json, experiment_name):
    values = {metric: [] for metric in METRICS}
    values[FOLD_IDS_KEY] = []

    ordered_folds = sorted(
        summary_json,
        key=lambda fold: fold_sort_key(experiment_name, fold.get("fold")),
    )

    for fold in ordered_folds:
        fold_id = normalize_fold_id(fold.get("fold"))
        values[FOLD_IDS_KEY].append(fold_id)

        test = fold["test"]
        report = test["classification_report"]

        if "Accuracy" in values:
            values["Accuracy"].append(test["accuracy"])

        for metric_name, report_label in SEROTYPE_LABELS.items():
            if metric_name not in values:
                continue

            if report_label not in report:
                values[metric_name].append(np.nan)
                continue

            values[metric_name].append(report[report_label]["f1-score"])

    validate_metric_fold_lengths(values, f"{experiment_name} inference summary")
    return values
''',
        "fold-aware metric extraction",
    )

    patched = _replace_once(
        patched,
        '''def build_fold_values_df(metric_values_by_model, models):
    rows = []

    for model in models:
        metric_values = metric_values_by_model.get(model)
        if metric_values is None:
            continue

        max_len = max((len(metric_values.get(metric, [])) for metric in METRICS), default=0)
        for fold_idx in range(max_len):
            for metric in METRICS:
                values = metric_values.get(metric, [])
                value = values[fold_idx] if fold_idx < len(values) else np.nan
                rows.append({
                    "model": model,
                    "model_display": display_model_name(model),
                    "fold_index": fold_idx + 1,
                    "metric": metric,
                    "metric_display": display_metric_name(metric),
                    "value": float(value) if np.isfinite(value) else np.nan,
                })

    return pd.DataFrame(rows)




def display_fold_label(experiment_name, fold_index):
    """Return a paper-friendly fold label when the split strategy has semantic folds."""
    try:
        idx = int(fold_index)
    except (TypeError, ValueError):
        return str(fold_index)

    if experiment_name == "ohe_continent_e100" and 1 <= idx <= len(CONTINENT_FOLD_ORDER):
        return CONTINENT_FOLD_ORDER[idx - 1]

    if experiment_name == "ohe_timebin_e100" and 1 <= idx <= len(TEMPORAL_FOLD_ORDER):
        return TEMPORAL_FOLD_ORDER[idx - 1]

    return f"Fold {idx}"


def add_fold_labels(fold_df, experiment_name):
    if fold_df.empty:
        return fold_df.copy()
    out = fold_df.copy()
    out["fold_label"] = out["fold_index"].apply(lambda idx: display_fold_label(experiment_name, idx))
    return out


def build_fold_metric_wide_df(fold_df, experiment_name):
    """Paper-friendly wide table: one row per model/fold and one column per metric."""
    if fold_df.empty:
        return pd.DataFrame()

    labeled = add_fold_labels(fold_df, experiment_name)
    wide = labeled.pivot_table(
        index=["model", "model_display", "fold_index", "fold_label"],
        columns="metric_display",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.sort_values(["model", "fold_index"])
    return wide
''',
        '''def build_fold_values_df(metric_values_by_model, models, experiment_name):
    rows = []
    validate_model_fold_alignment(
        metric_values_by_model,
        models,
        f"{experiment_name} fold-level report",
    )

    for model in models:
        metric_values = metric_values_by_model.get(model)
        if metric_values is None:
            continue

        fold_ids = metric_values[FOLD_IDS_KEY]
        for fold_idx, fold_id in enumerate(fold_ids, start=1):
            for metric in METRICS:
                value = metric_values[metric][fold_idx - 1]
                rows.append({
                    "model": model,
                    "model_display": display_model_name(model),
                    "fold_index": fold_idx,
                    "fold_id": fold_id,
                    "fold_label": display_fold_label(experiment_name, fold_id),
                    "metric": metric,
                    "metric_display": display_metric_name(metric),
                    "value": float(value) if np.isfinite(value) else np.nan,
                })

    return pd.DataFrame(rows)


def add_fold_labels(fold_df, experiment_name):
    if fold_df.empty:
        return fold_df.copy()
    if "fold_id" not in fold_df.columns:
        raise ValueError("fold_id is missing from fold-level reporting data")
    out = fold_df.copy()
    out["fold_label"] = out["fold_id"].apply(
        lambda fold_id: display_fold_label(experiment_name, fold_id)
    )
    return out


def build_fold_metric_wide_df(fold_df, experiment_name):
    """Paper-friendly wide table: one row per model/fold and one column per metric."""
    if fold_df.empty:
        return pd.DataFrame()

    labeled = add_fold_labels(fold_df, experiment_name)
    wide = labeled.pivot_table(
        index=["model", "model_display", "fold_index", "fold_id", "fold_label"],
        columns="metric_display",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.sort_values(["model", "fold_index"])
    return wide
''',
        "fold-aware long and wide tables",
    )

    patched = _replace_once(
        patched,
        '''    fold_indices = sorted(
        int(fold_idx)
        for fold_idx in fold_df["fold_index"].dropna().unique()
    )
    column_labels = [f"Split {fold_idx}" for fold_idx in fold_indices]
''',
        '''    fold_metadata = (
        fold_df[["fold_index", "fold_label"]]
        .drop_duplicates()
        .sort_values("fold_index")
    )
    fold_indices = fold_metadata["fold_index"].astype(int).tolist()
    column_labels = fold_metadata["fold_label"].astype(str).tolist()
''',
        "semantic heatmap labels",
    )

    patched = _replace_once(
        patched,
        '            title = display_fold_label(experiment_name, fold_index)\n',
        '            title = str(one_fold.iloc[0]["fold_label"])\n',
        "fold-grid title",
    )

    patched = _replace_once(
        patched,
        '''    global_metric_values = {
        model: {metric: [] for metric in METRICS}
        for model in MODELS
    }
''',
        '''    global_metric_values = {
        model: {
            FOLD_IDS_KEY: [],
            **{metric: [] for metric in METRICS},
        }
        for model in MODELS
    }
''',
        "global fold identifiers",
    )

    patched = _replace_once(
        patched,
        '            metric_values = extract_metric_values(summary_json)\n',
        '            metric_values = extract_metric_values(summary_json, experiment)\n',
        "experiment-aware extraction",
    )

    patched = _replace_once(
        patched,
        '''            for metric in METRICS:
                global_metric_values[model][metric].extend(metric_values[metric])
''',
        '''            global_metric_values[model][FOLD_IDS_KEY].extend(
                f"{experiment}::{fold_id}"
                for fold_id in metric_values[FOLD_IDS_KEY]
            )
            for metric in METRICS:
                global_metric_values[model][metric].extend(metric_values[metric])
''',
        "globally unique fold identifiers",
    )

    patched = _replace_once(
        patched,
        '                "fold_values": build_fold_values_df(metric_values, models),\n',
        '                "fold_values": build_fold_values_df(metric_values, models, experiment),\n',
        "fold-aware table construction",
    )

    patched = _replace_once(
        patched,
        '''def compute_statistical_tests(metric_values_by_model, experiment_name, group_name, models):
    present_models = [
''',
        '''def compute_statistical_tests(metric_values_by_model, experiment_name, group_name, models):
    validate_model_fold_alignment(
        metric_values_by_model,
        models,
        f"{experiment_name} / {group_name} statistical comparison",
    )
    present_models = [
''',
        "statistical fold alignment",
    )

    patched = _replace_once(
        patched,
        '        "split_strategy", "fold_index", "fold",\n        "Acc", "DENV1", "DENV2", "DENV3", "DENV4",\n',
        '        "split_strategy", "fold_index", "fold",\n        "Accuracy", "DENV1", "DENV2", "DENV3", "DENV4",\n',
        "numeric result-table accuracy column",
    )

    return patched


def main() -> None:
    repo_root = _find_repo_root()
    original, source_label = _load_original_from_git(repo_root)
    patched = _apply_fold_identity_fix(original)
    compile(patched, str(CURRENT_FILE), "exec")

    print(f"Fold-safe reporting code loaded from {source_label}.")
    print("Fold identities will be read from inference_summary.json.")

    runtime_globals = {
        "__name__": "__main__",
        "__file__": str(CURRENT_FILE),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(patched, str(CURRENT_FILE), "exec"), runtime_globals)


if __name__ == "__main__":
    main()
