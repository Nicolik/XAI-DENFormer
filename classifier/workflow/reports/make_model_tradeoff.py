try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import math
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import paths
import classifier.workflow.config as config
from classifier.workflow.utils import build_classifier_model


MODELS = [
    {"name": "denformer_first", "model_type": "denformer", "pooling": "first"},
    {"name": "denformer_max", "model_type": "denformer", "pooling": "max"},
    {"name": "denformer_mean", "model_type": "denformer", "pooling": "mean"},
    {"name": "longformer", "model_type": "longformer", "pooling": "mean"},
    {"name": "performer", "model_type": "performer", "pooling": "mean"},
    {"name": "ffnn", "model_type": "ffnn", "pooling": "mean"},
    {"name": "logreg", "model_type": "logreg", "pooling": "mean"},
]

DENFORMER_VARIANTS = [
    "denformer_first",
    "denformer_max",
    "denformer_mean",
]

SELECTED_DENFORMER = "denformer_mean"

OTHER_MODEL_NAMES = [
    model_info["name"] for model_info in MODELS
    if model_info["name"] not in DENFORMER_VARIANTS
]

COMPARISON_GROUPS = {
    "all_models": [model_info["name"] for model_info in MODELS],
    "denformer_variants": DENFORMER_VARIANTS,
    "selected_denformer_vs_others": [SELECTED_DENFORMER] + OTHER_MODEL_NAMES,
}

EXPERIMENTS = [
    "ohe_cdhit_e100",
    "ohe_continent_e100",
    "ohe_timebin_e100",
]

DISPLAY_NAMES = {
    "ffnn": "FFNN",
    "denformer_first": "DENFormer-first",
    "denformer_mean": "DENFormer-mean",
    "denformer_max": "DENFormer-max",
    "longformer": "Longformer",
    "performer": "Performer",
    "logreg": "LogisticRegression",
}

EXPERIMENT_TITLES = {
    "all_experiments": "Overall",
    "ohe_cdhit_e100": "CD-HIT cluster-aware folds",
    "ohe_continent_e100": "Geographical distribution by continent",
    "ohe_timebin_e100": "Temporal distribution",
}

MODEL_COLORS = {
    # Keep colors consistent with make_metrics.py.
    # DENFormer variants use lighter shades of the same blue family.
    "denformer_first": "#c6dbef",
    "denformer_max": "#9ecae1",
    "denformer_mean": "#6baed6",
    "longformer": "#ff7f0e",
    "performer": "#2ca02c",
    "ffnn": "#d62728",
    "logreg": "#9467bd",
}

RECAP_1X3_DIR_NAME = "recap_1x3"


def model_color(model_name):
    return MODEL_COLORS.get(model_name)


def build_model_color_map():
    fallback_colors = plt.cm.tab10(np.linspace(0, 1, max(len(MODELS), 1)))
    return {
        model_info["name"]: model_color(model_info["name"]) or fallback_colors[i]
        for i, model_info in enumerate(MODELS)
    }


def copy_figure_for_paper(source_path, destination_dir, destination_name=None):
    source_path = Path(source_path)
    if not source_path.exists():
        return None
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / (destination_name or source_path.name)
    shutil.copy2(source_path, destination_path)
    return destination_path


EXPERIMENT_PANEL_ORDER = [
    ("A", "all_experiments"),
    ("B", "ohe_cdhit_e100"),
    ("C", "ohe_continent_e100"),
    ("D", "ohe_timebin_e100"),
]

EXPERIMENT_ROW_PANEL_ORDER = [
    ("A", "ohe_cdhit_e100"),
    ("B", "ohe_continent_e100"),
    ("C", "ohe_timebin_e100"),
]

# Cross-split plots use the same marker shape for every point.
# The model is encoded by hue, while the evaluation protocol is encoded by
# a lighter/darker shade of that same model color. This avoids suggesting that
# different protocols correspond to different model architectures or weights.
EXPERIMENT_COLOR_LIGHTENING = {
    "ohe_cdhit_e100": 0.00,
    "ohe_continent_e100": 0.32,
    "ohe_timebin_e100": 0.58,
}

TITLE_SIZE = 22
AXIS_LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 15
LEGEND_TITLE_SIZE = 16

EMB_DIM_FOR_COMPLEXITY = config.EMB_DIM_OHE
MIN_BUBBLE_AREA = 180
MAX_BUBBLE_AREA = 3600
# Error bars use a darkened version of each model color, not a neutral black.
ERROR_BAR_ALPHA = 0.96
ERROR_BAR_LINEWIDTH = 1.75
ERROR_BAR_CAPSIZE = 4.2
ERROR_BAR_CAPTHICK = 1.60
LEGEND_MARKER_SIZE_MIN = 9.5
LEGEND_MARKER_SIZE_MAX = 30.0


ERROR_BAR_CHOICES = ("minmax", "std")
DEFAULT_ERROR_BAR_MODE = "std"

SUMMARY_CHOICES = ("mean", "median")
DEFAULT_SUMMARY_MODE = "mean"




def output_subdir_name(summary_mode, error_bar_mode):
    return f"{summary_mode}_{error_bar_mode}"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot model accuracy-efficiency trade-offs."
    )
    parser.add_argument(
        "--error-bar",
        choices=ERROR_BAR_CHOICES,
        default=DEFAULT_ERROR_BAR_MODE,
        help=(
            "Error-bar definition for plots. "
            "'minmax' draws error bars from fold minimum to fold maximum around the selected summary; "
            "'std' draws selected summary +/- standard deviation. Default: std."
        ),
    )
    parser.add_argument(
        "--summary",
        choices=SUMMARY_CHOICES,
        default=DEFAULT_SUMMARY_MODE,
        help=(
            "Central summary shown by each point. "
            "Use 'mean' with --error-bar std, or 'median' with --error-bar minmax. "
            "Default: mean."
        ),
    )
    return parser.parse_args()


TIMING_KEYS = [
    "total_time_sec",
    "samples_per_sec",
    "num_samples",
    "num_batches",
    "batch_size",
    "mean_batch_time_sec",
    "std_batch_time_sec",
]

DEVICE_KEYS = [
    "device",
    "gpu_name",
    "gpu_total_memory_gb",
    "gpu_compute_capability",
    "gpu_multiprocessor_count",
    "cuda_available",
    "cuda_version",
    "cudnn_version",
    "torch_version",
    "python_version",
    "platform",
]


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


def summary_column(prefix, summary_mode):
    return f"{prefix}_{summary_mode}"


def error_interval(mean, std, min_value, max_value, error_bar_mode):
    if error_bar_mode == "std":
        return mean - std, mean + std
    if error_bar_mode == "minmax":
        return min_value, max_value
    raise ValueError(f"Unsupported error_bar_mode: {error_bar_mode}")


def asymmetric_error(mean, lower, upper):
    lower_err = max(0.0, mean - lower)
    upper_err = max(0.0, upper - mean)
    return np.array([[lower_err], [upper_err]], dtype=float)



def get_tradeoff_bounds(df, error_bar_mode, summary_mode):
    """Return x/y summary values and lower/upper bounds used by trade-off plots."""
    x_col = summary_column("test_total_time_sec", summary_mode)
    y_col = summary_column("accuracy", summary_mode)

    df = df.dropna(
        subset=[x_col, "trainable_params", y_col]
    ).copy()

    if df.empty:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float) * 100.0

    if error_bar_mode == "std":
        x_lower = x - df["test_total_time_sec_std"].to_numpy(dtype=float)
        x_upper = x + df["test_total_time_sec_std"].to_numpy(dtype=float)
        y_lower = y - df["accuracy_std"].to_numpy(dtype=float) * 100.0
        y_upper = y + df["accuracy_std"].to_numpy(dtype=float) * 100.0
    elif error_bar_mode == "minmax":
        x_lower = df["test_total_time_sec_min"].to_numpy(dtype=float)
        x_upper = df["test_total_time_sec_max"].to_numpy(dtype=float)
        y_lower = df["accuracy_min"].to_numpy(dtype=float) * 100.0
        y_upper = df["accuracy_max"].to_numpy(dtype=float) * 100.0
    else:
        raise ValueError(f"Unsupported error_bar_mode: {error_bar_mode}")

    x_lower = np.nan_to_num(x_lower, nan=x)
    x_upper = np.nan_to_num(x_upper, nan=x)
    y_lower = np.nan_to_num(y_lower, nan=y)
    y_upper = np.nan_to_num(y_upper, nan=y)

    # Inference time cannot be negative. When using mean +/- std, folds with
    # very small runtimes can produce a negative lower bound; clamp that bound
    # to zero for both axis scaling and error-bar rendering.
    x_lower = np.maximum(x_lower, 0.0)

    return x, y, x_lower, x_upper, y_lower, y_upper

def compute_global_tradeoff_limits(plot_tables, error_bar_mode, summary_mode):
    """Compute common x/y axes and parameter scaling for all trade-off plots."""
    all_x_lower = []
    all_x_upper = []
    all_y_lower = []
    all_y_upper = []
    all_params = []

    for df in plot_tables:
        if df is None or df.empty:
            continue
        _, _, x_lower, x_upper, y_lower, y_upper = get_tradeoff_bounds(df, error_bar_mode, summary_mode)
        if x_lower.size == 0:
            continue
        all_x_lower.append(x_lower)
        all_x_upper.append(x_upper)
        all_y_lower.append(y_lower)
        all_y_upper.append(y_upper)
        params = df["trainable_params"].to_numpy(dtype=float) if "trainable_params" in df else np.array([])
        params = params[np.isfinite(params)]
        if params.size:
            all_params.append(params)

    if not all_x_lower:
        return None, None

    x_lower = np.concatenate(all_x_lower)
    x_upper = np.concatenate(all_x_upper)
    y_lower = np.concatenate(all_y_lower)
    y_upper = np.concatenate(all_y_upper)

    valid_x = np.isfinite(x_lower) & np.isfinite(x_upper)
    valid_y = np.isfinite(y_lower) & np.isfinite(y_upper)
    x_lower = x_lower[valid_x]
    x_upper = x_upper[valid_x]
    y_lower = y_lower[valid_y]
    y_upper = y_upper[valid_y]

    x_min_raw = float(np.nanmin(x_lower))
    x_max_raw = float(np.nanmax(x_upper))
    y_min_raw = float(np.nanmin(y_lower))
    y_max_raw = float(np.nanmax(y_upper))

    if x_max_raw <= x_min_raw:
        x_min_raw -= 0.5
        x_max_raw += 0.5
    if y_max_raw <= y_min_raw:
        y_min_raw -= 0.5
        y_max_raw += 0.5

    x_range = x_max_raw - x_min_raw
    y_range = y_max_raw - y_min_raw

    axis_limits = {
        "x_min": x_min_raw - x_range * 0.10,
        "x_max": x_max_raw + x_range * 0.10,
        "y_min": max(0.0, y_min_raw - y_range * 0.10),
        "y_max": y_max_raw + y_range * 0.10,
    }

    if all_params:
        params = np.concatenate(all_params)
        param_limits = {
            "p_min": float(np.nanmin(params)),
            "p_max": float(np.nanmax(params)),
        }
    else:
        param_limits = None

    return axis_limits, param_limits


def _tradeoff_tick_step(span):
    """Choose a simple tick step for timing/accuracy axes."""
    span = float(span) if np.isfinite(span) else 1.0
    if span <= 1.0:
        return 0.2
    if span <= 2.5:
        return 0.5
    if span <= 6.0:
        return 1.0
    if span <= 16.0:
        return 2.0
    return 5.0


def apply_tradeoff_axis_ticks(ax):
    """Use physically meaningful ticks for trade-off plots.

    The x-axis is inference time, so tick labels must start at 0. The axis
    itself may extend slightly below 0 only as visual padding, so bubbles near
    zero are not clipped. Accuracy is bounded, so the y-axis may have headroom
    above 100% for error bars, but tick labels should not exceed 100%.
    """
    x_min, x_max = ax.get_xlim()

    x_step = _tradeoff_tick_step(x_max - max(0.0, x_min))
    x_start = math.ceil(max(0.0, x_min) / x_step) * x_step
    x_ticks = np.arange(x_start, x_max + x_step * 0.5, x_step)
    x_ticks = x_ticks[x_ticks >= -1e-9]
    if x_ticks.size:
        ax.set_xticks(x_ticks)

    y_min, y_max = ax.get_ylim()
    y_tick_min = max(0.0, math.ceil(y_min / 5.0) * 5.0)
    y_tick_max = min(100.0, math.floor(y_max / 5.0) * 5.0)
    if y_tick_max < 100.0 and y_max >= 100.0:
        y_tick_max = 100.0
    if y_tick_max >= y_tick_min:
        ax.set_yticks(np.arange(y_tick_min, y_tick_max + 1e-9, 5.0))


def format_params(value):
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def legend_marker_size_from_params(value, p_min, p_max):
    """Legend marker diameter in points, scaled monotonically from plotted bubble area.

    The plotted scatter size is an area in points^2. Using the square root
    preserves the same visual ordering in the legend while capping very large
    markers so that the all-model legend remains readable. Exact trainable
    parameter counts are shown in the label.
    """
    area = bubble_size(np.array([value], dtype=float), p_min, p_max)[0]
    diameter = math.sqrt(float(area)) * 0.50
    return float(np.clip(diameter, LEGEND_MARKER_SIZE_MIN, LEGEND_MARKER_SIZE_MAX))


def representative_params_by_model(df):
    params = {}
    if df is None or df.empty or "model" not in df or "trainable_params" not in df:
        return params
    for model, model_df in df.groupby("model"):
        values = pd.to_numeric(model_df["trainable_params"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size:
            params[str(model)] = float(np.nanmedian(values))
    return params


def model_param_label(model, params_by_model):
    label = DISPLAY_NAMES.get(model, model)
    value = params_by_model.get(model)
    if value is None or not np.isfinite(value):
        return label
    return f"{label} ({format_params(value)})"


def sorted_model_names_by_params(model_names, params_by_model):
    """Sort legend entries from the smallest to the largest model."""
    def sort_key(model):
        value = params_by_model.get(model)
        if value is None or not np.isfinite(value):
            return (1, float("inf"), DISPLAY_NAMES.get(model, model))
        return (0, float(value), DISPLAY_NAMES.get(model, model))

    return sorted(model_names, key=sort_key)


def build_model_param_handles(model_names, color_map, params_by_model, p_min, p_max):
    handles = []
    for model in sorted_model_names_by_params(model_names, params_by_model):
        value = params_by_model.get(model)
        marker_size = 8.0
        if value is not None and np.isfinite(value):
            marker_size = legend_marker_size_from_params(value, p_min, p_max)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color_map[model],
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=marker_size,
                label=model_param_label(model, params_by_model),
            )
        )
    return handles


def build_model(model_type, pooling):
    args = SimpleNamespace(model_type=model_type, pooling=pooling)
    device = torch.device("cpu")

    model = build_classifier_model(
        args=args,
        emb_dim=EMB_DIM_FOR_COMPLEXITY,
        config=config,
        device=device,
        attn=False,
    )
    model.eval()
    return model


def count_params(model):
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "state_dict_values": int(sum(t.numel() for t in model.state_dict().values())),
    }


def build_complexity_table():
    rows = []

    for model_info in MODELS:
        row = {
            "model": model_info["name"],
            "model_type": model_info["model_type"],
            "pooling": model_info["pooling"],
        }

        try:
            model = build_model(model_info["model_type"], model_info["pooling"])
            row.update(count_params(model))
            row["complexity_source"] = "model_parameters"
            del model
        except Exception as exc:
            print(f"Could not count parameters for {model_info['name']}: {exc}")
            row.update({
                "total_params": np.nan,
                "trainable_params": np.nan,
                "state_dict_values": np.nan,
                "complexity_source": "failed",
            })

        rows.append(row)

    return pd.DataFrame(rows)


def extract_fold_record(fold_result, model_name, experiment):
    if "test" not in fold_result:
        return None

    test_metrics = fold_result.get("test", {})
    test_timing = fold_result.get("test_timing", {})
    device_info = fold_result.get("device_info", {})

    row = {
        "model": model_name,
        "experiment": experiment,
        "fold": fold_result.get("fold"),
        "accuracy": test_metrics.get("accuracy"),
    }

    for key in TIMING_KEYS:
        row[f"test_{key}"] = test_timing.get(key, np.nan)

    for key in DEVICE_KEYS:
        row[key] = device_info.get(key, np.nan)

    if row["accuracy"] is None:
        return None

    return row


def load_fold_table(base_dir):
    rows = []

    for experiment in EXPERIMENTS:
        for model_info in MODELS:
            model_name = model_info["name"]
            summary_path = (
                base_dir
                / model_name
                / experiment
                / "metrics"
                / "inference_summary.json"
            )

            if not summary_path.exists():
                print(f"Missing metrics, model will not appear in {experiment}: {summary_path}")
                continue

            with open(summary_path, "r") as f:
                summary_json = json.load(f)

            for fold_result in summary_json:
                row = extract_fold_record(fold_result, model_name, experiment)
                if row is not None:
                    rows.append(row)

    return pd.DataFrame(rows)


def first_non_null(values):
    for value in values:
        if pd.notna(value):
            return value
    return np.nan


def summarize_experiment(fold_df, complexity_df, experiment, model_names=None):
    if experiment == "all_experiments":
        subset = fold_df.copy()
    else:
        subset = fold_df[fold_df["experiment"] == experiment].copy()

    if model_names is not None:
        subset = subset[subset["model"].isin(model_names)].copy()

    if subset.empty:
        return pd.DataFrame()

    rows = []
    selected_model_infos = [
        model_info for model_info in MODELS
        if model_names is None or model_info["name"] in model_names
    ]

    for model_info in selected_model_infos:
        model_name = model_info["name"]
        model_rows = subset[subset["model"] == model_name].copy()

        if model_rows.empty:
            continue

        valid_timing = model_rows.dropna(subset=["test_total_time_sec"])
        if valid_timing.empty:
            print(
                f"Timing missing for {model_name} in {experiment}. "
                "Run 02_run_inference.py again with the revised inference.py."
            )
            continue

        acc_mean, acc_median, acc_std, acc_min, acc_max = summarize_values(
            model_rows["accuracy"].dropna().to_numpy(dtype=float)
        )
        time_mean, time_median, time_std, time_min, time_max = summarize_values(
            valid_timing["test_total_time_sec"].to_numpy(dtype=float)
        )

        samples_per_sec_values = valid_timing["test_samples_per_sec"].dropna().to_numpy(dtype=float)
        sps_mean, sps_median, sps_std, sps_min, sps_max = summarize_values(samples_per_sec_values)

        row = {
            "model": model_name,
            "experiment": experiment,
            "accuracy_mean": acc_mean,
            "accuracy_median": acc_median,
            "accuracy_std": acc_std,
            "accuracy_min": acc_min,
            "accuracy_max": acc_max,
            "accuracy_mean_std": f"{acc_mean:.4f} ({acc_std:.4f})",
            "accuracy_mean_minmax": f"{acc_mean:.4f} [{acc_min:.4f}-{acc_max:.4f}]",
            "accuracy_median_std": f"{acc_median:.4f} ({acc_std:.4f})",
            "accuracy_median_minmax": f"{acc_median:.4f} [{acc_min:.4f}-{acc_max:.4f}]",
            "test_total_time_sec_mean": time_mean,
            "test_total_time_sec_median": time_median,
            "test_total_time_sec_std": time_std,
            "test_total_time_sec_min": time_min,
            "test_total_time_sec_max": time_max,
            "test_total_time_sec_mean_std": f"{time_mean:.4f} ({time_std:.4f})",
            "test_total_time_sec_mean_minmax": f"{time_mean:.4f} [{time_min:.4f}-{time_max:.4f}]",
            "test_total_time_sec_median_std": f"{time_median:.4f} ({time_std:.4f})",
            "test_total_time_sec_median_minmax": f"{time_median:.4f} [{time_min:.4f}-{time_max:.4f}]",
            "test_samples_per_sec_mean": sps_mean,
            "test_samples_per_sec_median": sps_median,
            "test_samples_per_sec_std": sps_std,
            "test_samples_per_sec_min": sps_min,
            "test_samples_per_sec_max": sps_max,
            "n_folds": int(len(model_rows)),
            "n_timed_folds": int(len(valid_timing)),
        }

        for key in DEVICE_KEYS:
            row[key] = first_non_null(valid_timing[key].tolist()) if key in valid_timing else np.nan

        complexity_row = complexity_df[complexity_df["model"] == model_name]
        if not complexity_row.empty:
            row.update(complexity_row.iloc[0].to_dict())

        rows.append(row)

    return pd.DataFrame(rows)


def bubble_size(values, global_min, global_max):
    values = np.asarray(values, dtype=float)

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        return np.full(values.shape, MIN_BUBBLE_AREA, dtype=float)

    # Marker sizes must always be finite and non-negative.  This also allows
    # the legend to include a clean reference value smaller than the smallest
    # plotted model, e.g. 100K when the smallest model has about 192K params.
    values = np.where(np.isfinite(values), values, global_min)
    values = np.maximum(values, 0.0)
    global_min = max(float(global_min), 0.0)
    global_max = max(float(global_max), global_min)

    if math.isclose(global_min, global_max):
        return np.full(values.shape, (MIN_BUBBLE_AREA + MAX_BUBBLE_AREA) / 2, dtype=float)

    scaled = (np.sqrt(values) - np.sqrt(global_min)) / (
        np.sqrt(global_max) - np.sqrt(global_min)
    )
    scaled = np.clip(scaled, 0.0, 1.0)
    sizes = MIN_BUBBLE_AREA + scaled * (MAX_BUBBLE_AREA - MIN_BUBBLE_AREA)
    return np.clip(sizes, MIN_BUBBLE_AREA, MAX_BUBBLE_AREA)


def build_legend_values(params):
    """
    Use rounded, human-readable reference values for the bubble legend.
    The values are illustrative anchors, not exact model parameter counts.
    """
    values = np.asarray(params, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([])

    vmin = float(values.min())
    vmax = float(values.max())

    candidates = np.array([100_000, 400_000, 2_000_000], dtype=float)

    # Keep the fixed, clean anchors when they are appropriate for the current range.
    if vmin <= 200_000 * 1.25 and vmax >= candidates[-1] * 0.75:
        return candidates.astype(int)

    # Fallback for other ranges: rounded min, median, max.
    def round_nice(v):
        if v >= 1_000_000:
            return round(v / 1_000_000, 1) * 1_000_000
        if v >= 100_000:
            return round(v / 100_000) * 100_000
        if v >= 10_000:
            return round(v / 10_000) * 10_000
        return round(v / 1_000) * 1_000

    legend_values = np.array([round_nice(vmin), round_nice(np.median(values)), round_nice(vmax)], dtype=float)
    return np.unique(legend_values.astype(int))


def save_plot_notes(df, experiment, out_path, error_bar_mode, summary_mode):
    gpu_names = sorted(str(v) for v in df["gpu_name"].dropna().unique()) if "gpu_name" in df else []
    if gpu_names:
        gpu_text = "GPU: " + "; ".join(gpu_names)
    else:
        gpu_text = "GPU: not available in timing metadata"

    lines = [
        f"Experiment: {experiment}",
        gpu_text,
        "Bubble size = trainable parameters",
        f"Summary = {summary_mode}",
        "Error bars = fold min-max" if error_bar_mode == "minmax" else "Error bars = std across folds",
        "",
        "Per-model summary:",
    ]

    for _, row in df.sort_values(summary_column("test_total_time_sec", summary_mode)).iterrows():
        lines.append(
            f"- {row['model']}: "
            f"accuracy={row[summary_column('accuracy', summary_mode)] * 100.0:.2f}% "
            f"(std={row['accuracy_std'] * 100.0:.2f}%), "
            f"test_time={row[summary_column('test_total_time_sec', summary_mode)]:.4f}s "
            f"(std={row['test_total_time_sec_std']:.4f}s), "
            f"trainable_params={int(row['trainable_params']) if pd.notna(row['trainable_params']) else 'NA'}"
        )

    out_path.write_text("\n".join(lines) + "\n")


def lighten_color(color, amount=0.45):
    """
    Return a lighter version of a matplotlib color.
    amount=0 keeps the original color; amount=1 returns white.
    """
    rgba = np.array(plt.matplotlib.colors.to_rgba(color), dtype=float)
    rgba[:3] = rgba[:3] + (1.0 - rgba[:3]) * amount
    return tuple(rgba)


def darken_color(color, amount=0.45):
    """Return a darker version of a matplotlib color.

    amount=0 keeps the original color; amount=1 returns black.
    Used for confidence/error bars so they remain linked to the model color
    while staying visible over semi-transparent bubbles.
    """
    rgba = np.array(plt.matplotlib.colors.to_rgba(color), dtype=float)
    rgba[:3] = rgba[:3] * (1.0 - amount)
    return tuple(rgba)


def error_bar_color_for(point_color):
    return darken_color(point_color, amount=0.42)


def add_bubble_size_legend(ax, legend_values, p_min, p_max):
    """Draw a custom bubble-size legend in the lower-right corner.

    The vertical positions are fixed and intentionally spacious so that the
    largest reference bubble does not overlap the smaller ones.
    """
    if len(legend_values) == 0:
        return

    legend_values = np.asarray(legend_values, dtype=float)
    legend_values = np.sort(legend_values)

    box_x = 0.790
    box_y = 0.035
    box_w = 0.190
    box_h = 0.425

    box = plt.matplotlib.patches.FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.003",
        facecolor="white",
        edgecolor="0.82",
        linewidth=1.0,
        alpha=0.92,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(box)

    ax.text(
        box_x + box_w / 2.0,
        box_y + box_h - 0.055,
        "Trainable parameters",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=LEGEND_TITLE_SIZE,
        zorder=12,
    )

    # Top-to-bottom: small, medium, large.  The lower item has extra space
    # because its bubble is the largest.
    if len(legend_values) == 1:
        y_positions = np.array([box_y + 0.17])
    elif len(legend_values) == 2:
        y_positions = np.array([box_y + 0.265, box_y + 0.125])
    else:
        y_positions = np.array([box_y + 0.295, box_y + 0.205, box_y + 0.090])
        if len(legend_values) != 3:
            y_positions = np.linspace(box_y + box_h - 0.13, box_y + 0.09, len(legend_values))

    x_circle = box_x + 0.045
    x_label = box_x + 0.090

    for value, y_pos in zip(legend_values, y_positions):
        size = bubble_size(np.array([value], dtype=float), p_min, p_max)[0]

        ax.scatter(
            [x_circle],
            [y_pos],
            s=size,
            transform=ax.transAxes,
            alpha=0.72,
            edgecolors="black",
            linewidths=0.8,
            color="gray",
            zorder=11,
            clip_on=False,
        )

        ax.text(
            x_label,
            y_pos,
            format_params(value),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LEGEND_SIZE,
            zorder=12,
        )



def add_model_color_legend(ax, df, color_map, p_min=None, p_max=None):
    """Add a compact model legend with trainable-parameter counts."""
    present_models = set(df["model"].astype(str).tolist())
    model_names = [
        model_info["name"]
        for model_info in MODELS
        if model_info["name"] in present_models
    ]
    if not model_names:
        return

    params_by_model = representative_params_by_model(df)
    if p_min is None or p_max is None:
        values = np.array(list(params_by_model.values()), dtype=float)
        values = values[np.isfinite(values)]
        p_min = float(values.min()) if values.size else 0.0
        p_max = float(values.max()) if values.size else 1.0

    handles = build_model_param_handles(model_names, color_map, params_by_model, p_min, p_max)
    model_legend = ax.legend(
        handles=handles,
        title="Model (trainable parameters)",
        loc="lower right",
        bbox_to_anchor=(0.992, 0.035),
        borderaxespad=0.0,
        fontsize=max(13, LEGEND_SIZE),
        title_fontsize=max(14, LEGEND_TITLE_SIZE),
        framealpha=0.93,
        labelspacing=1.05,
    )
    ax.add_artist(model_legend)


def add_cross_split_legends(ax, df, color_map, legend_values, p_min, p_max):
    """Add cross-split legends inside the plotting area."""
    present_models = set(df["model"].astype(str).tolist())
    model_names = [
        model_info["name"]
        for model_info in MODELS
        if model_info["name"] in present_models
    ]
    params_by_model = representative_params_by_model(df)

    model_handles = build_model_param_handles(model_names, color_map, params_by_model, p_min, p_max)
    if model_handles:
        model_legend = ax.legend(
            handles=model_handles,
            title="Model (trainable parameters)",
            loc="upper right",
            bbox_to_anchor=(0.992, 0.992),
            borderaxespad=0.0,
            fontsize=max(12, LEGEND_SIZE - 1),
            title_fontsize=max(13, LEGEND_TITLE_SIZE - 1),
            framealpha=0.93,
            labelspacing=1.00,
        )
        ax.add_artist(model_legend)

    shade_handles = []
    neutral_base = "0.15"
    for experiment in EXPERIMENTS:
        shade_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=lighten_color(
                    neutral_base,
                    amount=EXPERIMENT_COLOR_LIGHTENING.get(experiment, 0.0),
                ),
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=9,
                label=EXPERIMENT_TITLES.get(experiment, experiment),
            )
        )

    shade_legend = ax.legend(
        handles=shade_handles,
        title="Evaluation protocol",
        loc="lower right",
        bbox_to_anchor=(0.992, 0.035),
        borderaxespad=0.0,
        fontsize=max(10, LEGEND_SIZE - 3),
        title_fontsize=max(11, LEGEND_TITLE_SIZE - 3),
        framealpha=0.93,
    )
    ax.add_artist(shade_legend)

def save_cross_split_plot_notes(df, group_name, out_path, error_bar_mode, summary_mode):
    gpu_names = sorted(str(v) for v in df["gpu_name"].dropna().unique()) if "gpu_name" in df else []
    if gpu_names:
        gpu_text = "GPU: " + "; ".join(gpu_names)
    else:
        gpu_text = "GPU: not available in timing metadata"

    lines = [
        f"Comparison group: {group_name}",
        gpu_text,
        "Bubble size = trainable parameters",
        "Color = model",
        "Shade = evaluation protocol",
        f"Summary = {summary_mode}",
        "Error bars = fold min-max" if error_bar_mode == "minmax" else "Error bars = std across folds",
        "",
        "Per-model/per-protocol summary:",
    ]

    ordered_models = [model_info["name"] for model_info in MODELS]
    df = df.copy()
    df["model_order"] = df["model"].map({name: i for i, name in enumerate(ordered_models)})
    df["experiment_order"] = df["experiment"].map({name: i for i, name in enumerate(EXPERIMENTS)})

    for _, row in df.sort_values(["model_order", "experiment_order"]).iterrows():
        lines.append(
            f"- {row['model']} | {row['experiment']}: "
            f"accuracy={row[summary_column('accuracy', summary_mode)] * 100.0:.2f}% "
            f"(std={row['accuracy_std'] * 100.0:.2f}%), "
            f"test_time={row[summary_column('test_total_time_sec', summary_mode)]:.4f}s "
            f"(std={row['test_total_time_sec_std']:.4f}s), "
            f"trainable_params={int(row['trainable_params']) if pd.notna(row['trainable_params']) else 'NA'}"
        )

    out_path.write_text("\n".join(lines) + "\n")


def save_cross_split_tradeoff_plot(df, group_name, out_path, error_bar_mode, summary_mode, axis_limits=None, param_limits=None, show_error_bars=True):
    df = df.dropna(
        subset=[summary_column("test_total_time_sec", summary_mode), "trainable_params", summary_column("accuracy", summary_mode), "experiment"]
    ).copy()

    if df.empty:
        print(f"No plottable cross-split timing data for {group_name}")
        return

    df = df[df["experiment"].isin(EXPERIMENTS)].copy()
    if df.empty:
        print(f"No individual split data for {group_name}")
        return

    x, y, x_lower, x_upper, y_lower, y_upper = get_tradeoff_bounds(df, error_bar_mode, summary_mode)
    params = df["trainable_params"].to_numpy(dtype=float)

    if param_limits is not None:
        p_min = float(param_limits["p_min"])
        p_max = float(param_limits["p_max"])
    else:
        p_min = float(np.nanmin(params))
        p_max = float(np.nanmax(params))

    fig, ax = plt.subplots(figsize=(16.0, 10.0))

    color_map = build_model_color_map()

    model_order = {model_info["name"]: i for i, model_info in enumerate(MODELS)}
    experiment_order = {experiment: i for i, experiment in enumerate(EXPERIMENTS)}
    df = df.copy()
    df["model_order"] = df["model"].map(model_order)
    df["experiment_order"] = df["experiment"].map(experiment_order)
    # Draw larger bubbles first, so smaller overlapping bubbles remain visible on top.
    df = df.sort_values(["trainable_params", "model_order", "experiment_order"], ascending=[False, True, True])

    for _, row in df.iterrows():
        point_color = lighten_color(
            color_map[row["model"]],
            amount=EXPERIMENT_COLOR_LIGHTENING.get(row["experiment"], 0.0),
        )
        err_color = error_bar_color_for(point_color)

        x_center = float(row[summary_column("test_total_time_sec", summary_mode)])
        y_center = float(row[summary_column("accuracy", summary_mode)] * 100.0)

        if error_bar_mode == "std":
            x_low, x_high = error_interval(
                x_center,
                float(row["test_total_time_sec_std"]),
                float(row["test_total_time_sec_min"]),
                float(row["test_total_time_sec_max"]),
                error_bar_mode,
            )
            y_low, y_high = error_interval(
                y_center,
                float(row["accuracy_std"] * 100.0),
                float(row["accuracy_min"] * 100.0),
                float(row["accuracy_max"] * 100.0),
                error_bar_mode,
            )
        else:
            x_low = float(row["test_total_time_sec_min"])
            x_high = float(row["test_total_time_sec_max"])
            y_low = float(row["accuracy_min"] * 100.0)
            y_high = float(row["accuracy_max"] * 100.0)

        if show_error_bars:
            ax.errorbar(
                x_center,
                y_center,
                xerr=asymmetric_error(x_center, x_low, x_high),
                yerr=asymmetric_error(y_center, y_low, y_high),
                fmt="none",
                ecolor=err_color,
                elinewidth=ERROR_BAR_LINEWIDTH,
                capsize=ERROR_BAR_CAPSIZE,
                capthick=ERROR_BAR_CAPTHICK,
                alpha=ERROR_BAR_ALPHA,
                zorder=9,
            )

        ax.scatter(
            x_center,
            y_center,
            s=bubble_size(
                np.array([row["trainable_params"]], dtype=float),
                p_min,
                p_max,
            )[0],
            alpha=0.76,
            edgecolors="black",
            linewidths=0.9,
            color=point_color,
            marker="o",
            zorder=4,
        )

    ax.set_xlabel("Test-set inference time [s]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Accuracy [%]", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(
        "Accuracy-Efficiency Trade-off Across Evaluation Protocols",
        fontsize=TITLE_SIZE,
        pad=26,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x_min"], axis_limits["x_max"])
        ax.set_ylim(axis_limits["y_min"], axis_limits["y_max"])
    else:
        x_min_raw = float(np.nanmin(x_lower))
        x_max_raw = float(np.nanmax(x_upper))
        y_min_raw = float(np.nanmin(y_lower))
        y_max_raw = float(np.nanmax(y_upper))
        x_range = max(x_max_raw - x_min_raw, 1e-9)
        y_range = max(y_max_raw - y_min_raw, 1e-9)
        ax.set_xlim(x_min_raw - x_range * 0.10, x_max_raw + x_range * 0.10)
        ax.set_ylim(max(0.0, y_min_raw - y_range * 0.10), y_max_raw + y_range * 0.10)

    apply_tradeoff_axis_ticks(ax)

    legend_values = build_legend_values(
        np.array([p_min, p_max], dtype=float) if param_limits is not None else params
    )

    add_cross_split_legends(ax, df, color_map, legend_values, p_min, p_max)

    note_path = out_path.with_suffix(".txt")
    save_cross_split_plot_notes(df, group_name, note_path, error_bar_mode, summary_mode)
    print(f"Saved: {note_path}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_tradeoff_on_ax(
    ax,
    df,
    experiment,
    error_bar_mode,
    summary_mode,
    axis_limits=None,
    param_limits=None,
    color_map=None,
    show_xlabel=True,
    show_ylabel=True,
    panel_label=None,
    show_error_bars=True,
):
    df = df.dropna(
        subset=[summary_column("test_total_time_sec", summary_mode), "trainable_params", summary_column("accuracy", summary_mode)]
    ).copy()

    if df.empty:
        ax.axis("off")
        return None, None, None

    # Draw larger bubbles first, so smaller overlapping bubbles remain visible on top.
    df = df.sort_values(["trainable_params", summary_column("test_total_time_sec", summary_mode)], ascending=[False, True])
    _, _, x_lower, x_upper, y_lower, y_upper = get_tradeoff_bounds(df, error_bar_mode, summary_mode)
    params = df["trainable_params"].to_numpy(dtype=float)

    if param_limits is not None:
        p_min = float(param_limits["p_min"])
        p_max = float(param_limits["p_max"])
    else:
        p_min = float(np.nanmin(params))
        p_max = float(np.nanmax(params))

    if color_map is None:
        color_map = build_model_color_map()

    for _, row in df.iterrows():
        point_color = color_map[row["model"]]
        err_color = error_bar_color_for(point_color)

        x_center = float(row[summary_column("test_total_time_sec", summary_mode)])
        y_center = float(row[summary_column("accuracy", summary_mode)] * 100.0)

        if error_bar_mode == "std":
            x_low, x_high = error_interval(
                x_center,
                float(row["test_total_time_sec_std"]),
                float(row["test_total_time_sec_min"]),
                float(row["test_total_time_sec_max"]),
                error_bar_mode,
            )
            y_low, y_high = error_interval(
                y_center,
                float(row["accuracy_std"] * 100.0),
                float(row["accuracy_min"] * 100.0),
                float(row["accuracy_max"] * 100.0),
                error_bar_mode,
            )
        else:
            x_low = float(row["test_total_time_sec_min"])
            x_high = float(row["test_total_time_sec_max"])
            y_low = float(row["accuracy_min"] * 100.0)
            y_high = float(row["accuracy_max"] * 100.0)

        if show_error_bars:
            ax.errorbar(
                x_center,
                y_center,
                xerr=asymmetric_error(x_center, x_low, x_high),
                yerr=asymmetric_error(y_center, y_low, y_high),
                fmt="none",
                ecolor=err_color,
                elinewidth=ERROR_BAR_LINEWIDTH,
                capsize=ERROR_BAR_CAPSIZE,
                capthick=ERROR_BAR_CAPTHICK,
                alpha=ERROR_BAR_ALPHA,
                zorder=9,
            )

        ax.scatter(
            x_center,
            y_center,
            s=bubble_size(
                np.array([row["trainable_params"]], dtype=float),
                p_min,
                p_max,
            )[0],
            alpha=0.72,
            edgecolors="black",
            linewidths=0.7,
            color=point_color,
            zorder=4,
        )

    ax.set_xlabel("Test-set inference time [s]" if show_xlabel else "", fontsize=max(12, AXIS_LABEL_SIZE - 5))
    ax.set_ylabel("Accuracy [%]" if show_ylabel else "", fontsize=max(12, AXIS_LABEL_SIZE - 5))
    title = EXPERIMENT_TITLES.get(experiment, experiment)
    if panel_label:
        title = f"{panel_label}. {title}"
    ax.set_title(title, fontsize=max(13, TITLE_SIZE - 7), fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=max(10, TICK_SIZE - 4))

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x_min"], axis_limits["x_max"])
        ax.set_ylim(axis_limits["y_min"], axis_limits["y_max"])
    else:
        x_min_raw = float(np.nanmin(x_lower))
        x_max_raw = float(np.nanmax(x_upper))
        y_min_raw = float(np.nanmin(y_lower))
        y_max_raw = float(np.nanmax(y_upper))
        x_range = max(x_max_raw - x_min_raw, 1e-9)
        y_range = max(y_max_raw - y_min_raw, 1e-9)
        ax.set_xlim(x_min_raw - x_range * 0.10, x_max_raw + x_range * 0.10)
        ax.set_ylim(max(0.0, y_min_raw - y_range * 0.10), y_max_raw + y_range * 0.10)

    apply_tradeoff_axis_ticks(ax)

    return color_map, p_min, p_max


def add_tradeoff_panel_legend(fig, model_names, color_map, p_min, p_max, params, params_by_model=None, bottom_y=0.02):
    """Add a bottom legend combining model identity and trainable parameters."""
    if params_by_model is None:
        params_by_model = {}

    model_handles = build_model_param_handles(
        model_names,
        color_map,
        params_by_model,
        p_min,
        p_max,
    )
    if not model_handles:
        return

    # One item per model, with exact trainable parameter count in the label.
    # Marker size still follows the bubble-size mapping, but is capped to keep
    # the all-model panel readable.
    n_items = len(model_handles)
    fig.legend(
        handles=model_handles,
        title="Model (trainable parameters)",
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_y),
        ncol=n_items,
        frameon=False,
        fontsize=max(13, LEGEND_SIZE),
        title_fontsize=max(14, LEGEND_TITLE_SIZE),
        columnspacing=1.65,
        handletextpad=0.85,
        labelspacing=1.05,
    )

def save_tradeoff_panel(plot_tables, group_name, out_path, error_bar_mode, summary_mode, axis_limits=None, param_limits=None, layout="2x2", show_error_bars=True):
    if layout == "2x2":
        panel_order = EXPERIMENT_PANEL_ORDER
        nrows, ncols = 2, 2
        fig_size = (16.0, 11.0)
        bottom = 0.16
        hspace = 0.42
        wspace = 0.18
        bottom_y = 0.025
    elif layout == "1x3":
        panel_order = EXPERIMENT_ROW_PANEL_ORDER
        nrows, ncols = 1, 3
        fig_size = (21.0, 6.9)
        bottom = 0.34
        hspace = 0.0
        wspace = 0.18
        bottom_y = 0.045
    else:
        raise ValueError(f"Unsupported panel layout: {layout}")

    available = [
        (panel_label, experiment_name)
        for panel_label, experiment_name in panel_order
        if (group_name, experiment_name) in plot_tables
    ]
    if not available:
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    color_map = build_model_color_map()
    all_params = []
    params_by_model_for_legend = {}
    p_min = None
    p_max = None

    for ax, (panel_label, experiment_name) in zip(axes, available):
        df = plot_tables[(group_name, experiment_name)]
        used_color_map, used_p_min, used_p_max = draw_tradeoff_on_ax(
            ax,
            df,
            experiment_name,
            error_bar_mode,
            summary_mode,
            axis_limits=axis_limits,
            param_limits=param_limits,
            color_map=color_map,
            show_xlabel=(layout == "1x3") or (panel_label in {"C", "D"}),
            show_ylabel=(panel_label in {"A", "C"}) if layout == "2x2" else (panel_label == "A"),
            panel_label=panel_label,
            show_error_bars=show_error_bars,
        )
        if used_color_map is not None:
            color_map = used_color_map
            p_min = used_p_min if p_min is None else min(p_min, used_p_min)
            p_max = used_p_max if p_max is None else max(p_max, used_p_max)
            params = df["trainable_params"].to_numpy(dtype=float)
            params = params[np.isfinite(params)]
            if params.size:
                all_params.append(params)
            params_by_model_for_legend.update(representative_params_by_model(df))

    for ax in axes[len(available):]:
        ax.axis("off")

    if all_params and p_min is not None and p_max is not None:
        model_names = [model for model in COMPARISON_GROUPS[group_name] if any(model in plot_tables[(group_name, exp)]["model"].values for _, exp in available)]
        add_tradeoff_panel_legend(
            fig,
            model_names,
            color_map,
            p_min,
            p_max,
            np.concatenate(all_params),
            params_by_model=params_by_model_for_legend,
            bottom_y=bottom_y,
        )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=bottom, wspace=wspace, hspace=hspace)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_plot(df, experiment, out_path, error_bar_mode, summary_mode, axis_limits=None, param_limits=None, show_error_bars=True):
    df = df.dropna(
        subset=[summary_column("test_total_time_sec", summary_mode), "trainable_params", summary_column("accuracy", summary_mode)]
    ).copy()

    if df.empty:
        print(f"No plottable timing data for {experiment}")
        return

    # Draw larger bubbles first, so smaller overlapping bubbles remain visible on top.
    df = df.sort_values(["trainable_params", summary_column("test_total_time_sec", summary_mode)], ascending=[False, True])

    x, y, x_lower, x_upper, y_lower, y_upper = get_tradeoff_bounds(df, error_bar_mode, summary_mode)
    params = df["trainable_params"].to_numpy(dtype=float)

    if param_limits is not None:
        p_min = float(param_limits["p_min"])
        p_max = float(param_limits["p_max"])
    else:
        p_min = float(np.nanmin(params))
        p_max = float(np.nanmax(params))

    fig, ax = plt.subplots(figsize=(16.0, 10.0))

    color_map = build_model_color_map()

    for _, row in df.iterrows():
        point_color = color_map[row["model"]]
        err_color = error_bar_color_for(point_color)

        x_center = float(row[summary_column("test_total_time_sec", summary_mode)])
        y_center = float(row[summary_column("accuracy", summary_mode)] * 100.0)

        if error_bar_mode == "std":
            x_low, x_high = error_interval(
                x_center,
                float(row["test_total_time_sec_std"]),
                float(row["test_total_time_sec_min"]),
                float(row["test_total_time_sec_max"]),
                error_bar_mode,
            )
            y_low, y_high = error_interval(
                y_center,
                float(row["accuracy_std"] * 100.0),
                float(row["accuracy_min"] * 100.0),
                float(row["accuracy_max"] * 100.0),
                error_bar_mode,
            )
        else:
            x_low = float(row["test_total_time_sec_min"])
            x_high = float(row["test_total_time_sec_max"])
            y_low = float(row["accuracy_min"] * 100.0)
            y_high = float(row["accuracy_max"] * 100.0)

        if show_error_bars:
            ax.errorbar(
                x_center,
                y_center,
                xerr=asymmetric_error(x_center, x_low, x_high),
                yerr=asymmetric_error(y_center, y_low, y_high),
                fmt="none",
                ecolor=err_color,
                elinewidth=ERROR_BAR_LINEWIDTH,
                capsize=ERROR_BAR_CAPSIZE,
                capthick=ERROR_BAR_CAPTHICK,
                alpha=ERROR_BAR_ALPHA,
                zorder=9,
            )

        ax.scatter(
            row[summary_column("test_total_time_sec", summary_mode)],
            row[summary_column("accuracy", summary_mode)] * 100.0,
            s=bubble_size(
                np.array([row["trainable_params"]], dtype=float),
                p_min,
                p_max,
            )[0],
            alpha=0.72,
            edgecolors="black",
            linewidths=0.8,
            color=point_color,
            zorder=4,
        )

    ax.set_xlabel("Test-set inference time [s]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Accuracy [%]", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(
        f"Accuracy-Efficiency Trade-off\n"
        f"{EXPERIMENT_TITLES.get(experiment, experiment)}",
        fontsize=TITLE_SIZE,
        pad=26,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x_min"], axis_limits["x_max"])
        ax.set_ylim(axis_limits["y_min"], axis_limits["y_max"])
    else:
        x_min_raw = float(np.nanmin(x_lower))
        x_max_raw = float(np.nanmax(x_upper))
        y_min_raw = float(np.nanmin(y_lower))
        y_max_raw = float(np.nanmax(y_upper))
        x_range = max(x_max_raw - x_min_raw, 1e-9)
        y_range = max(y_max_raw - y_min_raw, 1e-9)
        ax.set_xlim(x_min_raw - x_range * 0.10, x_max_raw + x_range * 0.10)
        ax.set_ylim(max(0.0, y_min_raw - y_range * 0.10), y_max_raw + y_range * 0.10)

    apply_tradeoff_axis_ticks(ax)

    add_model_color_legend(ax, df, color_map, p_min, p_max)

    legend_values = build_legend_values(
        np.array([p_min, p_max], dtype=float) if param_limits is not None else params
    )
    add_bubble_size_legend(ax, legend_values, p_min, p_max)

    note_path = out_path.with_suffix(".txt")
    save_plot_notes(df, experiment, note_path, error_bar_mode, summary_mode)
    print(f"Saved: {note_path}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    error_bar_mode = args.error_bar
    summary_mode = args.summary

    base_dir = Path(paths.logs_dir)
    out_dir = base_dir / "aggregate_model_tradeoff" / output_subdir_name(summary_mode, error_bar_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {out_dir}")

    recap_1x3_dir = out_dir / RECAP_1X3_DIR_NAME
    recap_1x3_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using recap 1x3 figure directory: {recap_1x3_dir}")

    fold_df = load_fold_table(base_dir)
    fold_table_path = out_dir / "fold_level_inference_timing.csv"
    fold_df.to_csv(fold_table_path, index=False, sep=";")
    print(f"Saved: {fold_table_path}")

    complexity_df = build_complexity_table()
    complexity_path = out_dir / "model_complexity.csv"
    complexity_df.to_csv(complexity_path, index=False, sep=";")
    print(f"Saved: {complexity_path}")

    # First build all plot tables. This lets us compute one common x/y range
    # for every model-tradeoff figure before saving any plot.
    plot_tables = {}
    cross_split_tables = {}
    all_tables_for_limits = []

    for group_name, model_names in COMPARISON_GROUPS.items():
        individual_split_tables = []

        for experiment in EXPERIMENTS + ["all_experiments"]:
            df = summarize_experiment(
                fold_df=fold_df,
                complexity_df=complexity_df,
                experiment=experiment,
                model_names=model_names,
            )

            if df.empty:
                print(f"No data for {group_name} - {experiment}")
                continue

            plot_tables[(group_name, experiment)] = df
            all_tables_for_limits.append(df)

            if experiment in EXPERIMENTS:
                individual_split_tables.append(df)

        if individual_split_tables:
            cross_split_df = pd.concat(individual_split_tables, ignore_index=True)
            cross_split_tables[group_name] = cross_split_df
            all_tables_for_limits.append(cross_split_df)

    axis_limits, param_limits = compute_global_tradeoff_limits(
        all_tables_for_limits,
        error_bar_mode,
        summary_mode,
    )

    if axis_limits is not None:
        limits_path = out_dir / "global_model_tradeoff_axis_limits.json"
        with open(limits_path, "w") as f:
            json.dump(
                {
                    "error_bar_mode": error_bar_mode,
                    "summary_mode": summary_mode,
                    "axis_limits": axis_limits,
                    "param_limits": param_limits,
                },
                f,
                indent=2,
            )
        print(f"Saved: {limits_path}")

    for group_name, model_names in COMPARISON_GROUPS.items():
        group_out_dir = out_dir / group_name
        group_out_dir.mkdir(parents=True, exist_ok=True)

        for experiment in EXPERIMENTS + ["all_experiments"]:
            df = plot_tables.get((group_name, experiment))
            if df is None or df.empty:
                continue

            csv_path = group_out_dir / f"{experiment}_{group_name}_model_tradeoff_timing.csv"
            df.to_csv(csv_path, index=False, sep=";")
            print(f"Saved: {csv_path}")

            plot_path = group_out_dir / f"{experiment}_{group_name}_model_tradeoff_timing.png"
            save_tradeoff_plot(
                df,
                experiment,
                plot_path,
                error_bar_mode,
                summary_mode,
                axis_limits=axis_limits,
                param_limits=param_limits,
                show_error_bars=True,
            )
            print(f"Saved: {plot_path}")

            plot_no_error_path = group_out_dir / f"{experiment}_{group_name}_model_tradeoff_timing_no_errorbars.png"
            save_tradeoff_plot(
                df,
                experiment,
                plot_no_error_path,
                error_bar_mode,
                summary_mode,
                axis_limits=axis_limits,
                param_limits=param_limits,
                show_error_bars=False,
            )
            print(f"Saved: {plot_no_error_path}")

        cross_split_df = cross_split_tables.get(group_name)
        if cross_split_df is not None and not cross_split_df.empty:
            cross_split_csv_path = group_out_dir / f"cross_split_{group_name}_model_tradeoff_timing.csv"
            cross_split_df.to_csv(cross_split_csv_path, index=False, sep=";")
            print(f"Saved: {cross_split_csv_path}")

            cross_split_plot_path = group_out_dir / f"cross_split_{group_name}_model_tradeoff_timing.png"
            save_cross_split_tradeoff_plot(
                cross_split_df,
                group_name,
                cross_split_plot_path,
                error_bar_mode,
                summary_mode,
                axis_limits=axis_limits,
                param_limits=param_limits,
                show_error_bars=True,
            )
            print(f"Saved: {cross_split_plot_path}")

            cross_split_no_error_plot_path = group_out_dir / f"cross_split_{group_name}_model_tradeoff_timing_no_errorbars.png"
            save_cross_split_tradeoff_plot(
                cross_split_df,
                group_name,
                cross_split_no_error_plot_path,
                error_bar_mode,
                summary_mode,
                axis_limits=axis_limits,
                param_limits=param_limits,
                show_error_bars=False,
            )
            print(f"Saved: {cross_split_no_error_plot_path}")

        panel_2x2_path = group_out_dir / f"all_protocols_{group_name}_model_tradeoff_panel_2x2.png"
        save_tradeoff_panel(
            plot_tables,
            group_name,
            panel_2x2_path,
            error_bar_mode,
            summary_mode,
            axis_limits=axis_limits,
            param_limits=param_limits,
            layout="2x2",
            show_error_bars=True,
        )
        if panel_2x2_path.exists():
            print(f"Saved: {panel_2x2_path}")

        panel_2x2_no_error_path = group_out_dir / f"all_protocols_{group_name}_model_tradeoff_panel_2x2_no_errorbars.png"
        save_tradeoff_panel(
            plot_tables,
            group_name,
            panel_2x2_no_error_path,
            error_bar_mode,
            summary_mode,
            axis_limits=axis_limits,
            param_limits=param_limits,
            layout="2x2",
            show_error_bars=False,
        )
        if panel_2x2_no_error_path.exists():
            print(f"Saved: {panel_2x2_no_error_path}")

        panel_1x3_path = group_out_dir / f"split_protocols_{group_name}_model_tradeoff_panel_1x3.png"
        save_tradeoff_panel(
            plot_tables,
            group_name,
            panel_1x3_path,
            error_bar_mode,
            summary_mode,
            axis_limits=axis_limits,
            param_limits=param_limits,
            layout="1x3",
            show_error_bars=True,
        )
        if panel_1x3_path.exists():
            print(f"Saved: {panel_1x3_path}")
            recap_path = copy_figure_for_paper(
                panel_1x3_path,
                recap_1x3_dir,
                f"{group_name}_model_tradeoff_panel_1x3.png",
            )
            if recap_path is not None:
                print(f"Saved: {recap_path}")

        panel_1x3_no_error_path = group_out_dir / f"split_protocols_{group_name}_model_tradeoff_panel_1x3_no_errorbars.png"
        save_tradeoff_panel(
            plot_tables,
            group_name,
            panel_1x3_no_error_path,
            error_bar_mode,
            summary_mode,
            axis_limits=axis_limits,
            param_limits=param_limits,
            layout="1x3",
            show_error_bars=False,
        )
        if panel_1x3_no_error_path.exists():
            print(f"Saved: {panel_1x3_no_error_path}")
            recap_no_error_path = copy_figure_for_paper(
                panel_1x3_no_error_path,
                recap_1x3_dir,
                f"{group_name}_model_tradeoff_panel_1x3_no_errorbars.png",
            )
            if recap_no_error_path is not None:
                print(f"Saved: {recap_no_error_path}")


if __name__ == "__main__":
    main()
