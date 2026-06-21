try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

from pathlib import Path
import argparse
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import paths
from classifier.workflow.utils import class_names


MODELS = [
    'denformer_first',
    'denformer_max',
    'denformer_mean',
    'longformer',
    'performer',
    'ffnn',
    'logreg',
]

DENFORMER_VARIANTS = [
    'denformer_first',
    'denformer_max',
    'denformer_mean',
]

SELECTED_DENFORMER = 'denformer_mean'

OTHER_MODELS = [
    model for model in MODELS
    if model not in DENFORMER_VARIANTS
]

COMPARISON_GROUPS = {
    'all_models': MODELS,
    'denformer_variants': DENFORMER_VARIANTS,
    'selected_denformer_vs_others': [SELECTED_DENFORMER] + OTHER_MODELS,
}

EXPERIMENTS = [
    'ohe_cdhit_e100',
    'ohe_continent_e100',
    'ohe_timebin_e100',
]

MODEL_AGGREGATE_STRATEGIES = [
    ('ohe_continent_e100', 'Geographical'),
    ('ohe_timebin_e100', 'Temporal'),
    ('ohe_cdhit_e100', 'CD-HIT'),
]

CM_FILES = [
    'confusion_matrices_test_hconcat_counts.png',
    'confusion_matrices_test_hconcat_all_normalized.png',
    'confusion_matrices_test_hconcat_row_normalized.png',
]

LABEL_WIDTH = 900


# -----------------------------------------------------------------------------
# Existing model-comparison aggregates
# -----------------------------------------------------------------------------
def add_label(img, label):
    h, w = img.shape[:2]

    canvas = np.full((h, LABEL_WIDTH + w, 3), 255, dtype=np.uint8)
    canvas[:, LABEL_WIDTH:] = img

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, h // 350)
    font_scale = max(2.0, h / 900)

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    while text_w > LABEL_WIDTH - 80:
        font_scale *= 0.9
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    x = (LABEL_WIDTH - text_w) // 2
    y = (h + text_h) // 2

    cv2.putText(
        canvas,
        label,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    return canvas


def resize_to_width(img, target_width):
    h, w = img.shape[:2]

    if w == target_width:
        return img

    new_h = int(h * target_width / w)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)


def aggregate_confusion_matrices(base_dir, out_dir, experiment, group_name, models):
    experiment_out_dir = out_dir / group_name / experiment
    experiment_out_dir.mkdir(parents=True, exist_ok=True)

    for cm_file in CM_FILES:
        rows = []

        for model in models:
            img_path = base_dir / model / experiment / 'metrics' / cm_file

            if not img_path.exists():
                print(f'Missing: {img_path}')
                continue

            img = cv2.imread(str(img_path))

            if img is None:
                print(f'Could not read: {img_path}')
                continue

            img = add_label(img, model)
            rows.append(img)

        if not rows:
            print(f'No images found for {group_name} - {experiment} - {cm_file}')
            continue

        target_width = max(row.shape[1] for row in rows)
        rows = [resize_to_width(row, target_width) for row in rows]

        aggregated = cv2.vconcat(rows)

        out_path = experiment_out_dir / cm_file.replace(
            'confusion_matrices_test_hconcat',
            f'confusion_matrices_test_{group_name}_vconcat',
        )

        cv2.imwrite(str(out_path), aggregated)
        print(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# DENFormer model-level aggregate: rows = split strategies, columns = folds
# -----------------------------------------------------------------------------
def display_model_name(model_name):
    if model_name.startswith('denformer_'):
        return 'DENFormer-' + model_name.split('_', 1)[1]
    return model_name


def output_model_filename(model_name):
    if model_name.startswith('denformer_'):
        return 'DENFormer_' + model_name.split('_', 1)[1]
    return model_name


def display_fold_name(fold):
    fold = str(fold)
    if fold.startswith('fold_'):
        suffix = fold.split('fold_', 1)[1]
        if suffix.isdigit():
            return suffix
    return fold


def read_inference_summary(metrics_dir):
    summary_path = metrics_dir / 'inference_summary.json'
    if not summary_path.exists():
        print(f'Missing inference summary: {summary_path}')
        return []

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    rows = []
    for item in summary:
        test_result = item.get('test', {})
        cm = test_result.get('confusion_matrix')
        if cm is None:
            print(f"Missing test confusion matrix in: {summary_path} | fold={item.get('fold')}")
            continue
        rows.append({
            'fold': str(item.get('fold')),
            'accuracy': test_result.get('accuracy'),
            'confusion_matrix': np.asarray(cm, dtype=int),
        })

    def fold_key(row):
        fold = row['fold']
        try:
            return (0, int(fold))
        except ValueError:
            return (1, fold)

    return sorted(rows, key=fold_key)


def collect_denformer_model_matrices(base_dir, model_name):
    collected = []
    for experiment, strategy_label in MODEL_AGGREGATE_STRATEGIES:
        metrics_dir = base_dir / model_name / experiment / 'metrics'
        fold_rows = read_inference_summary(metrics_dir)
        if not fold_rows:
            print(f'No fold-level confusion matrices found for {model_name} | {experiment}')
        collected.append({
            'experiment': experiment,
            'strategy_label': strategy_label,
            'folds': fold_rows,
        })
    return collected


def aggregate_denformer_model_confmats(base_dir, out_dir, model_name, num_classes=4):
    collected = collect_denformer_model_matrices(base_dir, model_name)
    max_folds = max((len(row['folds']) for row in collected), default=0)

    if max_folds == 0:
        print(f'No DENFormer model aggregate generated for {model_name}: no matrices found.')
        return

    labels = class_names(num_classes)
    vmax = max(
        int(fold['confusion_matrix'].max())
        for row in collected
        for fold in row['folds']
    )
    vmax = max(vmax, 1)

    n_rows = len(collected)
    fig_width = max(3.25 * max_folds + 1.1, 10.5)
    fig_height = max(3.15 * n_rows + 0.9, 8.0)
    fig, axes = plt.subplots(
        n_rows,
        max_folds,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    im = None
    for row_idx, row in enumerate(collected):
        strategy_label = row['strategy_label']
        folds = row['folds']

        for col_idx in range(max_folds):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(folds):
                ax.axis('off')
                continue

            fold = folds[col_idx]
            cm = fold['confusion_matrix']
            im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax)

            title = f"Fold {display_fold_name(fold['fold'])}"
            if fold['accuracy'] is not None:
                title += f"\nAcc. {fold['accuracy'] * 100:.2f}%"
            ax.set_title(title, fontsize=12, pad=8)

            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(labels, fontsize=10)
            # Keep only class tick labels inside each panel. Repeated axis labels
            # make this dense aggregate figure harder to read and can overlap
            # across neighbouring subplots.

            threshold = vmax * 0.5
            for i in range(num_classes):
                for j in range(num_classes):
                    value = int(cm[i, j])
                    text_color = 'white' if value > threshold else 'black'
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha='center',
                        va='center',
                        color=text_color,
                        fontsize=11,
                        fontweight='bold',
                    )

        left_ax = axes[row_idx, 0]
        left_ax.annotate(
            strategy_label,
            xy=(-0.42, 0.5),
            xycoords='axes fraction',
            ha='right',
            va='center',
            rotation=90,
            fontsize=15,
            fontweight='bold',
        )

    fig.suptitle(display_model_name(model_name), fontsize=18, y=0.985)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.075,
        wspace=0.35,
        hspace=0.55,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{output_model_filename(model_name)}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print(f'Saved: {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate saved confusion matrices without recomputing model outputs.'
    )
    parser.add_argument(
        '--model-aggregate-only',
        action='store_true',
        help='Only generate DENFormer model-level aggregate confusion matrices.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(paths.logs_dir)
    out_dir = base_dir / 'aggregate_confusion_matrices'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_aggregate_only:
        for group_name, models in COMPARISON_GROUPS.items():
            for experiment in EXPERIMENTS:
                aggregate_confusion_matrices(
                    base_dir=base_dir,
                    out_dir=out_dir,
                    experiment=experiment,
                    group_name=group_name,
                    models=models,
                )

    model_aggregate_dir = out_dir / 'model_aggregate'
    for model_name in DENFORMER_VARIANTS:
        aggregate_denformer_model_confmats(
            base_dir=base_dir,
            out_dir=model_aggregate_dir,
            model_name=model_name,
            num_classes=4,
        )


if __name__ == '__main__':
    main()
