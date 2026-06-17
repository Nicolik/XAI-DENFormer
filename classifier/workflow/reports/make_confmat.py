try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

from pathlib import Path
import cv2
import numpy as np
import paths


MODELS = [
    "denformer_first",
    "denformer_max",
    "denformer_mean",
    "longformer",
    "performer",
    "ffnn",
    "logreg",
]

DENFORMER_VARIANTS = [
    "denformer_first",
    "denformer_max",
    "denformer_mean",
]

SELECTED_DENFORMER = "denformer_mean"

OTHER_MODELS = [
    model for model in MODELS
    if model not in DENFORMER_VARIANTS
]

COMPARISON_GROUPS = {
    "all_models": MODELS,
    "denformer_variants": DENFORMER_VARIANTS,
    "selected_denformer_vs_others": [SELECTED_DENFORMER] + OTHER_MODELS,
}

EXPERIMENTS = [
    "ohe_cdhit_e100",
    "ohe_continent_e100",
    "ohe_timebin_e100",
]

CM_FILES = [
    "confusion_matrices_test_hconcat_counts.png",
    "confusion_matrices_test_hconcat_all_normalized.png",
    "confusion_matrices_test_hconcat_row_normalized.png",
]

LABEL_WIDTH = 900


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
            img_path = base_dir / model / experiment / "metrics" / cm_file

            if not img_path.exists():
                print(f"Missing: {img_path}")
                continue

            img = cv2.imread(str(img_path))

            if img is None:
                print(f"Could not read: {img_path}")
                continue

            img = add_label(img, model)
            rows.append(img)

        if not rows:
            print(f"No images found for {group_name} - {experiment} - {cm_file}")
            continue

        target_width = max(row.shape[1] for row in rows)
        rows = [resize_to_width(row, target_width) for row in rows]

        aggregated = cv2.vconcat(rows)

        out_path = experiment_out_dir / cm_file.replace(
            "confusion_matrices_test_hconcat",
            f"confusion_matrices_test_{group_name}_vconcat",
        )

        cv2.imwrite(str(out_path), aggregated)
        print(f"Saved: {out_path}")


def main():
    base_dir = Path(paths.logs_dir)
    out_dir = base_dir / "aggregate_confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_name, models in COMPARISON_GROUPS.items():
        for experiment in EXPERIMENTS:
            aggregate_confusion_matrices(
                base_dir=base_dir,
                out_dir=out_dir,
                experiment=experiment,
                group_name=group_name,
                models=models,
            )


if __name__ == "__main__":
    main()
