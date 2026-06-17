import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

from classifier.config import DEFAULT_EPOCHS


def plot_confusion_matrix(y_true, y_pred, class_names, out_path=None, normalize=False):
    """
    Plot a confusion matrix with better readability.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        out_path: Path to save figure (if None, shows instead)
        normalize: If True, show percentages (with colorbar);
                   if False, show raw counts (no colorbar).
    """
    set_plot_style(8, 8, font_factor=1.5)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # avoid div/0
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
        ax.set_title("Normalized Confusion Matrix")
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title("Confusion Matrix")

    for text in disp.text_.ravel():  # disp.text_ contains the text objects
        text.set_fontsize(14)

    # Improve layout
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.grid(False)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Confusion matrix saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def set_plot_style(fx, fy, font_factor=1., line_factor=1.):
    """Standardize matplotlib rcParams for readability."""
    plt.rcParams.update({
        "figure.figsize": (fx, fy),
        "font.size": 12*font_factor,
        "axes.titlesize": 14*font_factor,
        "axes.labelsize": 12*font_factor,
        "xtick.labelsize": 11*font_factor,
        "ytick.labelsize": 11*font_factor,
        "legend.fontsize": 11*font_factor,
        "lines.linewidth": 2*line_factor,
        "lines.markersize": 6*line_factor,
        "grid.alpha": 0.7
    })


def plot_roc_curves(y_true, y_score, class_names, out_path=None):
    set_plot_style(7, 7)
    n_classes = len(class_names)
    fig, ax = plt.subplots()

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-Class ROC Curves")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"ROC curves saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_pr_curves(y_true, y_score, class_names, out_path=None):
    set_plot_style(7, 7)
    n_classes = len(class_names)
    fig, ax = plt.subplots()

    # Use matplotlib default cycle to keep color consistent
    colors = plt.cm.tab10.colors if n_classes <= 10 else plt.cm.tab20.colors

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_score[:, i])
        ap = average_precision_score(y_true == i, y_score[:, i])

        color = colors[i % len(colors)]

        # Plot PR curve
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{class_names[i]} (AP = {ap:.2f})")

        # Add random chance reference line with same color
        chance_rate = np.mean(y_true == i)
        ax.hlines(
            y=chance_rate,
            xmin=0, xmax=1,
            colors=[color],
            linestyles="--",
            lw=1.2
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision-Recall Curves")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"PR curves saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def get_args(from_file=False):
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument("--k", type=int, default=3, help="k-mer length used for extracting embeddings with dna2vec")
    parser.add_argument("--one-hot", action="store_true", help="Use one-hot encoding instead of k-mer embeddings")
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction for validation split")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for test split")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    if from_file:
        parser.add_argument('--split_file', required=True, help='CSV/JSON file containing precomputed train/val/test splits.')
        parser.add_argument('--fold', default=None, help='Optional fold id to run. If omitted, all folds in split_file are trained.')
        parser.add_argument('--run_name', default=None, help='Optional name appended to output folders.')
        parser.add_argument('--save_test_predictions', action='store_true', help='Save per-sample predictions for the test split.')

    return parser.parse_args()


def print_args(args):
    """
    Print all arguments from an argparse.Namespace in a nice format.
    """
    print("\nArguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")


def save_classification_metrics(all_labels, all_preds, all_probs, num_classes, metrics_dir, subset, class_dict=None):
    """
    Compute and save classification metrics in tidy format:
    - Per-class precision, recall, specificity, AUROC, AUPR
    - Macro averages
    - Global accuracy
    """
    if class_dict is None:
        class_dict = {}

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # Per-class precision and recall
    precisions = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # Specificity per class
    cm = confusion_matrix(all_labels, all_preds)
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)

    # AUROC and AUPR per class
    aurocs, auprs = [], []
    for i in range(num_classes):
        y_true = (all_labels == i).astype(int)
        y_prob = all_probs[:, i]
        try:
            aurocs.append(roc_auc_score(y_true, y_prob))
        except ValueError:
            aurocs.append(np.nan)
        try:
            auprs.append(average_precision_score(y_true, y_prob))
        except ValueError:
            auprs.append(np.nan)

    # Build per-class table using CLASS_DICT
    df_per_class = pd.DataFrame({
        "class": [class_dict.get(i, f"Class {i}") for i in range(num_classes)],
        "precision": precisions,
        "recall": recalls,
        "specificity": specificities,
        "auroc": aurocs,
        "aupr": auprs
    })

    # Macro averages
    df_macro = pd.DataFrame([{
        "class": "Macro Avg",
        "precision": np.nanmean(precisions),
        "recall": np.nanmean(recalls),
        "specificity": np.nanmean(specificities),
        "auroc": np.nanmean([x for x in aurocs if not np.isnan(x)]),
        "aupr": np.nanmean([x for x in auprs if not np.isnan(x)])
    }])

    # Accuracy row
    df_acc = pd.DataFrame([{
        "class": "Overall Accuracy",
        "precision": np.nan,
        "recall": np.nan,
        "specificity": np.nan,
        "auroc": np.nan,
        "aupr": np.nan,
        "accuracy": acc
    }])

    # Combine all
    df_out = pd.concat([df_per_class, df_macro, df_acc], ignore_index=True)

    # Save
    csv_path = os.path.join(metrics_dir, f"classification_metrics_{subset}.csv")
    df_out.to_csv(csv_path, index=False, sep=';')
    print(f"Metrics saved to {csv_path}")


def plot_training_curves(train_stats, metrics_dir):
    """
    Generate and save training curves for accuracy and loss.

    Args:
        train_stats (list of dict): Each dict should contain
            'epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss'.
        metrics_dir (str): Directory to save plots.
    """
    set_plot_style(8, 6)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(train_stats)

    # === Accuracy plot ===
    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(df["epoch"])  # ensure integer ticks only
    plt.tight_layout()

    acc_png = os.path.join(metrics_dir, "train_trend_accuracy.png")
    plt.savefig(acc_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved accuracy plot to {acc_png}")

    # === Loss plot ===
    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(df["epoch"])  # ensure integer ticks only
    plt.tight_layout()

    loss_png = os.path.join(metrics_dir, "train_trend_loss.png")
    plt.savefig(loss_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved loss plot to {loss_png}")


DENV_PALETTE = {
    0: "#66c2a5",  # DENV-1
    1: "#fc8d62",  # DENV-2
    2: "#8da0cb",  # DENV-3
    3: "#e78ac3",  # DENV-4
}


def _embedding_colors(y):
    return [DENV_PALETTE.get(int(label), "#808080") for label in y]


def _embedding_legend_handles(class_names, labels=None):
    import matplotlib.lines as mlines
    if labels is None:
        labels = sorted(DENV_PALETTE.keys())
    handles = []
    for label in labels:
        label_int = int(label)
        name = class_names[label_int] if label_int < len(class_names) else f"Class {label_int}"
        handles.append(
            mlines.Line2D(
                [], [],
                marker='o',
                linestyle='None',
                markersize=8,
                markerfacecolor=DENV_PALETTE.get(label_int, "#808080"),
                markeredgecolor=DENV_PALETTE.get(label_int, "#808080"),
                label=name,
            )
        )
    return handles


def plot_embedding(X_2d, y, label_names, title, filename, label_name="Dim"):
    labels = sorted(np.unique(y).astype(int).tolist())
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=_embedding_colors(y),
        s=14,
        alpha=0.82,
        edgecolors='none',
    )
    ax.legend(
        handles=_embedding_legend_handles(label_names, labels),
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=14,
        borderpad=0.8,
        handletextpad=0.6,
    )
    ax.set_xlabel(f"{label_name}-1", fontsize=18)
    ax.set_ylabel(f"{label_name}-2", fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(False)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_tsne_umap(X_pca, X_tsne, X_umap, y, class_names, title, filename):
    labels = sorted(np.unique(y).astype(int).tolist())
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=False)

    titles = ["PCA", "t-SNE", "UMAP"]
    embeddings = [X_pca, X_tsne, X_umap]

    for ax, emb, t in zip(axes, embeddings, titles):
        ax.scatter(
            emb[:, 0], emb[:, 1],
            c=_embedding_colors(y),
            s=12,
            alpha=0.78,
            edgecolors="none",
        )
        ax.set_xlabel(f"{t}-1", fontsize=16)
        ax.set_ylabel(f"{t}-2", fontsize=16)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_title(t, fontsize=17, pad=8)
        ax.grid(False)

    legend_handles = _embedding_legend_handles(class_names, labels)
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.875, 0.5),
        fontsize=13,
        frameon=True,
        borderpad=0.7,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.055, right=0.855, bottom=0.14, top=0.90, wspace=0.30)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention(attn, save_path, layer=0, head=0, chunk=0, token_limit=200):
    """
    attn: list[num_layers] -> list[num_chunks] -> Tensor[B, nhead, Lc, Lc]
    """
    attn_map = attn[layer][chunk][0, head].cpu().numpy()  # take first sample in batch
    if token_limit is not None and attn_map.shape[0] > token_limit:
        attn_map = attn_map[:token_limit, :token_limit]  # crop for plotting

    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_map, cmap="viridis")
    plt.title(f"Layer {layer}, Head {head}, Chunk {chunk}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_latest_model_path(model_dir: str) -> str:
    """
    Find the latest saved model file in `model_dir` following the pattern:
    model_<epoch>_<timestamp>.pt
    Returns the full path to the latest file, or None if not found.
    """
    # pattern = re.compile(r"model-k(\d+)_(\d+)_(\d+)\.pt")
    pattern = re.compile(r"model-(?:k(\d+)|ohe)_(\d+)_(\d+)\.pt")

    latest_epoch = -1
    latest_ts = -1
    latest_file = None

    for fname in os.listdir(model_dir):
        m = pattern.match(fname)
        if m:
            if m.group(1) is not None:
                k = int(m.group(1))
            epoch = int(m.group(2))
            ts = int(m.group(3))
            # print(f"k={k}, epoch={epoch}, ts={ts}")
            # Sort primarily by epoch, then by timestamp
            if (epoch > latest_epoch) or (epoch == latest_epoch and ts > latest_ts):
                latest_epoch = epoch
                latest_ts = ts
                latest_file = os.path.join(model_dir, fname)

    return latest_file
