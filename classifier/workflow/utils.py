import os
import re
import ast
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd

import classifier.config

SPLIT_ALIASES = {
    'train': 'train',
    'training': 'train',
    'tr': 'train',
    'val': 'val',
    'valid': 'val',
    'validation': 'val',
    'dev': 'val',
    'test': 'test',
    'te': 'test',
}

MODEL_TYPES = {'denformer', 'longformer', 'performer', 'ffnn', 'logreg'}
ATTN_MODEL_TYPES = {'denformer_attn'}
POOLINGS = {'first', 'mean', 'max'}


def normalize_split_name(value):
    value = str(value).strip().lower()
    if value not in SPLIT_ALIASES:
        raise ValueError(f'Unknown split value {value!r}. Expected train, val or test.')
    return SPLIT_ALIASES[value]


def parse_index_cell(value):
    """Parse a CSV/JSON cell that may contain one index or a list of indices."""
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
        parsed = ast.literal_eval(text)
        return [int(x) for x in parsed]

    for sep in [',', ';', ' ']:
        if sep in text:
            return [int(x) for x in text.replace(';', ',').replace(' ', ',').split(',') if x != '']

    return [int(text)]


def _fold_sort_key(fold_id):
    try:
        return int(fold_id)
    except (TypeError, ValueError):
        return str(fold_id)


def load_split_file(split_file):
    """
    Return a list of dicts:
      [{'fold_id': '0', 'train_idx': np.ndarray, 'val_idx': ..., 'test_idx': ...}, ...]
    """
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f'Split file not found: {split_file}')

    suffix = split_file.suffix.lower()
    if suffix == '.json':
        with open(split_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if 'folds' in data:
                data = data['folds']
            else:
                data = [data]
        folds = []
        for pos, item in enumerate(data):
            fold_id = str(item.get('fold', item.get('fold_id', pos)))
            folds.append({
                'fold_id': fold_id,
                'train_idx': np.array(parse_index_cell(item['train_idx']), dtype=np.int64),
                'val_idx': np.array(parse_index_cell(item['val_idx']), dtype=np.int64),
                'test_idx': np.array(parse_index_cell(item.get('test_idx', [])), dtype=np.int64),
            })
        return folds

    df = pd.read_csv(split_file)
    df.columns = [c.strip() for c in df.columns]
    lower_to_original = {c.lower(): c for c in df.columns}

    split_col = lower_to_original.get('split') or lower_to_original.get('set') or lower_to_original.get('partition')
    index_col = (
        lower_to_original.get('index') or lower_to_original.get('idx') or
        lower_to_original.get('sample_idx') or lower_to_original.get('sample_index')
    )
    fold_col = lower_to_original.get('fold') or lower_to_original.get('fold_id') or lower_to_original.get('cv_fold')

    if split_col and index_col:
        work = df.copy()
        work['_split_norm'] = work[split_col].map(normalize_split_name)
        work['_fold_norm'] = work[fold_col].astype(str) if fold_col else 'holdout'

        folds = []
        for fold_id, fold_df in work.groupby('_fold_norm', sort=False):
            split_to_idx = {'train': [], 'val': [], 'test': []}
            for _, row in fold_df.iterrows():
                split_to_idx[row['_split_norm']].extend(parse_index_cell(row[index_col]))
            folds.append({
                'fold_id': str(fold_id),
                'train_idx': np.array(split_to_idx['train'], dtype=np.int64),
                'val_idx': np.array(split_to_idx['val'], dtype=np.int64),
                'test_idx': np.array(split_to_idx['test'], dtype=np.int64),
            })
        return sorted(folds, key=lambda x: _fold_sort_key(x['fold_id']))

    train_col = lower_to_original.get('train_idx') or lower_to_original.get('train_indices')
    val_col = lower_to_original.get('val_idx') or lower_to_original.get('valid_idx') or lower_to_original.get('validation_idx')
    test_col = lower_to_original.get('test_idx') or lower_to_original.get('test_indices')

    if train_col and val_col:
        folds = []
        for pos, row in df.iterrows():
            fold_id = str(row[fold_col]) if fold_col else ('holdout' if len(df) == 1 else str(pos))
            folds.append({
                'fold_id': fold_id,
                'train_idx': np.array(parse_index_cell(row[train_col]), dtype=np.int64),
                'val_idx': np.array(parse_index_cell(row[val_col]), dtype=np.int64),
                'test_idx': np.array(parse_index_cell(row[test_col]), dtype=np.int64) if test_col else np.array([], dtype=np.int64),
            })
        return folds

    raise ValueError(
        'Unsupported split file format. Use either long CSV columns '
        '(index, split[, fold]) or wide CSV/JSON columns '
        '(train_idx, val_idx[, test_idx, fold]).'
    )


def validate_indices(fold, n_samples):
    for name in ['train_idx', 'val_idx', 'test_idx']:
        idx = fold[name]
        if len(idx) == 0 and name != 'test_idx':
            raise ValueError(f"Fold {fold['fold_id']} has empty {name}.")
        if len(idx) and (idx.min() < 0 or idx.max() >= n_samples):
            raise ValueError(
                f"Fold {fold['fold_id']} has out-of-range values in {name}: "
                f"min={idx.min()}, max={idx.max()}, n_samples={n_samples}"
            )

    train = set(fold['train_idx'].tolist())
    val = set(fold['val_idx'].tolist())
    test = set(fold['test_idx'].tolist())
    overlaps = {
        'train-val': len(train & val),
        'train-test': len(train & test),
        'val-test': len(val & test),
    }
    bad = {k: v for k, v in overlaps.items() if v > 0}
    if bad:
        raise ValueError(f"Fold {fold['fold_id']} has overlapping splits: {bad}")


def get_latest_model_path(model_dir: str) -> str:
    """Find latest checkpoint in a fold directory."""
    if not os.path.isdir(model_dir):
        return None

    pattern = re.compile(r"model-(?:k\d+|ohe)_fold-(.+?)_epoch-(\d+)_(\d+)\.pt")
    latest_epoch = -1
    latest_ts = -1
    latest_file = None

    for fname in os.listdir(model_dir):
        m = pattern.match(fname)
        if not m:
            continue
        epoch = int(m.group(2))
        ts = int(m.group(3))
        if (epoch > latest_epoch) or (epoch == latest_epoch and ts > latest_ts):
            latest_epoch = epoch
            latest_ts = ts
            latest_file = os.path.join(model_dir, fname)
    return latest_file


# -----------------------------------------------------------------------------
# Shared run helpers for train/inference dataset
# -----------------------------------------------------------------------------

def safe_name(value):
    value = str(value)
    value = re.sub(r'[<>:"/\\|?*]', '_', value)
    value = value.replace(' ', '_')
    return value


def _extract_cli_value(argv, flag_name, default_value, allowed_values=None):
    cleaned = [argv[0]]
    value = default_value
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == flag_name:
            if i + 1 >= len(argv):
                allowed = ', '.join(sorted(allowed_values)) if allowed_values else 'a value'
                raise ValueError(f'{flag_name} requires one of: {allowed}')
            value = argv[i + 1]
            i += 2
            continue
        if arg.startswith(flag_name + '='):
            value = arg.split('=', 1)[1]
            i += 1
            continue
        cleaned.append(arg)
        i += 1

    if allowed_values is not None and value not in allowed_values:
        raise ValueError(f'Unsupported {flag_name}={value!r}. Choose one of: {sorted(allowed_values)}')
    return value, cleaned


def parse_run_args(get_args_func, argv=None, allow_attn=False):
    """
    Parse custom args unknown to classifier.utils.get_args(), then call get_args_func.

    Supported extra args:
      --pooling first|mean|max
      --model_type denformer|longformer|performer|ffnn|logreg   (train/inference)
      --model_type denformer_attn                   (attention inference)
    """
    if argv is None:
        argv = sys.argv

    allowed_model_types = ATTN_MODEL_TYPES if allow_attn else MODEL_TYPES
    pooling, cleaned = _extract_cli_value(argv, '--pooling', 'first', POOLINGS)
    model_type, cleaned = _extract_cli_value(cleaned, '--model_type', 'denformer_attn' if allow_attn else 'denformer', allowed_model_types)

    original_argv = sys.argv
    try:
        sys.argv = cleaned
        args = get_args_func(from_file=True)
    finally:
        sys.argv = original_argv

    args.pooling = pooling
    args.model_type = model_type

    if args.model_type in {'longformer', 'performer'} and args.pooling != 'mean':
        print(f"WARNING: forcing pooling='mean' for model_type={args.model_type} (single configuration).")
        args.pooling = 'mean'

    if args.model_type == 'denformer_attn':
        # Attention extraction should load checkpoints saved by denformer runs.
        args.train_model_type = 'denformer'

    return args


def resolve_k_type_and_emb_dim(args, emb_dim, emb_dim_ohe):
    if args.one_hot:
        return 'ohe', emb_dim_ohe
    return f'k{args.k}', emb_dim


def get_run_suffix(args, split_file):
    split_name = Path(split_file).stem
    return f"_{args.run_name}" if getattr(args, 'run_name', None) else f"_{split_name}"


def model_dir_name(model_type, pooling):
    if model_type == 'denformer':
        return f'denformer_{pooling}'
    if model_type == 'denformer_attn':
        return f'denformer_{pooling}'
    return model_type


def build_model_dir(output_dir, model_type, pooling, k_type, run_suffix, epochs):
    return os.path.join(output_dir, model_dir_name(model_type, pooling), f'{k_type}{run_suffix}_e{epochs}')


# ==============================
# XAI OUTPUT PATHS
# ==============================
AGGREGATE_XAI_DIR_NAME = 'aggregate_xai'
XAI_EMBEDDING_PLOTS_DIR_NAME = 'embeddings_xai'
DATA_XAI_EMBEDDINGS_DIR_NAME = 'data_xai_embeddings'

# Backward-compatible name used by older scripts. It now refers to the plot
# subdirectory under aggregate_xai, not to a top-level logs/xai_embeddings dir.
XAI_EMBEDDINGS_DIR_NAME = XAI_EMBEDDING_PLOTS_DIR_NAME


def build_aggregate_xai_root(logs_dir):
    """Base directory for derived aggregate XAI outputs, outside model folders."""
    return os.path.join(str(logs_dir), AGGREGATE_XAI_DIR_NAME)


def build_xai_embeddings_root(logs_dir):
    """Base directory for derived XAI embedding plots."""
    return os.path.join(str(logs_dir), AGGREGATE_XAI_DIR_NAME, XAI_EMBEDDING_PLOTS_DIR_NAME)


def build_xai_embeddings_data_root(logs_dir):
    """Base directory for raw extracted XAI embedding data (.npz)."""
    return os.path.join(str(logs_dir), DATA_XAI_EMBEDDINGS_DIR_NAME)


def build_xai_embeddings_data_dir(logs_dir, model_type, pooling, k_type, run_suffix, epochs):
    """Run-level directory for raw extracted XAI embedding data (.npz)."""
    return os.path.join(
        build_xai_embeddings_data_root(logs_dir),
        model_dir_name(model_type, pooling),
        f'{k_type}{run_suffix}_e{epochs}',
    )


def build_xai_output_dir(logs_dir, xai_kind, model_type, pooling, k_type, run_suffix, epochs):
    """
    Directory for derived XAI outputs.

    Raw XAI embedding arrays are stored under logs/data_xai_embeddings.
    Derived plots/tables are stored under logs/aggregate_xai/<xai_kind>.
    """
    return os.path.join(
        build_aggregate_xai_root(logs_dir),
        xai_kind,
        model_dir_name(model_type, pooling),
        f'{k_type}{run_suffix}_e{epochs}',
    )


def load_and_validate_folds(split_file, n_samples, fold_id=None):
    folds = load_split_file(split_file)
    if fold_id is not None:
        requested = str(fold_id)
        folds = [fold for fold in folds if str(fold['fold_id']) == requested]
        if not folds:
            raise ValueError(f"Fold {requested!r} not found in {split_file}")
    for fold in folds:
        validate_indices(fold, n_samples)
    return folds


def build_classifier_model(args, emb_dim, config, device, attn=False):
    """
    Build denformer/longformer/performer classifiers with a common script API.

    Expected package files:
      classifier.model.denformer
      classifier.model.denformer_attn
      classifier.model.longformer
      classifier.model.performer
    """
    model_type = getattr(args, 'model_type', 'denformer_attn' if attn else 'denformer')

    if attn or model_type == 'denformer_attn':
        from classifier.model.denformer_attn import TransformerClassifier
        model = TransformerClassifier(
            emb_dim, config.D_MODEL, config.NHEAD, config.FF_DIM, config.NUM_LAYERS, classifier.config.NUM_CLASSES,
            max_len=config.MAX_LEN, chunk_size=config.CHUNK_SIZE, dropout=config.DROPOUT,
            pooling=args.pooling,
        )
        return model.to(device)

    if model_type == 'denformer':
        from classifier.model.denformer import TransformerClassifier
        model = TransformerClassifier(
            emb_dim, config.D_MODEL, config.NHEAD, config.FF_DIM, config.NUM_LAYERS, classifier.config.NUM_CLASSES,
            max_len=config.MAX_LEN, chunk_size=config.CHUNK_SIZE, dropout=config.DROPOUT,
            pooling=args.pooling,
        )
        return model.to(device)

    if model_type == 'longformer':
        from classifier.model.longformer import LongformerClassifier
        window_size = getattr(config, 'LONGFORMER_ATTENTION_WINDOW', 512)
        model = LongformerClassifier(
            emb_dim, config.D_MODEL, config.NHEAD, config.FF_DIM, config.NUM_LAYERS, classifier.config.NUM_CLASSES,
            max_len=config.MAX_LEN, attention_window=window_size, dropout=config.DROPOUT,
            pooling='mean',
        )
        return model.to(device)


    if model_type == 'ffnn':
        from classifier.model.ffnn import FFNNClassifier
        model = FFNNClassifier(
            emb_dim, config.D_MODEL, config.NHEAD, config.FF_DIM, config.NUM_LAYERS, classifier.config.NUM_CLASSES,
            max_len=config.MAX_LEN, dropout=config.DROPOUT,
            pooling=args.pooling,
        )
        return model.to(device)

    if model_type == 'logreg':
        from classifier.model.logreg import LogisticRegressionClassifier
        model = LogisticRegressionClassifier(
            emb_dim, config.D_MODEL, config.NHEAD, config.FF_DIM, config.NUM_LAYERS, classifier.config.NUM_CLASSES,
            max_len=config.MAX_LEN, dropout=0.0,
            pooling=args.pooling,
        )
        return model.to(device)

    if model_type == 'performer':
        from classifier.model.performer import PerformerClassifier
        nb_features = getattr(config, 'PERFORMER_NB_FEATURES', 256)
        model = PerformerClassifier(
            emb_dim, config.D_MODEL, config.NHEAD, config.FF_DIM, config.NUM_LAYERS, classifier.config.NUM_CLASSES,
            max_len=config.MAX_LEN, nb_features=nb_features, dropout=config.DROPOUT,
            pooling='mean',
        )
        return model.to(device)

    raise ValueError(f'Unsupported model_type={model_type!r}')


# -----------------------------------------------------------------------------
# Shared reporting / plotting helpers
# -----------------------------------------------------------------------------

def class_names(num_classes):
    return [f'DENV{i + 1}' for i in range(num_classes)]


def normalize_cm(cm, normalize):
    cm = np.asarray(cm)

    if normalize == 'row':
        row_sums = cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        return plot_cm, '.2f', 1.0

    if normalize == 'all':
        total = cm.sum()
        plot_cm = cm / total if total > 0 else np.zeros_like(cm, dtype=float)
        return plot_cm, '.3f', max(1e-12, float(plot_cm.max()))

    return cm, 'd', max(1, int(cm.max()))


def format_cell_value(value, raw_value, normalize):
    if normalize == 'row':
        return f'{value:.2f}\n({int(raw_value)})'
    if normalize == 'all':
        return f'{value:.3f}\n({int(raw_value)})'
    return str(int(raw_value))


def plot_confusion_matrix(
    cm,
    title,
    save_path,
    num_classes,
    normalize='none',
    accuracy=None,
):
    cm = np.asarray(cm)
    plot_cm, fmt, vmax = normalize_cm(cm, normalize)
    labels = class_names(num_classes)

    fig, ax = plt.subplots(figsize=(7.5, 6.8), constrained_layout=True)
    im = ax.imshow(plot_cm, cmap='Blues', vmin=0, vmax=vmax)

    acc_text = f'\nAccuracy ({accuracy * 100:.2f} %)' if accuracy is not None else ''
    ax.set_title(f'{title}{acc_text}', fontsize=20, pad=18)
    ax.set_xlabel('Predicted', fontsize=18)
    ax.set_ylabel('True', fontsize=18)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', labelsize=16)

    threshold = vmax * 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = plot_cm[i, j]
            raw_value = cm[i, j]
            text_color = 'white' if value > threshold else 'black'
            text = format(value, fmt) if normalize == 'none' else format_cell_value(
                value,
                raw_value,
                normalize,
            )
            ax.text(
                j,
                i,
                text,
                ha='center',
                va='center',
                color=text_color,
                fontsize=15,
                fontweight='bold',
                linespacing=1.2,
            )

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def plot_confusion_matrices_hconcat(
    confmats,
    titles,
    save_path,
    num_classes,
    normalize='row',
    accuracies=None,
):
    if not confmats:
        return

    if accuracies is None:
        accuracies = [None] * len(confmats)

    labels = class_names(num_classes)
    processed = [normalize_cm(cm, normalize)[0] for cm in confmats]

    n = len(processed)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(max(7.0 * n, 10), 8.4),
        squeeze=False,
    )
    axes = axes[0]

    if normalize == 'row':
        vmax = 1.0
    elif normalize == 'all':
        vmax = max(max(1e-12, float(cm.max())) for cm in processed)
    else:
        vmax = max(max(1, int(cm.max())) for cm in processed)

    im = None

    for ax, cm_plot, cm_raw, title, accuracy in zip(
        axes,
        processed,
        confmats,
        titles,
        accuracies,
    ):
        cm_raw = np.asarray(cm_raw)
        im = ax.imshow(cm_plot, cmap='Blues', vmin=0, vmax=vmax)

        acc_text = f'\nAccuracy ({accuracy * 100:.2f} %)' if accuracy is not None else ''
        ax.set_title(f'{title}{acc_text}', fontsize=18, pad=26)
        ax.set_xlabel('Predicted', fontsize=16, labelpad=12)
        ax.set_ylabel('True', fontsize=16, labelpad=12)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.tick_params(axis='both', labelsize=14)

        threshold = vmax * 0.5

        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                value = cm_plot[i, j]
                raw_value = cm_raw[i, j]
                text_color = 'white' if value > threshold else 'black'
                text = format_cell_value(value, raw_value, normalize)
                ax.text(
                    j,
                    i,
                    text,
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=12,
                    fontweight='bold',
                    linespacing=1.25,
                )

    fig.subplots_adjust(
        left=0.035,
        right=0.965,
        top=0.82,
        bottom=0.20,
        wspace=0.28,
    )
    fig.colorbar(im, ax=axes.tolist(), shrink=0.72, pad=0.025)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


def compute_subset_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'classification_report': classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_subset_reports(
    fold_metrics_dir,
    fold_id,
    subset,
    y_true,
    y_pred,
    num_classes,
    title_prefix=None,
):
    subset_result = compute_subset_metrics(y_true, y_pred)
    accuracy = subset_result['accuracy']

    cm = None

    if subset == 'test':
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
        )
        subset_result['confusion_matrix'] = cm.tolist()

        title = f'{fold_id} - test'
        if title_prefix:
            title = f'{title_prefix} | {title}'

        for normalize, suffix in [
            ('none', 'counts'),
            ('row', 'row_normalized'),
            ('all', 'all_normalized'),
        ]:
            path = os.path.join(
                fold_metrics_dir,
                f'confusion_matrix_test_{suffix}.png',
            )
            plot_confusion_matrix(
                cm=cm,
                title=title,
                save_path=path,
                normalize=normalize,
                accuracy=accuracy,
                num_classes=num_classes,
            )
            subset_result[f'confusion_matrix_{suffix}_png'] = path

    return subset_result, cm
