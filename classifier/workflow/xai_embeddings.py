import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import paths
from classifier.config import CLASS_DICT
from classifier.workflow import config
from classifier.workflow.utils import (
    build_model_dir,
    build_xai_embeddings_data_dir,
    build_xai_output_dir,
    get_run_suffix,
    load_and_validate_folds,
    parse_run_args,
    resolve_k_type_and_emb_dim,
    safe_name,
)
from classifier.utils import get_args, plot_embedding, plot_pca_tsne_umap, print_args
from classifier.utils_data import get_dataset


DATASET_AGGREGATE_DIR_NAME = 'output-aggregate-dataset'
DATASET_AGGREGATE_SUBSET = 'dataset_test'


def parse_args():
    return parse_run_args(get_args, allow_attn=False)


def _umap_transform(X):
    try:
        from umap import UMAP
    except ImportError:
        print('[WARN] umap-learn is not installed; skipping UMAP.')
        return None
    return UMAP(n_components=2, random_state=42).fit_transform(X)


def _safe_tsne_perplexity(n):
    if n <= 3:
        return None
    return min(30, max(2, (n - 1) // 3))


def run_projection_file(npz_path, subset, title_suffix=None, output_dir=None):
    print(f'Loading {npz_path}')
    if output_dir is None:
        output_dir = os.path.dirname(npz_path)
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    X = data['embeddings']
    y = data['labels']
    print(f'Loaded shapes: X={X.shape}, y={y.shape}')

    if len(X) < 2:
        print('[WARN] Too few samples for projection; skipping.')
        return

    class_names = [CLASS_DICT[key] for key in sorted(CLASS_DICT.keys())]
    title_suffix = title_suffix or subset

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_csv = os.path.join(output_dir, f'pca_embeddings_{subset}.csv')
    pd.DataFrame({'x': X_pca[:, 0], 'y': X_pca[:, 1], 'label': y}).to_csv(pca_csv, index=False)
    pca_png = os.path.join(output_dir, f'pca_embeddings_{subset}.png')
    plot_embedding(X_pca, y, class_names, f'PCA of embeddings - {title_suffix}', pca_png, label_name='PCA')

    X_tsne = None
    perplexity = _safe_tsne_perplexity(len(X))
    if perplexity is not None:
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)
        tsne_csv = os.path.join(output_dir, f'tsne_embeddings_{subset}.csv')
        pd.DataFrame({'x': X_tsne[:, 0], 'y': X_tsne[:, 1], 'label': y}).to_csv(tsne_csv, index=False)
        tsne_png = os.path.join(output_dir, f'tsne_embeddings_{subset}.png')
        plot_embedding(X_tsne, y, class_names, f't-SNE of embeddings - {title_suffix}', tsne_png, label_name='t-SNE')
    else:
        print('[WARN] Too few samples for t-SNE; skipping.')

    X_umap = _umap_transform(X)
    if X_umap is not None:
        umap_csv = os.path.join(output_dir, f'umap_embeddings_{subset}.csv')
        pd.DataFrame({'x': X_umap[:, 0], 'y': X_umap[:, 1], 'label': y}).to_csv(umap_csv, index=False)
        umap_png = os.path.join(output_dir, f'umap_embeddings_{subset}.png')
        plot_embedding(X_umap, y, class_names, f'UMAP of embeddings - {title_suffix}', umap_png, label_name='UMAP')

    if X_tsne is not None and X_umap is not None:
        full_png = os.path.join(output_dir, f'full_embeddings_{subset}.png')
        plot_pca_tsne_umap(X_pca, X_tsne, X_umap, y, class_names, f'embeddings - {title_suffix}', full_png)


def _resolve_fold_embedding_dir(model_dir, xai_data_root, fold_dir_name):
    preferred = os.path.join(xai_data_root, fold_dir_name)
    if os.path.isdir(preferred):
        return preferred

    legacy = os.path.join(model_dir, 'xai', fold_dir_name)
    if os.path.isdir(legacy):
        print(f'[WARN] Using legacy embedding data dir: {legacy}')
        return legacy

    return preferred


def run_fold(args, fold, model_dir, xai_data_root, xai_plot_root):
    fold_id_safe = safe_name(fold['fold_id'])
    fold_dir_name = f'split_{fold_id_safe}'
    source_dir = _resolve_fold_embedding_dir(model_dir, xai_data_root, fold_dir_name)

    if not os.path.isdir(source_dir):
        print(f'[WARN] Missing XAI embedding data dir, skipping fold {fold_id_safe}: {source_dir}')
        return None

    print(f'\n==== XAI EMBEDDING PLOTS | FOLD {fold["fold_id"]} ====')
    ran = []
    out_dir = os.path.join(xai_plot_root, fold_dir_name)
    for subset in ['train', 'val', 'test']:
        npz_path = os.path.join(source_dir, f'embeddings_{subset}.npz')
        if not os.path.exists(npz_path):
            continue
        run_projection_file(npz_path, subset, title_suffix=f'fold {fold["fold_id"]} {subset}', output_dir=out_dir)
        ran.append(subset)
    return {
        'fold': fold['fold_id'],
        'source_embedding_dir': source_dir,
        'output_dir': out_dir,
        'subsets': ','.join(ran),
    }

def _load_embedding_npz(npz_path):
    data = np.load(npz_path)
    X = data['embeddings']
    y = data['labels']
    if 'indices' in data.files:
        indices = data['indices']
    else:
        indices = np.arange(len(y), dtype=np.int64)
        print(f'[WARN] No indices found in {npz_path}; duplicate removal will use local row numbers only.')
    return X, y, indices


def build_dataset_level_npz(model_dir, xai_data_root, subset='test'):
    pattern = os.path.join(xai_data_root, 'split_*', f'embeddings_{subset}.npz')
    npz_paths = sorted(glob.glob(pattern))

    if not npz_paths:
        legacy_pattern = os.path.join(model_dir, 'xai', 'split_*', f'embeddings_{subset}.npz')
        npz_paths = sorted(glob.glob(legacy_pattern))
        if npz_paths:
            print(f'[WARN] Using legacy fold-level embeddings for dataset aggregation: {legacy_pattern}')

    if not npz_paths:
        print(f'[WARN] No fold-level embeddings found for dataset aggregation: {pattern}')
        return None

    print(f'\n==== DATASET-LEVEL XAI EMBEDDING AGGREGATION | subset={subset} ====')
    X_list = []
    y_list = []
    idx_list = []
    source_list = []

    for npz_path in npz_paths:
        X, y, indices = _load_embedding_npz(npz_path)
        if len(X) == 0:
            print(f'[WARN] Empty embeddings file, skipping: {npz_path}')
            continue
        fold_name = os.path.basename(os.path.dirname(npz_path))
        print(f'Adding {fold_name}: X={X.shape}, y={y.shape}')
        X_list.append(X)
        y_list.append(y)
        idx_list.append(indices)
        source_list.extend([fold_name] * len(y))

    if not X_list:
        print('[WARN] No non-empty fold-level embeddings found; skipping dataset aggregation.')
        return None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    idx_all = np.concatenate(idx_list, axis=0)
    source_all = np.asarray(source_list)

    keep_positions = []
    seen = set()
    duplicate_count = 0
    for pos, idx in enumerate(idx_all.tolist()):
        key = int(idx)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        keep_positions.append(pos)

    if duplicate_count > 0:
        print(f'[WARN] Removed {duplicate_count} duplicated sample indices from dataset-level embeddings.')
        keep_positions = np.asarray(keep_positions, dtype=np.int64)
        X_all = X_all[keep_positions]
        y_all = y_all[keep_positions]
        idx_all = idx_all[keep_positions]
        source_all = source_all[keep_positions]

    aggregate_dir = os.path.join(xai_data_root, DATASET_AGGREGATE_DIR_NAME)
    os.makedirs(aggregate_dir, exist_ok=True)

    out_npz = os.path.join(aggregate_dir, f'embeddings_{DATASET_AGGREGATE_SUBSET}.npz')
    np.savez(
        out_npz,
        embeddings=X_all,
        labels=y_all,
        indices=idx_all,
        source_folds=source_all,
    )

    out_csv = os.path.join(aggregate_dir, f'embeddings_{DATASET_AGGREGATE_SUBSET}_summary.csv')
    pd.DataFrame({
        'index': idx_all,
        'label': y_all,
        'source_fold': source_all,
    }).to_csv(out_csv, index=False)

    print(f'Saved dataset-level embeddings -> {out_npz}')
    print(f'Saved dataset-level summary -> {out_csv}')
    print(f'Dataset-level shapes: X={X_all.shape}, y={y_all.shape}')
    return out_npz, aggregate_dir


def run_dataset_level_projection(model_dir, xai_data_root, xai_plot_root):
    result = build_dataset_level_npz(model_dir, xai_data_root, subset='test')
    if result is None:
        return None

    npz_path, data_aggregate_dir = result
    plot_aggregate_dir = os.path.join(xai_plot_root, DATASET_AGGREGATE_DIR_NAME)
    run_projection_file(
        npz_path=npz_path,
        subset=DATASET_AGGREGATE_SUBSET,
        title_suffix='dataset-level out-of-fold test',
        output_dir=plot_aggregate_dir,
    )
    return data_aggregate_dir, plot_aggregate_dir

def main():
    args = parse_args()
    print_args(args)

    k_type, _ = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)
    model_dir = build_model_dir(paths.logs_dir, args.model_type, args.pooling, k_type, run_suffix, args.epochs)
    xai_data_root = build_xai_embeddings_data_dir(paths.logs_dir, args.model_type, args.pooling, k_type, run_suffix, args.epochs)
    xai_plot_root = build_xai_output_dir(paths.logs_dir, 'embeddings_xai', args.model_type, args.pooling, k_type, run_suffix, args.epochs)

    print(f'Model dir: {model_dir}')
    print(f'XAI embedding data dir: {xai_data_root}')
    print(f'XAI embedding plot dir: {xai_plot_root}')
    samples, _ = get_dataset(paths.embeddings_dir, k_type)
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))

    rows = []
    for fold in folds:
        row = run_fold(args, fold, model_dir, xai_data_root, xai_plot_root)
        if row is not None:
            rows.append(row)

    aggregate_result = run_dataset_level_projection(model_dir, xai_data_root, xai_plot_root)
    if aggregate_result is not None:
        data_aggregate_dir, plot_aggregate_dir = aggregate_result
        rows.append({
            'fold': 'dataset',
            'source_embedding_dir': data_aggregate_dir,
            'output_dir': plot_aggregate_dir,
            'subsets': DATASET_AGGREGATE_SUBSET,
        })

    summary_path = os.path.join(xai_plot_root, 'xai_embeddings_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')


if __name__ == '__main__':
    main()
