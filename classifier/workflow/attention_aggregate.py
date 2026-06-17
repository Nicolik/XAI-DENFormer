import os
import re
import shutil
import pandas as pd
import numpy as np

import paths
from classifier.config import CLASS_DICT, LONGEST_SEQUENCE_LENGTH
from classifier.utils_genome_map import get_gene_boundaries, plot_attention_stacked_with_genes
from classifier.workflow import config
from classifier.workflow.utils import (
    build_model_dir,
    build_xai_output_dir,
    get_run_suffix,
    load_and_validate_folds,
    parse_run_args,
    resolve_k_type_and_emb_dim,
    safe_name,
)
from classifier.utils import get_args, print_args
from classifier.utils_attn import sum_attention_by_class
from classifier.utils_xai_cache import aggregate_profiles_from_npz, aggregate_overall_profile_from_npz, find_split_npz, infer_valid_mask_from_samples
from classifier.utils_data import get_dataset


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def _load_gene_boundaries(k):
    genome_end = LONGEST_SEQUENCE_LENGTH - (int(k) - 1)
    map_file = os.path.join(paths.msa_refseq_map_dir, 'coordinates_dengue_LONGEST.csv')
    map_df = pd.read_csv(map_file)
    return get_gene_boundaries(map_df, gene_name='Proteina', genome_end=genome_end), genome_end


def _normalize_profiles(raw_profiles, normalize):
    if not raw_profiles or normalize == 'no':
        return raw_profiles

    profiles = {k: np.asarray(v, dtype=float) for k, v in raw_profiles.items()}

    if normalize == 'global':
        all_values = np.concatenate(list(profiles.values()))
        vmin, vmax = float(all_values.min()), float(all_values.max())
        if vmax > vmin:
            return {k: (v - vmin) / (vmax - vmin) for k, v in profiles.items()}
        return {k: np.zeros_like(v) for k, v in profiles.items()}

    if normalize == 'class':
        out = {}
        for k, v in profiles.items():
            vmin, vmax = float(v.min()), float(v.max())
            out[k] = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
        return out

    raise ValueError(f'Unknown normalize={normalize!r}. Use no, global, or class.')


def sum_attention_by_class_from_dirs(input_dirs, output_dir, prefix='class', class_dict=None,
                                     normalize='class', divide=False):
    """
    Aggregate attention profiles by class across multiple fold-level numpy dirs.

    With divide=False this computes true sums across all saved samples. With
    divide=True it computes means across all saved samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r'batch(\d+)_sample(\d+)_class(\d+)_attn\.npy')

    class_sums = {}
    class_counts = {}
    total_files = 0

    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f'[WARN] Missing attention numpy dir, skipping: {input_dir}')
            continue

        for fname in os.listdir(input_dir):
            if not fname.endswith('_attn.npy'):
                continue
            match = pattern.search(fname)
            if not match:
                continue

            class_idx = int(match.group(3))
            arr = np.load(os.path.join(input_dir, fname))

            if class_idx not in class_sums:
                class_sums[class_idx] = np.zeros_like(arr, dtype=float)
                class_counts[class_idx] = 0

            class_sums[class_idx] += arr
            class_counts[class_idx] += 1
            total_files += 1

    if total_files == 0:
        print('[WARN] No attention .npy files found for dataset-level aggregation.')
        return {}, {}

    raw_profiles = {}
    count_rows = []
    for class_idx in sorted(class_sums):
        class_name = f'class {class_idx}' if class_dict is None else class_dict[class_idx]
        profile = class_sums[class_idx]
        if divide:
            profile = profile / max(class_counts[class_idx], 1)
        raw_profiles[class_name] = profile
        count_rows.append({
            'class_idx': class_idx,
            'class_name': class_name,
            'n_samples': int(class_counts[class_idx]),
        })
        print(
            f'[dataset aggregate] {class_name}: n={class_counts[class_idx]} | '
            f'min={profile.min():.6g} | max={profile.max():.6g}'
        )

    profiles = _normalize_profiles(raw_profiles, normalize=normalize)

    for class_name, profile in profiles.items():
        out_path = os.path.join(output_dir, f'{prefix}_{class_name}_sum.npy')
        np.save(out_path, profile)
        print(f'[INFO] Saved dataset-level attention for {class_name} -> {out_path}')

    counts_df = pd.DataFrame(count_rows)
    counts_path = os.path.join(output_dir, f'{prefix}_counts.csv')
    counts_df.to_csv(counts_path, index=False)
    print(f'[INFO] Saved dataset-level counts -> {counts_path}')

    return profiles, {row['class_name']: row['n_samples'] for row in count_rows}


ATTENTION_SMOOTH_WINDOWS = (51, 101, 201)


def plot_profiles(class_profiles, output_dir, args, prefix='denv'):
    gene_boundaries, genome_end = _load_gene_boundaries(args.k)

    # Raw profile. This remains the reference, unsmoothed output.
    plot_attention_stacked_with_genes(
        class_profiles=class_profiles,
        gene_boundaries=gene_boundaries,
        output_dir=output_dir,
        prefix=prefix,
        region_name='Region',
        xmax=genome_end,
    )

    # Additional smoothed versions for paper-friendly visualization.
    # Each smoothed profile is renormalized to 0-100 after smoothing,
    # preserving NaN padded positions so no artificial tail is plotted.
    for window in ATTENTION_SMOOTH_WINDOWS:
        plot_attention_stacked_with_genes(
            class_profiles=class_profiles,
            gene_boundaries=gene_boundaries,
            output_dir=output_dir,
            prefix=prefix,
            region_name='Region',
            xmax=genome_end,
            smooth_window=window,
            filename_suffix=f'_smoothed_{window}',
            renormalize_after_smoothing=True,
        )


def run_fold(args, fold, model_dir, xai_out_root, samples):
    fold_id_safe = safe_name(fold['fold_id'])
    fold_dir_name = f'split_{fold_id_safe}'
    attn_dir = os.path.join(model_dir, 'attention', fold_dir_name)
    npy_dir = os.path.join(attn_dir, 'numpy')
    npz_path = find_split_npz(attn_dir, 'attention', subset='test')
    valid_mask = infer_valid_mask_from_samples(samples[fold['test_idx']]) if npz_path is not None else None
    out_dir = os.path.join(xai_out_root, f'split_{fold_id_safe}', 'output-aggregate')

    if npz_path is None and not os.path.isdir(npy_dir):
        print(f'[WARN] Missing attention npz/numpy dir, skipping fold {fold_id_safe}: {attn_dir}')
        return None

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n==== ATTENTION AGGREGATE | FOLD {fold["fold_id"]} ====')
    if npz_path is not None:
        class_profiles, _ = aggregate_profiles_from_npz(
            npz_paths=[npz_path],
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=True,
            valid_masks=[valid_mask],
        )
    else:
        class_profiles = sum_attention_by_class(
            input_dir=npy_dir,
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=True,
            valid_masks=valid_masks,
        )

    plot_profiles(class_profiles, out_dir, args, prefix='denv')
    if npz_path is not None:
        overall_profiles, _ = aggregate_overall_profile_from_npz(
            npz_paths=[npz_path],
            output_dir=out_dir,
            prefix='overall',
            normalize='class',
            divide=True,
            valid_masks=[valid_mask],
            label='Overall',
        )
        plot_profiles(overall_profiles, out_dir, args, prefix='denv_overall')

    return {'fold': fold['fold_id'], 'output_dir': out_dir, 'n_classes': len(class_profiles)}


def run_dataset(args, folds, model_dir, xai_out_root, samples):
    attention_root = os.path.join(model_dir, 'attention')
    out_dir = os.path.join(xai_out_root, 'output-aggregate-dataset')

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    npz_paths = []
    valid_masks = []
    npy_dirs = []
    for fold in folds:
        fold_id_safe = safe_name(fold['fold_id'])
        split_dir = os.path.join(attention_root, f'split_{fold_id_safe}')
        npz_path = find_split_npz(split_dir, 'attention', subset='test')
        if npz_path is not None:
            npz_paths.append(npz_path)
            valid_masks.append(infer_valid_mask_from_samples(samples[fold['test_idx']]))
        npy_dirs.append(os.path.join(split_dir, 'numpy'))

    print('\n==== ATTENTION AGGREGATE | DATASET LEVEL ====')
    if npz_paths:
        print(f'[INFO] Reading dataset-level attention from {len(npz_paths)} split NPZ files')
        class_profiles, class_counts = aggregate_profiles_from_npz(
            npz_paths=npz_paths,
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=True,
            valid_masks=valid_masks,
        )
    else:
        print('[WARN] No split-level attention NPZ files found; falling back to legacy per-sample NPY directories')
        class_profiles, class_counts = sum_attention_by_class_from_dirs(
            input_dirs=npy_dirs,
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=False,
        )

    if class_profiles:
        plot_profiles(class_profiles, out_dir, args, prefix='denv_dataset')
        if npz_paths:
            overall_profiles, _ = aggregate_overall_profile_from_npz(
                npz_paths=npz_paths,
                output_dir=out_dir,
                prefix='overall_dataset',
                normalize='class',
                divide=True,
                valid_masks=valid_masks,
                label='Overall',
            )
            plot_profiles(overall_profiles, out_dir, args, prefix='denv_dataset_overall')
    else:
        print(f'[WARN] No dataset-level attention profiles generated in {out_dir}')

    return {
        'fold': 'dataset',
        'output_dir': out_dir,
        'n_classes': len(class_profiles),
        'n_samples': int(sum(class_counts.values())) if class_counts else 0,
    }


def main():
    args = parse_args()
    print_args(args)

    k_type, _ = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)
    model_dir = build_model_dir(paths.logs_dir, 'denformer', args.pooling, k_type, run_suffix, args.epochs)
    xai_out_root = build_xai_output_dir(paths.logs_dir, 'attention_aggregate', 'denformer', args.pooling, k_type, run_suffix, args.epochs)

    print(f'Model dir: {model_dir}')
    print(f'XAI output dir: {xai_out_root}')
    samples, _ = get_dataset(paths.embeddings_dir, k_type)
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))

    rows = []
    for fold in folds:
        row = run_fold(args, fold, model_dir, xai_out_root, samples)
        if row is not None:
            rows.append(row)

    dataset_row = run_dataset(args, folds, model_dir, xai_out_root, samples)
    rows.append(dataset_row)

    summary_path = os.path.join(xai_out_root, 'attention_aggregate_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')


if __name__ == '__main__':
    main()
