import os
import shutil
import pandas as pd

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
from classifier.utils_data import get_dataset
from classifier.utils_gxi import sum_gxi_by_class, sum_gxi_by_class_from_dirs
from classifier.utils_xai_cache import aggregate_profiles_from_npz, aggregate_overall_profile_from_npz, find_split_npz, infer_valid_mask_from_samples


GXI_SMOOTH_WINDOWS = (51, 101, 201)


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def _load_gene_boundaries(k):
    genome_end = LONGEST_SEQUENCE_LENGTH - (int(k) - 1)
    map_file = os.path.join(paths.msa_refseq_map_dir, 'coordinates_dengue_LONGEST.csv')
    map_df = pd.read_csv(map_file)
    return get_gene_boundaries(map_df, gene_name='Proteina', genome_end=genome_end), genome_end


def plot_profiles(class_profiles, output_dir, args, prefix='denv_gxi'):
    gene_boundaries, genome_end = _load_gene_boundaries(args.k)

    # Raw profile. This remains the reference, unsmoothed output.
    plot_attention_stacked_with_genes(
        class_profiles=class_profiles,
        gene_boundaries=gene_boundaries,
        output_dir=output_dir,
        prefix=prefix,
        region_name='Region',
        xmax=genome_end,
        ylabel='GxI contribution (%)',
        percent_yaxis=True,
    )

    # Additional smoothed versions for paper-friendly visualization.
    # Each smoothed profile is renormalized to 0-100 after smoothing,
    # preserving NaN padded positions so no artificial tail is plotted.
    for window in GXI_SMOOTH_WINDOWS:
        plot_attention_stacked_with_genes(
            class_profiles=class_profiles,
            gene_boundaries=gene_boundaries,
            output_dir=output_dir,
            prefix=prefix,
            region_name='Region',
            xmax=genome_end,
            ylabel='GxI contribution (%)',
            percent_yaxis=True,
            smooth_window=window,
            filename_suffix=f'_smoothed_{window}',
            renormalize_after_smoothing=True,
        )


def run_fold(args, fold, model_dir, xai_out_root, samples):
    fold_id_safe = safe_name(fold['fold_id'])
    fold_dir_name = f'split_{fold_id_safe}'
    gxi_dir = os.path.join(model_dir, 'gxi', fold_dir_name)
    npy_dir = os.path.join(gxi_dir, 'numpy')
    npz_path = find_split_npz(gxi_dir, 'gxi', subset='test')
    valid_mask = infer_valid_mask_from_samples(samples[fold['test_idx']]) if npz_path is not None else None
    out_dir = os.path.join(xai_out_root, f'split_{fold_id_safe}', 'output-aggregate')

    if npz_path is None and not os.path.isdir(npy_dir):
        print(f'[WARN] Missing GxI npz/numpy dir, skipping fold {fold_id_safe}: {gxi_dir}')
        return None

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n==== GxI AGGREGATE | FOLD {fold["fold_id"]} ====')
    if npz_path is not None:
        class_profiles, _ = aggregate_profiles_from_npz(
            npz_paths=[npz_path],
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=True,
            normalize_per_sample='abs_sum1',
            valid_masks=[valid_mask],
        )
    else:
        class_profiles = sum_gxi_by_class(
            input_dir=npy_dir,
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=True,
        )
    plot_profiles(class_profiles, out_dir, args, prefix='denv_gxi')
    if npz_path is not None:
        overall_profiles, _ = aggregate_overall_profile_from_npz(
            npz_paths=[npz_path],
            output_dir=out_dir,
            prefix='overall',
            normalize='class',
            divide=True,
            normalize_per_sample='abs_sum1',
            valid_masks=[valid_mask],
            label='Overall',
        )
        plot_profiles(overall_profiles, out_dir, args, prefix='denv_gxi_overall')
    return {'fold': fold['fold_id'], 'output_dir': out_dir, 'n_classes': len(class_profiles)}


def run_dataset(args, folds, model_dir, xai_out_root, samples):
    gxi_root = os.path.join(model_dir, 'gxi')
    out_dir = os.path.join(xai_out_root, 'output-aggregate-dataset')

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    npz_paths = []
    valid_masks = []
    npy_dirs = []
    for fold in folds:
        fold_id_safe = safe_name(fold['fold_id'])
        split_dir = os.path.join(gxi_root, f'split_{fold_id_safe}')
        npz_path = find_split_npz(split_dir, 'gxi', subset='test')
        if npz_path is not None:
            npz_paths.append(npz_path)
            valid_masks.append(infer_valid_mask_from_samples(samples[fold['test_idx']]))
        npy_dirs.append(os.path.join(split_dir, 'numpy'))

    print('\n==== GxI AGGREGATE | DATASET LEVEL ====')
    if npz_paths:
        print(f'[INFO] Reading dataset-level GxI from {len(npz_paths)} split NPZ files')
        class_profiles, class_counts = aggregate_profiles_from_npz(
            npz_paths=npz_paths,
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=True,
            normalize_per_sample='abs_sum1',
            valid_masks=valid_masks,
        )
    else:
        print('[WARN] No split-level GxI NPZ files found; falling back to legacy per-sample NPY directories')
        class_profiles, class_counts = sum_gxi_by_class_from_dirs(
            input_dirs=npy_dirs,
            output_dir=out_dir,
            prefix='class',
            class_dict=CLASS_DICT,
            normalize='class',
            divide=False,
        )

    if class_profiles:
        plot_profiles(class_profiles, out_dir, args, prefix='denv_gxi_dataset')
        if npz_paths:
            overall_profiles, _ = aggregate_overall_profile_from_npz(
                npz_paths=npz_paths,
                output_dir=out_dir,
                prefix='overall_dataset',
                normalize='class',
                divide=True,
                normalize_per_sample='abs_sum1',
                valid_masks=valid_masks,
                label='Overall',
            )
            plot_profiles(overall_profiles, out_dir, args, prefix='denv_gxi_dataset_overall')
    else:
        print(f'[WARN] No dataset-level GxI profiles generated in {out_dir}')

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
    xai_out_root = build_xai_output_dir(paths.logs_dir, 'gxi_aggregate', 'denformer', args.pooling, k_type, run_suffix, args.epochs)

    print(f'Model dir: {model_dir}')
    print(f'XAI output dir: {xai_out_root}')
    samples, _ = get_dataset(paths.embeddings_dir, k_type)
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))

    rows = []
    for fold in folds:
        row = run_fold(args, fold, model_dir, xai_out_root, samples)
        if row is not None:
            rows.append(row)
    rows.append(run_dataset(args, folds, model_dir, xai_out_root, samples))

    summary_path = os.path.join(xai_out_root, 'gxi_aggregate_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')


if __name__ == '__main__':
    main()
