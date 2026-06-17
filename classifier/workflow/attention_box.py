import os
import shutil
import pandas as pd

import paths
from classifier.config import LONGEST_SEQUENCE_LENGTH
from classifier.utils_genome_map import get_gene_boundaries
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
from classifier.utils_attn_box import (
    build_region_attention_long_df,
    kruskal_by_region,
    plot_attention_boxplots_by_region_rows,
    plot_effect_size_heatmap_regions_x_serotypes,
    plot_kruskal_effect_size_by_region_column,
    plot_kruskal_pvalues_by_region_column,
    plot_pvalue_heatmap_regions_x_serotypes,
    pairwise_serotype_stats_by_region,
    pairwise_stats_matrix,
)
from classifier.utils_data import get_dataset
from classifier.utils_xai_cache import build_region_long_df_from_npz, find_split_npz

CLASS_DICT = {0: 'DENV1', 1: 'DENV2', 2: 'DENV3', 3: 'DENV4'}
SEROTYPES_ORDER = ['DENV1', 'DENV2', 'DENV3', 'DENV4']
COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def _regions(k):
    genome_end = LONGEST_SEQUENCE_LENGTH - (int(k) - 1)
    map_file = os.path.join(paths.msa_refseq_map_dir, 'coordinates_dengue_LONGEST.csv')
    map_df = pd.read_csv(map_file)
    gene_boundaries = get_gene_boundaries(map_df, gene_name='Proteina', genome_end=genome_end)
    if isinstance(gene_boundaries, dict):
        return gene_boundaries
    return {name: (start, end) for name, start, end in gene_boundaries}


def _run_box_analysis(df_long, out_dir, regions_order, label):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'attention_by_region_long.csv')
    df_long.to_csv(csv_path, index=False)
    print(f'Saved -> {csv_path}')

    fig, _ = plot_attention_boxplots_by_region_rows(
        df_long,
        regions_order=regions_order,
        serotypes_order=SEROTYPES_ORDER,
        colors=COLORS,
        hide_yticks=True,
        show_serotype_labels_once=True,
        figsize_per_row=(6, 0.8),
    )
    fig_path = os.path.join(out_dir, 'boxplots_region_by_serotype.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved -> {fig_path}')

    kw_results = kruskal_by_region(
        df_long,
        regions_order=regions_order,
        serotypes_order=SEROTYPES_ORDER,
        fdr_correct=True,
    )
    kw_csv = os.path.join(out_dir, 'kruskal_by_region.csv')
    kw_results.to_csv(kw_csv)
    print(f'Saved -> {kw_csv}')

    fig, _, _ = plot_kruskal_pvalues_by_region_column(
        kw_results,
        regions_order,
        p_col='p_fdr' if 'p_fdr' in kw_results.columns else 'p',
        use_log=True,
        text_mode='stars',
        figsize=(2.0, 10),
    )
    fig_path = os.path.join(out_dir, 'heatmap_region_kruskal.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved -> {fig_path}')

    for effect_col in ['epsilon_squared_R', 'eta_squared_H']:
        if effect_col not in kw_results.columns:
            continue
        fig, _, _ = plot_kruskal_effect_size_by_region_column(
            kw_results,
            regions_order,
            effect_col=effect_col,
            effect_col_title=effect_col.replace('_', ' '),
            figsize=(2.0, 10),
            show_text=True,
            value_format='.2f',
        )
        fig_path = os.path.join(out_dir, f'heatmap_region_kruskal_{effect_col}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved -> {fig_path}')

    pairwise_stats = pairwise_serotype_stats_by_region(
        df_long,
        regions_order=regions_order,
        serotypes_order=SEROTYPES_ORDER,
        fdr_correct=True,
        fdr_scope='global',
    )
    pairwise_stats_csv = os.path.join(out_dir, 'pairwise_serotype_stats_by_region.csv')
    pairwise_stats.to_csv(pairwise_stats_csv, index=False)
    print(f'Saved -> {pairwise_stats_csv}')

    p_col = 'p_fdr' if 'p_fdr' in pairwise_stats.columns else 'p'
    pmat_pairs = pairwise_stats_matrix(pairwise_stats, value_col=p_col, regions_order=regions_order)
    pairwise_csv = os.path.join(out_dir, 'pairwise_serotype_pvalues_by_region.csv')
    pmat_pairs.to_csv(pairwise_csv)
    print(f'Saved -> {pairwise_csv}')

    vda_matrix = pairwise_stats_matrix(pairwise_stats, value_col='VDA_folded', regions_order=regions_order)
    vda_csv = os.path.join(out_dir, 'pairwise_serotype_vda_by_region.csv')
    vda_matrix.to_csv(vda_csv)
    print(f'Saved -> {vda_csv}')

    rb_matrix = pairwise_stats_matrix(pairwise_stats, value_col='rank_biserial', regions_order=regions_order)
    rb_csv = os.path.join(out_dir, 'pairwise_serotype_rank_biserial_by_region.csv')
    rb_matrix.to_csv(rb_csv)
    print(f'Saved -> {rb_csv}')

    fig, _ = plot_pvalue_heatmap_regions_x_serotypes(
        pmat_pairs,
        show_title=False,
        show_text=True,
        text_mode='stars',
        use_log=True,
        p_format='.3g',
        figsize=(8, 10),
        legend_orientation='vertical',
    )
    fig_path = os.path.join(out_dir, 'heatmap_region_x_serotype.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved -> {fig_path}')

    effect_heatmaps = [
        (
            vda_matrix,
            'vda_folded',
            'Vargha-Delaney A, folded',
            0.5,
            1.0,
        ),
        (
            rb_matrix,
            'rank_biserial',
            'Rank-biserial correlation',
            -1.0,
            1.0,
        ),
    ]
    for matrix, suffix, label_txt, vmin, vmax in effect_heatmaps:
        fig, _ = plot_effect_size_heatmap_regions_x_serotypes(
            matrix,
            show_title=False,
            show_text=True,
            value_format='.2f',
            figsize=(8, 10),
            vmin=vmin,
            vmax=vmax,
            cbar_label=label_txt,
            legend_orientation='vertical',
        )
        fig_path = os.path.join(out_dir, f'heatmap_region_x_serotype_{suffix}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved -> {fig_path}')

    return {
        'label': label,
        'output_dir': out_dir,
        'n_rows': len(df_long),
        'n_unique_files': df_long['file'].nunique() if 'file' in df_long.columns else None,
    }


def run_fold(args, fold, model_dir, xai_out_root):
    fold_id_safe = safe_name(fold['fold_id'])
    fold_dir_name = f'split_{fold_id_safe}'
    attn_dir = os.path.join(model_dir, 'attention', fold_dir_name)
    npy_dir = os.path.join(attn_dir, 'numpy')
    npz_path = find_split_npz(attn_dir, 'attention', subset='test')
    out_dir = os.path.join(xai_out_root, f'split_{fold_id_safe}', 'output-box')

    if npz_path is None and not os.path.isdir(npy_dir):
        print(f'[WARN] Missing attention npz/numpy dir, skipping fold {fold_id_safe}: {attn_dir}')
        return None

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n==== ATTENTION BOX | FOLD {fold["fold_id"]} ====')
    regions = _regions(args.k)
    regions_order = list(regions.keys())

    if npz_path is not None:
        df_long = build_region_long_df_from_npz(
            npz_path=npz_path,
            regions=regions,
            class_dict=CLASS_DICT,
            value_col='attention',
            region_reduce='mean',
            normalize_per_sample=False,
        )
    else:
        df_long = build_region_attention_long_df(
            input_dir=npy_dir,
            regions=regions,
            class_dict=CLASS_DICT,
            region_reduce='mean',
            normalize_per_sample=False,
        )
    df_long.insert(0, 'fold', fold['fold_id'])

    row = _run_box_analysis(
        df_long=df_long,
        out_dir=out_dir,
        regions_order=regions_order,
        label=f'fold_{fold["fold_id"]}',
    )
    row['fold'] = fold['fold_id']
    return row


def run_dataset(args, folds, model_dir, xai_out_root):
    out_dir = os.path.join(xai_out_root, 'output-box-dataset')

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('\n==== ATTENTION BOX | DATASET ====', flush=True)
    regions = _regions(args.k)
    regions_order = list(regions.keys())

    dfs = []
    skipped = []
    for fold in folds:
        fold_id_safe = safe_name(fold['fold_id'])
        fold_dir_name = f'split_{fold_id_safe}'
        split_dir = os.path.join(model_dir, 'attention', fold_dir_name)
        npy_dir = os.path.join(split_dir, 'numpy')
        npz_path = find_split_npz(split_dir, 'attention', subset='test')

        if npz_path is None and not os.path.isdir(npy_dir):
            print(f'[WARN] Missing attention npz/numpy dir, skipping fold {fold_id_safe}: {split_dir}')
            skipped.append(fold['fold_id'])
            continue

        if npz_path is not None:
            df_fold = build_region_long_df_from_npz(
                npz_path=npz_path,
                regions=regions,
                class_dict=CLASS_DICT,
                value_col='attention',
                region_reduce='mean',
                normalize_per_sample=False,
            )
        else:
            df_fold = build_region_attention_long_df(
                input_dir=npy_dir,
                regions=regions,
                class_dict=CLASS_DICT,
                region_reduce='mean',
                normalize_per_sample=False,
            )
        df_fold.insert(0, 'fold', fold['fold_id'])
        dfs.append(df_fold)

    if not dfs:
        print('[WARN] No fold-level attention files found. Dataset-level output-box was not generated.')
        return None

    df_long = pd.concat(dfs, axis=0, ignore_index=True)
    row = _run_box_analysis(
        df_long=df_long,
        out_dir=out_dir,
        regions_order=regions_order,
        label='dataset',
    )
    row['fold'] = 'dataset'
    row['n_folds'] = len(dfs)
    row['skipped_folds'] = ';'.join(map(str, skipped)) if skipped else ''
    return row


def main():
    args = parse_args()
    print_args(args)

    k_type, _ = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)
    model_dir = build_model_dir(paths.logs_dir, 'denformer', args.pooling, k_type, run_suffix, args.epochs)
    xai_out_root = build_xai_output_dir(paths.logs_dir, 'attention_box', 'denformer', args.pooling, k_type, run_suffix, args.epochs)

    print(f'Model dir: {model_dir}')
    print(f'XAI output dir: {xai_out_root}')
    samples, _ = get_dataset(paths.embeddings_dir, k_type)
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))

    rows = []
    for fold in folds:
        row = run_fold(args, fold, model_dir, xai_out_root)
        if row is not None:
            rows.append(row)

    dataset_row = run_dataset(args, folds, model_dir, xai_out_root)
    if dataset_row is not None:
        rows.append(dataset_row)

    summary_path = os.path.join(xai_out_root, 'attention_box_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')


if __name__ == '__main__':
    main()
