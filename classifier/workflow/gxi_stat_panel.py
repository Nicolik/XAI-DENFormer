import os
import pandas as pd
import matplotlib.pyplot as plt

import paths
from classifier.workflow import config
from classifier.workflow.utils import (
    build_xai_output_dir,
    get_run_suffix,
    parse_run_args,
    resolve_k_type_and_emb_dim,
)
from classifier.utils import get_args, print_args
from classifier.workflow.attention_stat_panel import (
    plot_pvalue_panel,
    plot_effect_size_panel,
    export_stat_panel_excel,
)


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def _load_inputs(box_dataset_dir):
    df_path = os.path.join(box_dataset_dir, 'gxi_by_region_long.csv')
    kw_path = os.path.join(box_dataset_dir, 'kruskal_by_region.csv')
    pair_path = os.path.join(box_dataset_dir, 'pairwise_serotype_stats_by_region.csv')
    missing = [p for p in [df_path, kw_path, pair_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('Missing required GxI box outputs:\n' + '\n'.join(missing))

    df = pd.read_csv(df_path)
    kw = pd.read_csv(kw_path)
    pairwise = pd.read_csv(pair_path)
    if 'region' not in kw.columns:
        kw = kw.rename(columns={kw.columns[0]: 'region'})
    regions_order = list(pd.unique(df['region']))
    return df, kw, pairwise, regions_order


def main():
    args = parse_args()
    print_args(args)

    k_type, _ = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)

    box_root = build_xai_output_dir(
        paths.logs_dir,
        'gxi_box',
        'denformer',
        args.pooling,
        k_type,
        run_suffix,
        args.epochs,
    )
    box_dataset_dir = os.path.join(box_root, 'output-box-dataset')
    out_dir = build_xai_output_dir(
        paths.logs_dir,
        'gxi_stat_panel',
        'denformer',
        args.pooling,
        k_type,
        run_suffix,
        args.epochs,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f'Input GxI box dataset dir: {box_dataset_dir}')
    print(f'Output GxI stat panel dir: {out_dir}')

    df, kw, pairwise, regions_order = _load_inputs(box_dataset_dir)
    title_prefix = f'{args.run_name} | denformer {args.pooling} | normalized abs GxI regional statistics'

    fig = plot_pvalue_panel(df, kw, pairwise, regions_order, title=title_prefix + ' | p-values', value_col='gxi')
    for ext in ['png', 'pdf']:
        path = os.path.join(out_dir, f'gxi_region_stats_pvalues_raw.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved -> {path}')
    plt.close(fig)

    fig = plot_effect_size_panel(df, kw, pairwise, regions_order, title=title_prefix + ' | effect sizes', value_col='gxi')
    for ext in ['png', 'pdf']:
        path = os.path.join(out_dir, f'gxi_region_stats_effect_size_raw.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved -> {path}')
    plt.close(fig)

    excel_path = export_stat_panel_excel(kw, pairwise, regions_order, out_dir)
    gxi_excel_path = os.path.join(out_dir, 'gxi_region_stats_raw.xlsx')
    if os.path.exists(excel_path) and excel_path != gxi_excel_path:
        os.replace(excel_path, gxi_excel_path)
        excel_path = gxi_excel_path
        print(f'Renamed Excel -> {excel_path}')

    summary_path = os.path.join(out_dir, 'gxi_stat_panel_summary.csv')
    pd.DataFrame([{
        'box_dataset_dir': box_dataset_dir,
        'output_dir': out_dir,
        'excel_path': excel_path,
        'n_gxi_rows': len(df),
        'n_regions': len(regions_order),
        'gxi_region_value': 'mean(abs(GxI) / sum(abs(GxI)) per sample) when read from NPZ/NPY',
        'kruskal_effect_size': 'epsilon_squared_R',
        'pairwise_effect_size': 'rank_biserial',
        'pvalue_column': 'p_fdr if available, else p',
    }]).to_csv(summary_path, index=False)
    print(f'Saved -> {summary_path}')


if __name__ == '__main__':
    main()
