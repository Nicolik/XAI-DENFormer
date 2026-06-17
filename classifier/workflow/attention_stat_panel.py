import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm

import paths
from classifier.workflow import config
from classifier.workflow.utils import (
    build_xai_output_dir,
    get_run_suffix,
    parse_run_args,
    resolve_k_type_and_emb_dim,
)
from classifier.utils import get_args, print_args
from classifier.utils_attn_box import robust_shared_ylim_from_values

SEROTYPES_ORDER = ['DENV1', 'DENV2', 'DENV3', 'DENV4']
COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
PAIRWISE_ORDER = ['DENV1vDENV2', 'DENV1vDENV3', 'DENV1vDENV4', 'DENV2vDENV3', 'DENV2vDENV4', 'DENV3vDENV4']
PAIRWISE_SHORT = ['DENV1v2', 'DENV1v3', 'DENV1v4', 'DENV2v3', 'DENV2v4', 'DENV3v4']


# Effect-size magnitude thresholds used only for visual annotation in the
# effect-size panel. They are intentionally heuristic, not inferential tests.
# Kruskal-Wallis epsilon-squared: small >= 0.01, medium >= 0.06, large >= 0.14.
# Rank-biserial / Cliff's delta: small >= 0.11, medium >= 0.28, large >= 0.43.
KW_EPS_THRESHOLDS = (0.01, 0.06, 0.14)
PAIRWISE_RB_THRESHOLDS = (0.11, 0.28, 0.43)


def _effect_stars(value, thresholds, signed=False):
    if not np.isfinite(value):
        return ''
    mag = abs(float(value)) if signed else float(value)
    if mag >= thresholds[2]:
        return '***'
    if mag >= thresholds[1]:
        return '**'
    if mag >= thresholds[0]:
        return '*'
    return ''


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def _safe_neglog10(values, cap=300.0):
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.any():
        clipped = np.clip(arr[mask], 10.0 ** (-cap), 1.0)
        out[mask] = -np.log10(clipped)
    return out


def _stars(p):
    if not np.isfinite(p):
        return ''
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return ''


def _clean_pairwise_label(label):
    return str(label).replace('DENV', '').replace('1v2', 'DENV1v2')


def _load_inputs(box_dataset_dir):
    df_path = os.path.join(box_dataset_dir, 'attention_by_region_long.csv')
    kw_path = os.path.join(box_dataset_dir, 'kruskal_by_region.csv')
    pair_path = os.path.join(box_dataset_dir, 'pairwise_serotype_stats_by_region.csv')

    missing = [p for p in [df_path, kw_path, pair_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('Missing required attention box outputs:\n' + '\n'.join(missing))

    df = pd.read_csv(df_path)
    kw = pd.read_csv(kw_path)
    pairwise = pd.read_csv(pair_path)

    if 'region' not in kw.columns:
        kw = kw.rename(columns={kw.columns[0]: 'region'})

    regions_order = list(pd.unique(df['region']))
    return df, kw, pairwise, regions_order


def _pivot_pairwise(pairwise, value_col, regions_order):
    matrix = pairwise.pivot(index='region', columns='comparison', values=value_col)
    present = [c for c in PAIRWISE_ORDER if c in matrix.columns]
    if not present:
        present = list(matrix.columns)
    return matrix.reindex(regions_order)[present]


def _draw_box_panel(fig, outer_spec, df, regions_order, value_col='attention'):
    nrows = len(regions_order)
    sub = gridspec.GridSpecFromSubplotSpec(nrows, 1, subplot_spec=outer_spec, hspace=0.0)
    axes = []
    # Use row-specific robust y-limits. A global 1st-99th percentile range
    # can still flatten low-dynamic regions when a few regions dominate the
    # upper tail, especially for sparse attribution signals such as GxI.
    shared_ylim = None
    for i, region in enumerate(regions_order):
        ax = fig.add_subplot(sub[i, 0])
        axes.append(ax)
        data = []
        for serotype in SEROTYPES_ORDER:
            vals = df[(df['region'] == region) & (df['serotype'] == serotype)][value_col].dropna().values
            data.append(vals)

        bp = ax.boxplot(
            data,
            positions=np.arange(1, len(SEROTYPES_ORDER) + 1),
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops=dict(color='brown', linewidth=1.5),
            boxprops=dict(linewidth=1.1),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
        )
        for box, color in zip(bp['boxes'], COLORS):
            box.set_facecolor(color)
            box.set_alpha(0.9)

        ax.set_ylabel(region, fontsize=13, rotation=0, labelpad=30, va='center')
        ax.set_xlim(0.4, len(SEROTYPES_ORDER) + 0.6)
        ax.set_yticks([])
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.35)

        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
        else:
            all_vals = np.concatenate([v for v in data if len(v) > 0]) if any(len(v) for v in data) else np.array([])
            if len(all_vals) > 0:
                fallback_ylim = robust_shared_ylim_from_values(
                    all_vals,
                    percentiles=(1, 99),
                    pad_frac=0.15,
                    nonnegative=False,
                )
                if fallback_ylim is not None:
                    ax.set_ylim(*fallback_ylim)

        if i == 0:
            ax.set_xticks(np.arange(1, len(SEROTYPES_ORDER) + 1))
            ax.set_xticklabels(SEROTYPES_ORDER, fontsize=13)
            ax.xaxis.tick_top()
            ax.tick_params(axis='x', pad=3)
        else:
            ax.set_xticks([])
    return axes


def _draw_kw_pvalue_panel(fig, spec, kw, regions_order):
    ax = fig.add_subplot(spec)
    p_col = 'p_fdr' if 'p_fdr' in kw.columns else 'p'
    work = kw.set_index('region').reindex(regions_order)
    pvals = work[p_col].astype(float).values.reshape(-1, 1)
    values = _safe_neglog10(pvals)
    im = ax.imshow(values, aspect='auto', cmap='Blues', vmin=0, vmax=300)
    ax.set_title('Kruskal-Wallis', fontsize=13, pad=8)
    ax.set_xticks([0])
    ax.set_xticklabels([''], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order, fontsize=13)
    ax.tick_params(length=0)
    for r, p in enumerate(pvals[:, 0]):
        ax.text(0, r, _stars(p), ha='center', va='center', fontsize=11, color='black')
    ax.set_xticks(np.arange(-.5, 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(regions_order), 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.4, alpha=0.45)
    return ax, im


def _draw_pairwise_pvalue_panel(fig, spec, pairwise, regions_order):
    ax = fig.add_subplot(spec)
    p_col = 'p_fdr' if 'p_fdr' in pairwise.columns else 'p'
    pmat = _pivot_pairwise(pairwise, p_col, regions_order)
    values = _safe_neglog10(pmat.values)
    im = ax.imshow(values, aspect='auto', cmap='Blues', vmin=0, vmax=300)
    ax.set_title('Pairwise post-hoc', fontsize=13, pad=8)
    xticks = list(range(len(pmat.columns)))
    labels = [PAIRWISE_SHORT[PAIRWISE_ORDER.index(c)] if c in PAIRWISE_ORDER else str(c) for c in pmat.columns]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order, fontsize=13)
    ax.tick_params(length=0)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, _stars(pmat.iloc[r, c]), ha='center', va='center', fontsize=10, color='black')
    ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(regions_order), 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.4, alpha=0.45)
    return ax, im


def _draw_kw_effect_panel(fig, spec, kw, regions_order):
    ax = fig.add_subplot(spec)
    effect_col = 'epsilon_squared_R'
    if effect_col not in kw.columns:
        raise ValueError(f'Missing {effect_col} in kruskal_by_region.csv')
    work = kw.set_index('region').reindex(regions_order)
    values = work[effect_col].astype(float).values.reshape(-1, 1)

    # Use discrete bins for the conventional effect-size magnitudes rather than
    # a continuous 0-1 colorbar. This makes small/medium/large effects visible
    # even when all values are close to zero.
    boundaries = [0.0, KW_EPS_THRESHOLDS[0], KW_EPS_THRESHOLDS[1], KW_EPS_THRESHOLDS[2], 1.0]
    norm = BoundaryNorm(boundaries, ncolors=256, clip=True)
    im = ax.imshow(values, aspect='auto', cmap='Blues', norm=norm)
    ax.set_title('Kruskal-Wallis ε²', fontsize=13, pad=8)
    ax.set_xticks([0])
    ax.set_xticklabels([''], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order, fontsize=13)
    ax.tick_params(length=0)
    for r, val in enumerate(values[:, 0]):
        txt = _effect_stars(val, KW_EPS_THRESHOLDS, signed=False)
        ax.text(0, r, txt, ha='center', va='center', fontsize=11, color='black')
    ax.set_xticks(np.arange(-.5, 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(regions_order), 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.4, alpha=0.45)
    return ax, im


def _draw_pairwise_effect_panel(fig, spec, pairwise, regions_order):
    ax = fig.add_subplot(spec)
    value_col = 'rank_biserial'
    if value_col not in pairwise.columns:
        if 'cliffs_delta' in pairwise.columns:
            value_col = 'cliffs_delta'
        else:
            raise ValueError('Missing rank_biserial/cliffs_delta in pairwise_serotype_stats_by_region.csv')
    mat = _pivot_pairwise(pairwise, value_col, regions_order)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(mat.values, aspect='auto', cmap='coolwarm', norm=norm)
    ax.set_title('Pairwise rank-biserial', fontsize=13, pad=8)
    xticks = list(range(len(mat.columns)))
    labels = [PAIRWISE_SHORT[PAIRWISE_ORDER.index(c)] if c in PAIRWISE_ORDER else str(c) for c in mat.columns]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order, fontsize=13)
    ax.tick_params(length=0)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat.iloc[r, c]
            txt = _effect_stars(val, PAIRWISE_RB_THRESHOLDS, signed=True)
            ax.text(c, r, txt, ha='center', va='center', fontsize=10, color='black')
    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(regions_order), 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.4, alpha=0.45)
    return ax, im


def plot_pvalue_panel(df, kw, pairwise, regions_order, title=None, value_col='attention'):
    fig = plt.figure(figsize=(15.5, max(8.0, 0.72 * len(regions_order))))
    gs = gridspec.GridSpec(1, 4, width_ratios=[5.0, 1.0, 5.9, 0.3], wspace=0.35)
    _draw_box_panel(fig, gs[0], df, regions_order, value_col=value_col)
    _, im1 = _draw_kw_pvalue_panel(fig, gs[1], kw, regions_order)
    _, im2 = _draw_pairwise_pvalue_panel(fig, gs[2], pairwise, regions_order)
    cax = fig.add_subplot(gs[3])
    fig.colorbar(im2, cax=cax, label=r'$-\log_{10}(p_{FDR})$')
    if title:
        fig.suptitle(title, fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.04)
    return fig


def plot_effect_size_panel(df, kw, pairwise, regions_order, title=None, value_col='attention'):
    fig = plt.figure(figsize=(17.8, max(8.0, 0.72 * len(regions_order))))

    # Keep the scientific panels separated from the legends.  The previous
    # central colorbar made the KW legend collide visually with the pairwise
    # heatmap.  Here both legends are placed in a reserved right-side area,
    # stacked vertically, because they encode different quantities:
    #   1) Kruskal-Wallis epsilon-squared magnitude classes;
    #   2) signed pairwise rank-biserial correlation.
    gs = gridspec.GridSpec(1, 3, width_ratios=[5.0, 1.05, 6.0], wspace=0.42)
    _draw_box_panel(fig, gs[0], df, regions_order, value_col=value_col)
    _, im_kw = _draw_kw_effect_panel(fig, gs[1], kw, regions_order)
    _, im_pair = _draw_pairwise_effect_panel(fig, gs[2], pairwise, regions_order)

    if title:
        fig.suptitle(title, fontsize=16, y=0.995)

    # Reserve enough space on the right before adding the two legend axes.
    fig.subplots_adjust(left=0.06, right=0.84, top=0.92, bottom=0.09)

    cax_kw = fig.add_axes([0.875, 0.58, 0.020, 0.30])
    cb_kw = fig.colorbar(
        im_kw,
        cax=cax_kw,
        boundaries=[0.0, KW_EPS_THRESHOLDS[0], KW_EPS_THRESHOLDS[1], KW_EPS_THRESHOLDS[2], 1.0],
        ticks=[
            KW_EPS_THRESHOLDS[0] / 2.0,
            (KW_EPS_THRESHOLDS[0] + KW_EPS_THRESHOLDS[1]) / 2.0,
            (KW_EPS_THRESHOLDS[1] + KW_EPS_THRESHOLDS[2]) / 2.0,
            (KW_EPS_THRESHOLDS[2] + 1.0) / 2.0,
        ],
    )
    cb_kw.ax.set_yticklabels(['<0.01', '* ≥0.01', '** ≥0.06', '*** ≥0.14'])
    cb_kw.ax.tick_params(labelsize=9)
    cb_kw.set_label('Kruskal-Wallis ε²', fontsize=10, labelpad=8)

    cax_pair = fig.add_axes([0.940, 0.18, 0.020, 0.64])
    cb_pair = fig.colorbar(im_pair, cax=cax_pair)
    cb_pair.set_ticks([-1.0, -0.43, -0.28, -0.11, 0.0, 0.11, 0.28, 0.43, 1.0])
    cb_pair.ax.tick_params(labelsize=9)
    cb_pair.set_label('Pairwise rank-biserial', fontsize=10, labelpad=8)

    fig.text(
        0.45,
        0.018,
        "Cell annotations indicate effect-size magnitude: * small, ** medium, *** large. "
        "KW ε² thresholds: 0.01/0.06/0.14; pairwise thresholds use |rank-biserial|: 0.11/0.28/0.43.",
        ha='center',
        va='bottom',
        fontsize=10,
    )
    return fig




def _ordered_regions(values, regions_order):
    cat = pd.Categorical(values, categories=regions_order, ordered=True)
    return cat


def _significance_from_p(p):
    return _stars(float(p)) if pd.notna(p) else ''


def _effect_magnitude_label(value, thresholds, signed=False):
    if pd.isna(value) or not np.isfinite(float(value)):
        return ''
    mag = abs(float(value)) if signed else float(value)
    if mag >= thresholds[2]:
        return 'large'
    if mag >= thresholds[1]:
        return 'medium'
    if mag >= thresholds[0]:
        return 'small'
    return 'negligible'


def _prepare_kruskal_p_sheet(kw, regions_order):
    work = kw.copy()
    if 'region' not in work.columns:
        work = work.rename(columns={work.columns[0]: 'region'})
    work['_region_order'] = _ordered_regions(work['region'], regions_order)
    work = work.sort_values('_region_order').drop(columns=['_region_order'])

    preferred = ['region', 'H', 'statistic', 'p', 'p_fdr', 'p_value', 'p_adjusted']
    cols = [c for c in preferred if c in work.columns]
    extra = [c for c in work.columns if c not in cols and c.lower().startswith('p')]
    cols = cols + extra
    if 'region' not in cols:
        cols = ['region'] + cols
    out = work[cols].copy()

    p_col = 'p_fdr' if 'p_fdr' in out.columns else ('p' if 'p' in out.columns else None)
    if p_col is not None:
        out[f'neg_log10_{p_col}'] = _safe_neglog10(out[p_col].astype(float).values)
        out['significance'] = out[p_col].apply(_significance_from_p)
    return out


def _prepare_pairwise_p_sheet(pairwise, regions_order):
    work = pairwise.copy()
    work['_region_order'] = _ordered_regions(work['region'], regions_order)
    if 'comparison' in work.columns:
        comp_order = {c: i for i, c in enumerate(PAIRWISE_ORDER)}
        work['_comparison_order'] = work['comparison'].map(comp_order).fillna(999).astype(int)
    else:
        work['_comparison_order'] = 0

    work = work.sort_values(['_region_order', '_comparison_order']).drop(columns=['_region_order', '_comparison_order'])
    preferred = ['region', 'comparison', 'p', 'p_fdr', 'p_value', 'p_adjusted', 'U', 'statistic']
    cols = [c for c in preferred if c in work.columns]
    extra = [c for c in work.columns if c not in cols and c.lower().startswith('p')]
    cols = cols + extra
    if 'region' not in cols:
        cols = ['region'] + cols
    if 'comparison' in work.columns and 'comparison' not in cols:
        cols.insert(1, 'comparison')
    out = work[cols].copy()

    p_col = 'p_fdr' if 'p_fdr' in out.columns else ('p' if 'p' in out.columns else None)
    if p_col is not None:
        out[f'neg_log10_{p_col}'] = _safe_neglog10(out[p_col].astype(float).values)
        out['significance'] = out[p_col].apply(_significance_from_p)
    return out


def _prepare_kruskal_effect_sheet(kw, regions_order):
    work = kw.copy()
    if 'region' not in work.columns:
        work = work.rename(columns={work.columns[0]: 'region'})
    work['_region_order'] = _ordered_regions(work['region'], regions_order)
    work = work.sort_values('_region_order').drop(columns=['_region_order'])

    preferred = ['region', 'epsilon_squared_R', 'eta_squared_H', 'H', 'statistic', 'n_total', 'n_groups']
    cols = [c for c in preferred if c in work.columns]
    if 'region' not in cols:
        cols = ['region'] + cols
    out = work[cols].copy()
    if 'epsilon_squared_R' in out.columns:
        out['epsilon_squared_R_magnitude'] = out['epsilon_squared_R'].apply(
            lambda v: _effect_magnitude_label(v, KW_EPS_THRESHOLDS, signed=False)
        )
        out['epsilon_squared_R_stars'] = out['epsilon_squared_R'].apply(
            lambda v: _effect_stars(v, KW_EPS_THRESHOLDS, signed=False)
        )
    return out


def _prepare_pairwise_effect_sheet(pairwise, regions_order):
    work = pairwise.copy()
    work['_region_order'] = _ordered_regions(work['region'], regions_order)
    if 'comparison' in work.columns:
        comp_order = {c: i for i, c in enumerate(PAIRWISE_ORDER)}
        work['_comparison_order'] = work['comparison'].map(comp_order).fillna(999).astype(int)
    else:
        work['_comparison_order'] = 0
    work = work.sort_values(['_region_order', '_comparison_order']).drop(columns=['_region_order', '_comparison_order'])

    effect_col = 'rank_biserial' if 'rank_biserial' in work.columns else 'cliffs_delta'
    preferred = [
        'region', 'comparison', effect_col, 'cliffs_delta', 'rank_biserial',
        'VDA', 'VDA_folded', 'common_language_effect_size', 'p', 'p_fdr'
    ]
    cols = []
    for c in preferred:
        if c in work.columns and c not in cols:
            cols.append(c)
    if 'region' not in cols:
        cols = ['region'] + cols
    if 'comparison' in work.columns and 'comparison' not in cols:
        cols.insert(1, 'comparison')
    out = work[cols].copy()
    if effect_col in out.columns:
        out[f'{effect_col}_abs'] = out[effect_col].abs()
        out[f'{effect_col}_magnitude'] = out[effect_col].apply(
            lambda v: _effect_magnitude_label(v, PAIRWISE_RB_THRESHOLDS, signed=True)
        )
        out[f'{effect_col}_stars'] = out[effect_col].apply(
            lambda v: _effect_stars(v, PAIRWISE_RB_THRESHOLDS, signed=True)
        )
    return out


def export_stat_panel_excel(kw, pairwise, regions_order, out_dir):
    """Export the four numerical tables behind the stat-panel figures."""
    xlsx_path = os.path.join(out_dir, 'attention_region_stats_raw.xlsx')

    sheets = {
        'kruskal_pvalues': _prepare_kruskal_p_sheet(kw, regions_order),
        'pairwise_pvalues': _prepare_pairwise_p_sheet(pairwise, regions_order),
        'kruskal_effect_size': _prepare_kruskal_effect_sheet(kw, regions_order),
        'pairwise_effect_size': _prepare_pairwise_effect_sheet(pairwise, regions_order),
    }

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            # Best-effort column sizing; supported by the openpyxl engine used by
            # pandas for .xlsx export. If a future engine lacks column_dimensions,
            # this will fail loudly instead of silently producing a malformed file.
            for idx, column in enumerate(table.columns, start=1):
                max_len = max([len(str(column))] + [len(str(v)) for v in table[column].head(200).values])
                width = min(max(max_len + 2, 10), 32)
                col_letter = worksheet.cell(row=1, column=idx).column_letter
                worksheet.column_dimensions[col_letter].width = width
            worksheet.freeze_panes = 'A2'

    print(f'Saved -> {xlsx_path}')
    return xlsx_path

def main():
    args = parse_args()
    print_args(args)

    k_type, _ = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)

    box_root = build_xai_output_dir(
        paths.logs_dir,
        'attention_box',
        'denformer',
        args.pooling,
        k_type,
        run_suffix,
        args.epochs,
    )
    box_dataset_dir = os.path.join(box_root, 'output-box-dataset')
    out_dir = build_xai_output_dir(
        paths.logs_dir,
        'attention_stat_panel',
        'denformer',
        args.pooling,
        k_type,
        run_suffix,
        args.epochs,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f'Input attention box dataset dir: {box_dataset_dir}')
    print(f'Output stat panel dir: {out_dir}')

    df, kw, pairwise, regions_order = _load_inputs(box_dataset_dir)

    title_prefix = f'{args.run_name} | denformer {args.pooling} | raw attention regional statistics'

    fig = plot_pvalue_panel(df, kw, pairwise, regions_order, title=title_prefix + ' | p-values')
    for ext in ['png', 'pdf']:
        path = os.path.join(out_dir, f'attention_region_stats_pvalues_raw.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved -> {path}')
    plt.close(fig)

    fig = plot_effect_size_panel(df, kw, pairwise, regions_order, title=title_prefix + ' | effect sizes')
    for ext in ['png', 'pdf']:
        path = os.path.join(out_dir, f'attention_region_stats_effect_size_raw.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved -> {path}')
    plt.close(fig)

    excel_path = export_stat_panel_excel(kw, pairwise, regions_order, out_dir)

    summary_path = os.path.join(out_dir, 'attention_stat_panel_summary.csv')
    pd.DataFrame([{
        'box_dataset_dir': box_dataset_dir,
        'output_dir': out_dir,
        'excel_path': excel_path,
        'n_rows': len(df),
        'n_regions': len(regions_order),
        'kruskal_effect_size': 'epsilon_squared_R',
        'pairwise_effect_size': 'rank_biserial',
        'pvalue_column': 'p_fdr if available, else p',
    }]).to_csv(summary_path, index=False)
    print(f'Saved -> {summary_path}')


if __name__ == '__main__':
    main()
