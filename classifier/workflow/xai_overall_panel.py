import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from classifier.config import LONGEST_SEQUENCE_LENGTH
from classifier.utils_genome_map import get_gene_boundaries
from classifier.workflow import config
from classifier.workflow.utils import (
    build_model_dir,
    build_xai_output_dir,
    get_run_suffix,
    parse_run_args,
    resolve_k_type_and_emb_dim,
)
from classifier.utils import get_args, print_args

SMOOTH_WINDOWS = (None, 51, 101, 201)
RECAP_1X3_DIR_NAME = 'recap_1x3'
RECAP_4X3_DIR_NAME = 'recap_4x3'
SEROTYPES_ORDER = ('DENV1', 'DENV2', 'DENV3', 'DENV4')
SEROTYPE_FILE_ALIASES = {
    'DENV1': ('DENV-1', 'DENV1'),
    'DENV2': ('DENV-2', 'DENV2'),
    'DENV3': ('DENV-3', 'DENV3'),
    'DENV4': ('DENV-4', 'DENV4'),
}
SPLIT_RUN_NAMES = ('cdhit', 'continent', 'timebin')
SPLIT_TITLES = {
    'cdhit': 'CD-HIT cluster-aware folds',
    'continent': 'Geographical distribution by continent',
    'timebin': 'Temporal distribution',
}


def _split_run_suffix(split_name):
    return f'_{split_name}'


def _split_panel_dir_name(k_type, split_name, epochs):
    return f'{k_type}{_split_run_suffix(split_name)}_e{epochs}'


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def _safe_profile(values):
    arr = np.asarray(values, dtype=float)
    return arr


def _moving_average_ignore_nan(values, window):
    if window is None or int(window) <= 1:
        return _safe_profile(values)
    window = int(window)
    if window % 2 == 0:
        raise ValueError('smooth_window must be odd for centered smoothing.')
    arr = _safe_profile(values)
    valid = np.isfinite(arr).astype(float)
    arr0 = np.where(np.isfinite(arr), arr, 0.0)
    kernel = np.ones(window, dtype=float)
    summed = np.convolve(arr0, kernel, mode='same')
    counts = np.convolve(valid, kernel, mode='same')
    out = np.divide(summed, counts, out=np.full_like(arr, np.nan, dtype=float), where=counts > 0)
    return out


def _renormalize_0_1(values):
    arr = _safe_profile(values)
    valid = np.isfinite(arr)
    if not valid.any():
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax <= vmin:
        out = np.zeros_like(arr, dtype=float)
        out[~valid] = np.nan
        return out
    return (arr - vmin) / (vmax - vmin)


def _load_gene_boundaries(k):
    genome_end = LONGEST_SEQUENCE_LENGTH - (int(k) - 1)
    map_file = os.path.join(paths.msa_refseq_map_dir, 'coordinates_dengue_LONGEST.csv')
    map_df = pd.read_csv(map_file)
    return get_gene_boundaries(map_df, gene_name='Proteina', genome_end=genome_end), genome_end


def _profile_path(xai_kind, pooling, k_type, run_suffix, epochs):
    root = build_xai_output_dir(paths.logs_dir, xai_kind, 'denformer', pooling, k_type, run_suffix, epochs)
    return os.path.join(root, 'output-aggregate-dataset', 'overall_dataset_Overall_sum.npy')


def _class_profile_path(xai_kind, pooling, k_type, run_suffix, epochs, serotype):
    root = build_xai_output_dir(paths.logs_dir, xai_kind, 'denformer', pooling, k_type, run_suffix, epochs)
    return os.path.join(root, 'output-aggregate-dataset', f'class_{serotype}_sum.npy')


def _resolve_class_profile_path(xai_kind, pooling, k_type, run_suffix, epochs, serotype):
    """Return the existing class profile path for a display serotype.

    Dataset-level aggregate files are written using classifier.config.CLASS_DICT,
    which currently uses DENV-1, DENV-2, ... with a hyphen.  The recap figure
    uses cleaner row labels DENV1, DENV2, ... .  Try both spellings so the
    panel works with current outputs and with any older no-hyphen outputs.
    """
    aliases = SEROTYPE_FILE_ALIASES.get(serotype, (serotype,))
    candidates = [
        _class_profile_path(xai_kind, pooling, k_type, run_suffix, epochs, alias)
        for alias in aliases
    ]
    for path in candidates:
        if os.path.exists(path):
            return path, candidates
    return candidates[0], candidates


def _format_profile(profile, smooth_window, target_len):
    values = _moving_average_ignore_nan(profile, smooth_window)
    if smooth_window is not None:
        values = _renormalize_0_1(values)
    return values[:target_len]


def _style_profile_axis(ax, ylabel, display_end):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18, labelpad=8)
    _apply_tight_xlim(ax, display_end)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=16)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_gene_background(ax, gene_boundaries, colors, gene_handles=None, gene_labels=None):
    ordered_boundaries = sorted(gene_boundaries.items(), key=lambda item: item[1][0])
    for i, (gene, (start, end)) in enumerate(ordered_boundaries):
        span = ax.axvspan(start, end, color=colors(i), alpha=0.3, label=gene, lw=0)
        if gene_handles is not None and gene_labels is not None and gene not in gene_labels:
            gene_handles.append(span)
            gene_labels.append(gene)


def _load_required(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {label} profile: {path}')
    return np.load(path)


def plot_overall_panel(attn_profile, gxi_profile, out_dir, args, smooth_window=None):
    os.makedirs(out_dir, exist_ok=True)
    gene_boundaries, genome_end = _load_gene_boundaries(args.k)

    target_len = min(len(attn_profile), len(gxi_profile), genome_end)
    x = np.arange(target_len)
    attn = _format_profile(attn_profile, smooth_window, target_len)
    gxi = _format_profile(gxi_profile, smooth_window, target_len)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 5.2),
        sharex=True,
        gridspec_kw={'hspace': 0.10},
    )

    colors = plt.get_cmap('tab20', len(gene_boundaries))
    gene_handles, gene_labels = [], []
    profiles = [
        ('Attention', attn),
        ('GxI', gxi),
    ]

    for ax, (ylabel, profile) in zip(axes, profiles):
        _add_gene_background(ax, gene_boundaries, colors, gene_handles, gene_labels)
        ax.plot(x, profile, lw=2, color='black')
        _style_profile_axis(ax, ylabel, genome_end)

    axes[-1].set_xlabel('Genomic RNA (Position)', fontsize=18, labelpad=6)

    fig.legend(
        gene_handles,
        gene_labels,
        loc='center left',
        bbox_to_anchor=(0.915, 0.5),
        ncol=1,
        fontsize=16,
        title='Region',
        title_fontsize=18,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])

    suffix = '' if smooth_window is None else f'_smoothed_{smooth_window}'
    out_png = os.path.join(out_dir, f'overall_attention_gxi_panel{suffix}.png')
    out_pdf = os.path.join(out_dir, f'overall_attention_gxi_panel{suffix}.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_png}')
    print(f'Saved -> {out_pdf}')
    return out_png

def _panel_filename(smooth_window=None):
    suffix = '' if smooth_window is None else f'_smoothed_{smooth_window}'
    return f'overall_attention_gxi_panel{suffix}.png'


def _recap_filename(smooth_window=None, cut=False):
    smooth_suffix = '' if smooth_window is None else f'_smoothed_{smooth_window}'
    cut_suffix = '_cut' if cut else ''
    return f'overall_attention_gxi_panel{smooth_suffix}{cut_suffix}_recap_1x3.png'


def _recap_4x3_filename(smooth_window=None, cut=False):
    smooth_suffix = '' if smooth_window is None else f'_smoothed_{smooth_window}'
    cut_suffix = '_cut' if cut else ''
    return f'serotype_attention_gxi_panel{smooth_suffix}{cut_suffix}_recap_4x3.png'


def _find_split_panel_dirs(model_panel_root, k_type, epochs):
    """Return the split panel directories needed for the manuscript 1x3 recap."""
    return {
        split: os.path.join(model_panel_root, _split_panel_dir_name(k_type, split, epochs))
        for split in SPLIT_RUN_NAMES
    }


def _last_finite_profile_len(path):
    """Return the last finite coordinate + 1 for a saved profile.

    Some aggregate profiles are padded to the longest reference length and have
    trailing NaNs for shorter serotype sequences.  Using len(profile) would keep
    a visually empty tail in *_cut figures.
    """
    arr = np.load(path, mmap_mode='r')
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        return 0
    return int(finite_idx[-1]) + 1


def _get_min_serotype_profile_len(args, k_type):
    """Return the shortest non-empty serotype-level dataset profile length.

    This is used for the *_cut recap panels, so all columns stop at the same
    genomic coordinate where all DENV1-DENV4 class profiles still have finite
    data.  This avoids blank/padded tails at the right edge of the plots.
    """
    _, genome_end = _load_gene_boundaries(args.k)
    lengths = []
    missing = []
    for split in SPLIT_RUN_NAMES:
        run_suffix = _split_run_suffix(split)
        for serotype in SEROTYPES_ORDER:
            for xai_kind in ('attention_aggregate', 'gxi_aggregate'):
                path, candidates = _resolve_class_profile_path(
                    xai_kind, args.pooling, k_type, run_suffix, args.epochs, serotype
                )
                if not os.path.exists(path):
                    missing.append(' OR '.join(candidates))
                    continue
                finite_len = _last_finite_profile_len(path)
                if finite_len <= 0:
                    missing.append(f'No finite values in {path}')
                else:
                    lengths.append(finite_len)

    if missing:
        return None, missing
    if not lengths:
        return None, ['No serotype-level dataset profiles were found.']
    return min(min(lengths), genome_end), []


def _add_top_legends(fig, gene_handles, gene_labels, signal_handles=None, top_y=0.985):
    """Place readable, centered two-line legends across the top of recap figures."""
    if signal_handles is not None:
        fig.legend(
            signal_handles,
            ['Attention', 'GxI'],
            loc='upper center',
            bbox_to_anchor=(0.5, top_y),
            ncol=2,
            fontsize=22,
            title='Signal',
            title_fontsize=23,
            frameon=False,
            handlelength=3.2,
            columnspacing=2.0,
            handletextpad=0.7,
        )
        region_y = top_y - 0.060
    else:
        region_y = top_y

    fig.legend(
        gene_handles,
        gene_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, region_y),
        ncol=len(gene_labels),
        fontsize=20,
        title='Region',
        title_fontsize=23,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.85,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


def _apply_tight_xlim(ax, display_end):
    """Use a tight x-range without leaving a blank sliver after the last point."""
    right = max(int(display_end) - 1, 1)
    ax.set_xlim(0, right)


def _add_serotype_group_labels(fig, axes, n_signal_rows):
    """Place one centered DENV label for each Attention/GxI row pair."""
    first_col_x0 = axes[0, 0].get_position().x0
    label_x = max(first_col_x0 - 0.010, 0.004)
    for serotype_idx, serotype in enumerate(SEROTYPES_ORDER):
        top_ax = axes[serotype_idx * 2, 0]
        bottom_ax = axes[serotype_idx * 2 + 1, 0]
        top_box = top_ax.get_position()
        bottom_box = bottom_ax.get_position()
        y = 0.5 * (top_box.y1 + bottom_box.y0)
        fig.text(
            label_x,
            y,
            serotype,
            va='center',
            ha='right',
            rotation=90,
            fontsize=22,
        )


def build_overall_panel_recap_1x3(model_panel_root, args, k_type, smooth_window=None, cut=False):
    """
    Build a clean 1x3 recap from dataset-level profiles for the three split strategies.

    This reads the already-computed attention/GxI aggregate .npy files only. It does
    not recompute model inference, attention inference, GxI, or aggregate profiles.
    """
    gene_boundaries, genome_end = _load_gene_boundaries(args.k)
    display_end = genome_end
    if cut:
        display_end, cut_missing = _get_min_serotype_profile_len(args, k_type)
        if display_end is None:
            print('Skipping recap 1x3 cut; missing required serotype-level dataset profile(s):')
            for path in cut_missing:
                print(f'  - {path}')
            return None

    split_profiles = []
    missing = []
    for split in SPLIT_RUN_NAMES:
        run_suffix = _split_run_suffix(split)
        attn_path = _profile_path('attention_aggregate', args.pooling, k_type, run_suffix, args.epochs)
        gxi_path = _profile_path('gxi_aggregate', args.pooling, k_type, run_suffix, args.epochs)
        if not os.path.exists(attn_path):
            missing.append(attn_path)
        if not os.path.exists(gxi_path):
            missing.append(gxi_path)
        if not missing or (os.path.exists(attn_path) and os.path.exists(gxi_path)):
            if os.path.exists(attn_path) and os.path.exists(gxi_path):
                attn_profile = np.load(attn_path)
                gxi_profile = np.load(gxi_path)
                target_len = min(len(attn_profile), len(gxi_profile), display_end)
                split_profiles.append(
                    {
                        'split': split,
                        'x': np.arange(target_len),
                        'attn': _format_profile(attn_profile, smooth_window, target_len),
                        'gxi': _format_profile(gxi_profile, smooth_window, target_len),
                    }
                )

    if missing:
        print('Skipping recap 1x3; missing required dataset-level profile(s):')
        for path in missing:
            print(f'  - {path}')
        return None

    if len(split_profiles) != len(SPLIT_RUN_NAMES):
        print('Skipping recap 1x3; not all split profiles were loaded.')
        return None

    recap_dir = os.path.join(model_panel_root, RECAP_1X3_DIR_NAME)
    os.makedirs(recap_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(31.0, 5.7),
        sharex=False,
        sharey=True,
        gridspec_kw={'hspace': 0.10, 'wspace': 0.08},
    )

    colors = plt.get_cmap('tab20', len(gene_boundaries))
    gene_handles, gene_labels = [], []

    for col, item in enumerate(split_profiles):
        axes[0, col].set_title(SPLIT_TITLES.get(item['split'], item['split']), fontsize=22, pad=12)

        for row, (ylabel, values_key) in enumerate((('Attention', 'attn'), ('GxI', 'gxi'))):
            ax = axes[row, col]
            handles = gene_handles if col == 0 and row == 0 else None
            labels = gene_labels if col == 0 and row == 0 else None
            _add_gene_background(ax, gene_boundaries, colors, handles, labels)
            ax.plot(item['x'], item[values_key], lw=1.9, color='black')
            _style_profile_axis(ax, ylabel, display_end)
            if row != 1:
                ax.tick_params(axis='x', labelbottom=False)

    fig.supxlabel('Genomic RNA position', fontsize=19, y=0.018)
    _add_top_legends(fig, gene_handles, gene_labels, signal_handles=None, top_y=0.985)
    plt.tight_layout(rect=[0, 0.045, 1.0, 0.845])

    out_png = os.path.join(recap_dir, _recap_filename(smooth_window, cut=cut))
    out_pdf = out_png.replace('.png', '.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved recap 1x3 -> {out_png}')
    print(f'Saved recap 1x3 -> {out_pdf}')
    return out_png


def build_serotype_panel_recap_4x3(model_panel_root, args, k_type, smooth_window=None, cut=False):
    """
    Build a serotype-specific recap that mirrors the 1x3 overall recap.

    The overall recap is a 2x3 panel: rows are Attention/GxI and columns are
    the three split strategies.  This serotype recap keeps the same visual
    grammar and expands the rows to DENV1-DENV4, each with two signal rows:
    DENV1 Attention, DENV1 GxI, ..., DENV4 Attention, DENV4 GxI.

    Inputs are the already generated dataset-level serotype aggregate profiles:
      - attention_aggregate/.../output-aggregate-dataset/class_DENV-*_sum.npy
      - gxi_aggregate/.../output-aggregate-dataset/class_DENV-*_sum.npy
    The loader also accepts legacy no-hyphen names such as class_DENV1_sum.npy.
    """
    gene_boundaries, genome_end = _load_gene_boundaries(args.k)
    display_end = genome_end
    if cut:
        display_end, cut_missing = _get_min_serotype_profile_len(args, k_type)
        if display_end is None:
            print('Skipping recap 4x3 cut; missing required serotype-level dataset profile(s):')
            for path in cut_missing:
                print(f'  - {path}')
            return None

    profiles = {}
    missing = []
    for split in SPLIT_RUN_NAMES:
        run_suffix = _split_run_suffix(split)
        for serotype in SEROTYPES_ORDER:
            attn_path, attn_candidates = _resolve_class_profile_path(
                'attention_aggregate', args.pooling, k_type, run_suffix, args.epochs, serotype
            )
            gxi_path, gxi_candidates = _resolve_class_profile_path(
                'gxi_aggregate', args.pooling, k_type, run_suffix, args.epochs, serotype
            )
            if not os.path.exists(attn_path):
                missing.append(' OR '.join(attn_candidates))
            if not os.path.exists(gxi_path):
                missing.append(' OR '.join(gxi_candidates))
            if os.path.exists(attn_path) and os.path.exists(gxi_path):
                attn_profile = np.load(attn_path)
                gxi_profile = np.load(gxi_path)
                target_len = min(len(attn_profile), len(gxi_profile), display_end)
                profiles[(serotype, split)] = {
                    'x': np.arange(target_len),
                    'attn': _format_profile(attn_profile, smooth_window, target_len),
                    'gxi': _format_profile(gxi_profile, smooth_window, target_len),
                }

    if missing:
        print('Skipping recap 4x3; missing required serotype-level dataset profile(s):')
        for path in missing:
            print(f'  - {path}')
        return None

    required = {(serotype, split) for serotype in SEROTYPES_ORDER for split in SPLIT_RUN_NAMES}
    if set(profiles) != required:
        print('Skipping recap 4x3; not all serotype/split profiles were loaded.')
        return None

    recap_dir = os.path.join(model_panel_root, RECAP_4X3_DIR_NAME)
    os.makedirs(recap_dir, exist_ok=True)

    n_signal_rows = len(SEROTYPES_ORDER) * 2
    fig, axes = plt.subplots(
        n_signal_rows,
        len(SPLIT_RUN_NAMES),
        figsize=(32.0, 17.6),
        sharex=False,
        sharey=True,
        gridspec_kw={'hspace': 0.065, 'wspace': 0.065},
    )

    colors = plt.get_cmap('tab20', len(gene_boundaries))
    gene_handles, gene_labels = [], []
    signal_handles = None

    row_specs = []
    for serotype in SEROTYPES_ORDER:
        row_specs.append((serotype, 'Attention', 'attn'))
        row_specs.append((serotype, 'GxI', 'gxi'))

    for row, (serotype, signal_label, values_key) in enumerate(row_specs):
        for col, split in enumerate(SPLIT_RUN_NAMES):
            ax = axes[row, col]
            item = profiles[(serotype, split)]
            handles = gene_handles if row == 0 and col == 0 else None
            labels = gene_labels if row == 0 and col == 0 else None
            _add_gene_background(ax, gene_boundaries, colors, handles, labels)

            linestyle = '-' if values_key == 'attn' else '--'
            line, = ax.plot(item['x'], item[values_key], lw=1.75, color='black', linestyle=linestyle)
            if signal_handles is None:
                attn_proxy, = ax.plot([], [], lw=2.0, color='black', linestyle='-')
                gxi_proxy, = ax.plot([], [], lw=2.0, color='black', linestyle='--')
                signal_handles = [attn_proxy, gxi_proxy]

            _style_profile_axis(ax, '', display_end)
            ax.tick_params(axis='x', labelsize=14)

            if row == 0:
                ax.set_title(SPLIT_TITLES.get(split, split), fontsize=22, pad=12)
            if row != n_signal_rows - 1:
                ax.tick_params(axis='x', labelbottom=False)

    fig.supxlabel('Genomic RNA position', fontsize=21, y=0.018)
    _add_top_legends(fig, gene_handles, gene_labels, signal_handles=signal_handles, top_y=0.985)
    plt.tight_layout(rect=[0.040, 0.045, 1.0, 0.825])
    _add_serotype_group_labels(fig, axes, n_signal_rows)

    out_png = os.path.join(recap_dir, _recap_4x3_filename(smooth_window, cut=cut))
    out_pdf = out_png.replace('.png', '.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved recap 4x3 -> {out_png}')
    print(f'Saved recap 4x3 -> {out_pdf}')
    return out_png

def main():
    args = parse_args()
    print_args(args)

    k_type, _ = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)

    # Raw model dir is printed only as orientation for the user; this script reads the
    # derived profiles from logs/aggregate_xai.
    model_dir = build_model_dir(paths.logs_dir, 'denformer', args.pooling, k_type, run_suffix, args.epochs)
    out_dir = build_xai_output_dir(paths.logs_dir, 'overall_panel', 'denformer', args.pooling, k_type, run_suffix, args.epochs)
    print(f'Model dir: {model_dir}')
    print(f'Panel output dir: {out_dir}')

    attn_path = _profile_path('attention_aggregate', args.pooling, k_type, run_suffix, args.epochs)
    gxi_path = _profile_path('gxi_aggregate', args.pooling, k_type, run_suffix, args.epochs)
    print(f'Attention profile: {attn_path}')
    print(f'GxI profile: {gxi_path}')

    attn_profile = _load_required(attn_path, 'attention overall')
    gxi_profile = _load_required(gxi_path, 'GxI overall')

    rows = []
    for window in SMOOTH_WINDOWS:
        out_png = plot_overall_panel(attn_profile, gxi_profile, out_dir, args, smooth_window=window)
        rows.append({'smooth_window': 'raw' if window is None else window, 'output_png': out_png})

    summary_path = os.path.join(out_dir, 'overall_attention_gxi_panel_summary.csv')
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')

    # Build/update the per-model 1x3 recap when all split-level panels are available.
    # out_dir points to .../overall_panel/<model_pooling>/<split>; the recap lives
    # one level above, i.e. .../overall_panel/<model_pooling>/recap_1x3.
    model_panel_root = os.path.dirname(out_dir)
    recap_rows = []
    for window in SMOOTH_WINDOWS:
        for cut in (False, True):
            recap_png = build_overall_panel_recap_1x3(
                model_panel_root, args, k_type, smooth_window=window, cut=cut
            )
            if recap_png is not None:
                recap_rows.append({
                    'smooth_window': 'raw' if window is None else window,
                    'cut': cut,
                    'output_png': recap_png,
                })

    if recap_rows:
        recap_summary_path = os.path.join(model_panel_root, RECAP_1X3_DIR_NAME, 'overall_attention_gxi_panel_recap_1x3_summary.csv')
        pd.DataFrame(recap_rows).to_csv(recap_summary_path, index=False)
        print(f'Saved recap summary -> {recap_summary_path}')

    serotype_recap_rows = []
    for window in SMOOTH_WINDOWS:
        for cut in (False, True):
            recap_png = build_serotype_panel_recap_4x3(
                model_panel_root, args, k_type, smooth_window=window, cut=cut
            )
            if recap_png is not None:
                serotype_recap_rows.append({
                    'smooth_window': 'raw' if window is None else window,
                    'cut': cut,
                    'output_png': recap_png,
                })

    if serotype_recap_rows:
        serotype_recap_summary_path = os.path.join(model_panel_root, RECAP_4X3_DIR_NAME, 'serotype_attention_gxi_panel_recap_4x3_summary.csv')
        pd.DataFrame(serotype_recap_rows).to_csv(serotype_recap_summary_path, index=False)
        print(f'Saved recap 4x3 summary -> {serotype_recap_summary_path}')


if __name__ == '__main__':
    main()
