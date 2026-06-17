import os
import re
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import mannwhitneyu, kruskal
import matplotlib.pyplot as plt


def robust_shared_ylim_from_values(values, percentiles=(1, 99), pad_frac=0.12, nonnegative=False):
    """Return robust y-limits from finite values using percentile clipping.

    This keeps boxplot panels readable while avoiding a few extreme values
    compressing the useful dynamic range. By default the lower limit is not
    forced to zero, because attention/GxI values often occupy a narrow positive
    interval and zero anchoring can flatten the boxes.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo_pct, hi_pct = percentiles
    lo, hi = np.nanpercentile(arr, [lo_pct, hi_pct])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        eps = abs(float(hi)) * 0.05 if hi != 0 else 1e-12
        lo, hi = float(lo) - eps, float(hi) + eps
    pad = (hi - lo) * float(pad_frac)
    y0, y1 = float(lo - pad), float(hi + pad)
    if nonnegative and np.nanmin(arr) >= 0 and y0 < 0:
        y0 = 0.0
    return y0, y1


def build_region_attention_long_df(
    input_dir: str,
    regions: dict,
    class_dict: dict,
    pattern: str = r"batch(\d+)_sample(\d+)_class(\d+)_attn\.npy",
    region_reduce: str = "mean",   # "mean" | "sum" | "median"
    normalize_per_sample: bool = False,  # optional: normalize each sample profile to sum=1
):
    """
    Convert per-sample attention profiles into region-level values in long/tidy format.

    Parameters
    ----------
    input_dir : str
        Directory with per-sample attention .npy files.
    regions : dict
        Mapping {region_name: (start, end)} using 0-based indexing, end exclusive.
        Example: {"C": (0, 400), "prM": (400, 900), ...}
        IMPORTANT: must match the length/coordinate system of your attention arrays.
    class_dict : dict
        Mapping from class index -> class name (e.g., {0:"DENV1",1:"DENV2",...})
    region_reduce : str
        How to reduce attention values inside a region to one number per sample.
    normalize_per_sample : bool
        If True, divides each sample's attention profile by its sum (so values become comparable as proportions).

    Returns
    -------
    pd.DataFrame with columns:
        batch, sample, class, serotype, region, attention
    """
    rx = re.compile(pattern)
    files = [f for f in os.listdir(input_dir) if f.endswith("_attn.npy")]

    reducers = {
        "mean": np.mean,
        "sum": np.sum,
        "median": np.median,
    }
    if region_reduce not in reducers:
        raise ValueError(f"region_reduce must be one of {list(reducers)}")

    reduce_fn = reducers[region_reduce]
    rows = []

    for fname in files:
        m = rx.search(fname)
        if not m:
            continue
        b, s, c = map(int, m.groups())
        serotype = class_dict.get(c, f"class {c}")

        arr = np.load(os.path.join(input_dir, fname)).astype(float)
        if arr.ndim != 1:
            raise ValueError(f"{fname}: expected 1D attention profile, got shape {arr.shape}")

        if normalize_per_sample:
            denom = arr.sum()
            if denom > 0:
                arr = arr / denom

        for region_name, (start, end) in regions.items():
            start = max(0, int(start))
            end = min(len(arr), int(end))
            if end <= start:
                continue
            val = float(reduce_fn(arr[start:end]))
            rows.append({
                "batch": b,
                "sample": s,
                "class": c,
                "serotype": serotype,
                "region": region_name,
                "attention": val,
                "file": fname,
            })

    return pd.DataFrame(rows)


def plot_attention_boxplots_by_region_rows(
    df: pd.DataFrame,
    region_col: str = "region",
    serotype_col: str = "serotype",
    value_col: str = "attention",
    regions_order=None,
    serotypes_order=None,
    colors=None,
    hide_yticks: bool = True,
    show_serotype_labels_once: bool = True,
    show_n: bool = False,
    whis=(5, 95),
    showfliers: bool = False,
    figsize_per_row=(8, 1.1),
    ypad_frac: float = 0.12,
    robust_ylim_percentiles=(1, 99),
    share_y_across_regions: bool = False,
    region_fontsize: int = 14,
    xtick_fontsize: int = 13,
    title_fontsize: int = 18,
    hspace: float = 0.0,        # try 0.0 or even -0.05 (can overlap if too small)
    top: float = 0.93,          # reduce if you want less headroom
    bottom: float = 0.03,
    left: float = 0.16,         # increase if region labels get clipped
    right: float = 0.99,
    x_tick_pad: float = 2,      # was 8; big contributor to vertical space
    hide_suptitle: bool = True,
):
    """
    One subplot per genomic region (rows). Each subplot contains boxplots for all serotypes.
    Optimized for many regions (compact vertical layout).
    """

    if regions_order is None:
        regions_order = list(pd.unique(df[region_col]))
    if serotypes_order is None:
        serotypes_order = list(pd.unique(df[serotype_col]))

    if colors is None:
        # DENV1, DENV2, DENV3, DENV4
        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

    nrows = len(regions_order)
    fig_w, fig_h_per = figsize_per_row
    fig = plt.figure(figsize=(fig_w, fig_h_per * nrows))
    axes = []

    shared_ylim = None
    if share_y_across_regions and robust_ylim_percentiles is not None:
        shared_ylim = robust_shared_ylim_from_values(
            df[value_col].dropna().values,
            percentiles=robust_ylim_percentiles,
            pad_frac=ypad_frac,
            nonnegative=False,
        )

    for i, region in enumerate(regions_order):
        ax = fig.add_subplot(nrows, 1, i + 1)
        axes.append(ax)

        data, ns = [], []
        for s in serotypes_order:
            vals = df[(df[region_col] == region) & (df[serotype_col] == s)][value_col].dropna().values
            data.append(vals)
            ns.append(len(vals))

        positions = np.arange(1, len(serotypes_order) + 1)

        if all(len(v) == 0 for v in data):
            ax.text(0.5, 0.5, f"{region}: no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            if hide_yticks:
                ax.set_yticks([])
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=showfliers,
            whis=whis,
            medianprops=dict(color="brown", linewidth=1.6),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        for box, color in zip(bp["boxes"], colors[:len(serotypes_order)]):
            box.set_facecolor(color)
            box.set_alpha(0.9)

        # Region label on the left
        ax.set_ylabel(region, fontsize=region_fontsize, rotation=0, labelpad=25, va="center")

        # X labels only on top row
        if show_serotype_labels_once:
            if i == 0:
                labels = []
                for s, n in zip(serotypes_order, ns):
                    labels.append(f"{s}\n(n={n})" if show_n else s)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=xtick_fontsize)
                ax.xaxis.tick_top()
                ax.tick_params(axis="x", pad=x_tick_pad)
            else:
                ax.set_xticks([])
        else:
            labels = []
            for s, n in zip(serotypes_order, ns):
                labels.append(f"{s}\n(n={n})" if show_n else s)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=xtick_fontsize)

        # Robust y-limits. By default, use a per-region 1st-99th percentile
        # range. This keeps each genomic row readable when different regions
        # have very different dynamic ranges.
        all_vals = np.concatenate([v for v in data if len(v) > 0])
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
        elif len(all_vals) > 0:
            if robust_ylim_percentiles is not None:
                ylim = robust_shared_ylim_from_values(
                    all_vals,
                    percentiles=robust_ylim_percentiles,
                    pad_frac=ypad_frac,
                    nonnegative=False,
                )
                if ylim is not None:
                    ax.set_ylim(*ylim)
            else:
                lo, hi = np.min(all_vals), np.max(all_vals)
                if hi > lo:
                    pad = (hi - lo) * ypad_frac
                    ax.set_ylim(lo - pad, hi + pad)

        if hide_yticks:
            ax.set_yticks([])

        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlim(0.4, len(serotypes_order) + 0.6)

        if i < nrows - 1:
            ax.set_xlabel("")

    if not hide_suptitle:
        fig.suptitle(
            "Attention distributions by genomic region and serotype",
            fontsize=title_fontsize,
            y=0.985
        )
    fig.subplots_adjust(hspace=hspace, top=top, bottom=bottom, left=left, right=right)

    return fig, np.array(axes)




def _clip_unit_interval(value):
    """Clip finite scalar effect sizes to [0, 1]; keep NaN unchanged."""
    if value is None or not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0.0, 1.0))


def _effect_size_label(value, thresholds):
    """
    Classify an absolute/non-negative effect size using ascending thresholds.
    thresholds example: ((0.01, 'small'), (0.06, 'medium'), (0.14, 'large')).
    """
    if value is None or not np.isfinite(value):
        return "NA"
    label = "negligible"
    for threshold, name in thresholds:
        if abs(value) >= threshold:
            label = name
    return label


def classify_rank_biserial(value):
    """Vargha-Delaney thresholds for rank-biserial / Cliff's delta."""
    return _effect_size_label(value, ((0.11, "small"), (0.28, "medium"), (0.43, "large")))


def classify_vda(value):
    """Classify VDA after folding around 0.5, so direction is handled separately."""
    if value is None or not np.isfinite(value):
        return "NA"
    folded = max(float(value), 1.0 - float(value))
    return _effect_size_label(folded, ((0.56, "small"), (0.64, "medium"), (0.71, "large")))


def classify_eta_epsilon(value):
    """Conventional eta/epsilon-squared thresholds."""
    return _effect_size_label(value, ((0.01, "small"), (0.06, "medium"), (0.14, "large")))

def kruskal_by_region(
    df: pd.DataFrame,
    region_col: str = "region",
    serotype_col: str = "serotype",
    value_col: str = "attention",
    regions_order=None,
    serotypes_order=None,
    min_n: int = 2,
    fdr_correct: bool = True,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Performs a Kruskal-Wallis H test for each region,
    testing whether any serotype differs within that region.

    Effect sizes:
      - eta_squared_H = (H - k + 1) / (N - k)
      - epsilon_squared_R = H / (N - 1)

    Returns a DataFrame with:
        index   = regions
        columns = n_groups, N, H, p, p_fdr, eta_squared_H,
                  epsilon_squared_R, effect_size_magnitude
    """

    if regions_order is None:
        regions_order = list(pd.unique(df[region_col]))
    if serotypes_order is None:
        serotypes_order = list(pd.unique(df[serotype_col]))

    results = []

    for region in regions_order:
        sub = df[df[region_col] == region]

        groups = []
        valid_serotypes = []

        for s in serotypes_order:
            values = sub[sub[serotype_col] == s][value_col].dropna().values
            if len(values) >= min_n:
                groups.append(values)
                valid_serotypes.append(s)

        n_groups = len(groups)
        N = int(sum(len(g) for g in groups))

        if n_groups < 2 or N <= n_groups:
            results.append({
                "region": region,
                "n_groups": n_groups,
                "N": N,
                "H": np.nan,
                "p": np.nan,
                "eta_squared_H": np.nan,
                "epsilon_squared_R": np.nan,
                "effect_size_magnitude": "NA",
            })
            continue

        H, p = kruskal(*groups)
        eta_squared_H = _clip_unit_interval((H - n_groups + 1.0) / (N - n_groups))
        epsilon_squared_R = _clip_unit_interval(H / (N - 1.0))

        results.append({
            "region": region,
            "n_groups": n_groups,
            "N": N,
            "H": float(H),
            "p": float(p),
            "eta_squared_H": eta_squared_H,
            "epsilon_squared_R": epsilon_squared_R,
            "effect_size_magnitude": classify_eta_epsilon(epsilon_squared_R),
        })

    out = pd.DataFrame(results).set_index("region")

    if fdr_correct and not out["p"].isna().all():
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            raise ImportError("statsmodels is required for FDR correction")

        mask = out["p"].notna()
        p_orig = out.loc[mask, "p"].values
        _, p_adj, _, _ = multipletests(p_orig, method=fdr_method)
        print(f"[kruskal] p_orig: {len(p_orig)}, p_adj: {len(p_adj)}")
        out.loc[mask, "p_fdr"] = p_adj

    sort_col = "p_fdr" if "p_fdr" in out.columns else "p"
    return out.sort_values(sort_col)

def truncated_cmap(cmap_name="Blues", minval=0.2, maxval=0.75, n=256):
    from matplotlib.colors import LinearSegmentedColormap
    """
    Create a truncated colormap to avoid very dark or very light extremes.
    minval/maxval are fractions of the original colormap.
    """
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc", colors
    )




def pairwise_serotype_stats_by_region(
    df: pd.DataFrame,
    region_col: str = "region",
    serotype_col: str = "serotype",
    value_col: str = "attention",
    regions_order=None,
    serotypes_order=None,
    alternative: str = "two-sided",
    min_n: int = 2,
    label_fmt: str = "{a}v{b}",
    fdr_correct: bool = True,
    fdr_method: str = "fdr_bh",
    fdr_scope: str = "global",  # "global" or "within_region"
) -> pd.DataFrame:
    """
    Mann-Whitney pairwise serotype comparisons by region with effect sizes.

    Effect sizes:
      - VDA = U / (n_a * n_b), where U is scipy's statistic for group a.
      - rank_biserial = Cliff's delta = 2 * VDA - 1.

    Positive rank_biserial means the first serotype in the pair tends to have
    higher values than the second; negative means the opposite.
    """
    if regions_order is None:
        regions_order = list(pd.unique(df[region_col]))
    if serotypes_order is None:
        serotypes_order = list(pd.unique(df[serotype_col]))

    rows = []
    pairs = list(combinations(serotypes_order, 2))

    for region in regions_order:
        sub = df[df[region_col] == region]
        for a, b in pairs:
            xa = sub[sub[serotype_col] == a][value_col].dropna().values
            xb = sub[sub[serotype_col] == b][value_col].dropna().values
            comparison = label_fmt.format(a=a, b=b)

            row = {
                "region": region,
                "serotype_a": a,
                "serotype_b": b,
                "comparison": comparison,
                "n_a": int(len(xa)),
                "n_b": int(len(xb)),
                "U": np.nan,
                "p": np.nan,
                "VDA": np.nan,
                "VDA_folded": np.nan,
                "rank_biserial": np.nan,
                "cliffs_delta": np.nan,
                "effect_size_magnitude": "NA",
                "dominant_serotype": "NA",
            }

            if len(xa) >= min_n and len(xb) >= min_n:
                U, p = mannwhitneyu(xa, xb, alternative=alternative)
                vda = float(U) / float(len(xa) * len(xb))
                rank_biserial = 2.0 * vda - 1.0
                row.update({
                    "U": float(U),
                    "p": float(p),
                    "VDA": vda,
                    "VDA_folded": max(vda, 1.0 - vda),
                    "rank_biserial": rank_biserial,
                    "cliffs_delta": rank_biserial,
                    "effect_size_magnitude": classify_rank_biserial(rank_biserial),
                    "dominant_serotype": a if rank_biserial > 0 else (b if rank_biserial < 0 else "tie"),
                })

            rows.append(row)

    out = pd.DataFrame(rows)

    if fdr_correct and not out["p"].isna().all():
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            raise ImportError("statsmodels is required for FDR correction")

        out["p_fdr"] = np.nan
        if fdr_scope == "global":
            mask = out["p"].notna()
            _, adj, _, _ = multipletests(out.loc[mask, "p"].values, method=fdr_method)
            out.loc[mask, "p_fdr"] = adj
        elif fdr_scope == "within_region":
            for region in regions_order:
                mask = (out["region"] == region) & out["p"].notna()
                if mask.sum() == 0:
                    continue
                _, adj, _, _ = multipletests(out.loc[mask, "p"].values, method=fdr_method)
                out.loc[mask, "p_fdr"] = adj
        else:
            raise ValueError("fdr_scope must be 'global' or 'within_region'")

    sort_cols = ["region", "p_fdr"] if "p_fdr" in out.columns else ["region", "p"]
    return out.sort_values(sort_cols)


def pairwise_stats_matrix(stats_df: pd.DataFrame, value_col: str, regions_order=None) -> pd.DataFrame:
    """Pivot long pairwise statistics to region x comparison matrix."""
    matrix = stats_df.pivot(index="region", columns="comparison", values=value_col)
    if regions_order is not None:
        matrix = matrix.reindex(regions_order)
    return matrix

def pvalues_pairwise_serotypes_by_region(
    df: pd.DataFrame,
    region_col: str = "region",
    serotype_col: str = "serotype",
    value_col: str = "attention",
    regions_order=None,
    serotypes_order=None,
    alternative: str = "two-sided",
    min_n: int = 2,
    label_fmt: str = "{a}v{b}",
    bh_correct: bool = False,
    bh_method: str = "fdr_bh",
    bh_scope: str = "within_region",  # "within_region" or "global"
    out_col_suffix: str = "_bh",
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      index   = regions
      columns = pairwise comparisons (e.g., DENV1vDENV2, ...)
      values  = Mann–Whitney U p-values comparing serotypes within each region

    If bh_correct=True, returns BH-adjusted p-values instead (or alongside, see below).
    BH scope options:
      - "within_region": BH is applied separately within each region across its pairwise tests.
      - "global": BH is applied once across all region x pairwise tests in the matrix.
    """
    if regions_order is None:
        regions_order = list(pd.unique(df[region_col]))
    if serotypes_order is None:
        serotypes_order = list(pd.unique(df[serotype_col]))

    pairs = list(combinations(serotypes_order, 2))
    colnames = [label_fmt.format(a=a, b=b) for a, b in pairs]

    pmat = pd.DataFrame(index=regions_order, columns=colnames, dtype=float)

    # Compute raw p-values
    for region in regions_order:
        sub = df[df[region_col] == region]

        for (a, b), col in zip(pairs, colnames):
            xa = sub[sub[serotype_col] == a][value_col].dropna().values
            xb = sub[sub[serotype_col] == b][value_col].dropna().values

            if len(xa) < min_n or len(xb) < min_n:
                pmat.loc[region, col] = np.nan
                continue

            _, p = mannwhitneyu(xa, xb, alternative=alternative)
            pmat.loc[region, col] = p

    if not bh_correct:
        return pmat

    # BH adjustment
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        raise ImportError("statsmodels is required for BH/FDR correction")

    p_adj = pmat.copy()

    if bh_scope == "within_region":
        # Adjust each region row separately
        for region in p_adj.index:
            row = p_adj.loc[region]
            mask = row.notna()
            if mask.sum() == 0:
                continue
            _, adj, _, _ = multipletests(row[mask].values, method=bh_method)
            p_adj.loc[region, mask] = adj

    elif bh_scope == "global":
        # Adjust all tests together
        flat = p_adj.values.ravel()
        mask = ~np.isnan(flat)
        if mask.sum() > 0:
            p_orig = flat[mask]
            _, adj, _, _ = multipletests(p_orig, method=bh_method)
            print(f"[global] p_orig: {len(p_orig)}, adj: {len(adj)}")
            flat_adj = flat.copy()
            flat_adj[mask] = adj
            p_adj.iloc[:, :] = flat_adj.reshape(p_adj.shape)

    else:
        raise ValueError("bh_scope must be 'within_region' or 'global'")

    # Return adjusted matrix (or you could return both raw+adj if you prefer)
    p_adj.columns = [c + out_col_suffix for c in p_adj.columns]
    return p_adj


def plot_pvalue_heatmap_regions_x_serotypes(
    pmat: pd.DataFrame,
    show_title: bool = False,
    title: str = "Serotype differences by region",
    figsize=(8, 10),
    show_text: bool = True,
    text_mode: str = "stars",     # "stars" | "p" | "both"
    p_format: str = ".2g",
    nan_text: str = "",
    xtick_fontsize: int = 13,
    ytick_fontsize: int = 14,
    title_fontsize: int = 18,
    text_fontsize: int = 11,
    use_log: bool = False,
    vmin: float = 0.0,
    vmax: float = 1.0,
    legend_orientation: str = "horizontal",  # "horizontal" | "vertical" | "none"
    legend_fraction: float = 0.05,          # thickness: horizontal height OR vertical width
    legend_pad: float = 0.015,              # distance from axes
    legend_label_fontsize: int = 14,
    legend_tick_fontsize: int = 12,
    tight_rect_bottom: float = 0.06,        # used when horizontal legend
):
    """
    Heatmap with rows = genomic regions, columns = serotypes.

    - Default: raw p-values (light blue colormap)
    - Optional: log scale using -log10(p)

    Star coding:
      *   p < 0.05
      **  p < 0.01
      *** p < 0.001
    """

    raw_p = pmat.values.astype(float)

    # ----- build values to plot -----
    if use_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            p_min = 1e-300
            safe_p = np.clip(raw_p, p_min, 1.0)
            data = -np.log10(safe_p)
            # data = -np.log10(raw_p)
        data[~np.isfinite(data)] = np.nan
        cmap = truncated_cmap("Blues", minval=0.2, maxval=0.75)
        cbar_label = r"$-\log_{10}(p)$"
        # sensible defaults if user did not override
        if vmin == 0.0 and vmax == 1.0:
            vmin, vmax = 0.0, np.nanmax(data)
    else:
        data = raw_p.copy()
        data[~np.isfinite(data)] = np.nan
        cmap = truncated_cmap("Blues", minval=0.2, maxval=0.75).reversed()
        cbar_label = "p-value"

    # ----- plotting -----
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # ticks / labels
    # xticklabels =  [c.replace("vs", " vs\n") for c in list(pmat.columns)]
    xticklabels = [c.replace("vDENV", "v") for c in list(pmat.columns)]
    ax.set_xticks(np.arange(pmat.shape[1]))
    ax.set_xticklabels(xticklabels, fontsize=xtick_fontsize)

    ax.xaxis.tick_top()  # move serotype labels to the top
    ax.tick_params(axis="x", bottom=False)

    ax.set_yticks(np.arange(pmat.shape[0]))
    ax.set_yticklabels(list(pmat.index), fontsize=ytick_fontsize)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)

    # gridlines
    ax.set_xticks(np.arange(-0.5, pmat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pmat.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # colorbar
    # cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    # cbar.set_label(cbar_label, rotation=90)
    # cbar = fig.colorbar(
    #     im,
    #     ax=ax,
    #     orientation="horizontal",  # 👈 key change
    #     fraction=0.06,  # height of the colorbar
    #     pad=0.01  # distance from the plot
    # )
    #
    # cbar.set_label(cbar_label, fontsize=14)
    # cbar.ax.tick_params(labelsize=12)

    # annotations (ALWAYS based on raw p-values)
    if show_text:
        for i in range(pmat.shape[0]):
            for j in range(pmat.shape[1]):
                p = raw_p[i, j]

                if not np.isfinite(p):
                    txt = nan_text
                else:
                    if p < 1e-3:
                        stars = "***"
                    elif p < 1e-2:
                        stars = "**"
                    elif p < 5e-2:
                        stars = "*"
                    else:
                        stars = ""

                    if text_mode == "stars":
                        txt = stars
                    elif text_mode == "p":
                        txt = format(p, p_format)
                    elif text_mode == "both":
                        txt = f"{stars}\n{format(p, p_format)}" if stars else format(p, p_format)
                    else:
                        raise ValueError('text_mode must be "stars", "p", or "both"')

                ax.text(
                    j, i, txt,
                    ha="center",
                    va="center",
                    fontsize=text_fontsize,
                    color="black"
                )

    # fig.tight_layout()

    # --- legend / colorbar ---
    if legend_orientation is None or legend_orientation.lower() == "none":
        cbar = None
        fig.tight_layout()
    elif legend_orientation.lower() == "horizontal":
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            fraction=legend_fraction,
            pad=legend_pad,
        )
        cbar.set_label(cbar_label, fontsize=legend_label_fontsize)
        cbar.ax.tick_params(labelsize=legend_tick_fontsize)

        # keep the plot tight while leaving a small strip for the bottom legend
        fig.tight_layout(rect=[0, tight_rect_bottom, 1, 1])

    elif legend_orientation.lower() == "vertical":
        # Passing ax=ax makes the colorbar span the full height of the heatmap axis by default
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            fraction=legend_fraction,
            pad=legend_pad,
        )
        cbar.set_label(cbar_label, fontsize=legend_label_fontsize, rotation=90)
        cbar.ax.tick_params(labelsize=legend_tick_fontsize)

        # No special rect needed here; vertical cbar doesn't eat bottom space
        fig.tight_layout()
    else:
        raise ValueError('legend_orientation must be "horizontal", "vertical", or "none"')

    return fig, ax


def plot_kruskal_pvalues_by_region_column(
    kw_df: pd.DataFrame,
    regions_order,
    p_col: str = "p",                 # "p" or "p_fdr"
    p_col_title: str = "Kruskal-Wallis",
    show_title: bool = False,
    title: str = "Global serotype differences by region (Kruskal–Wallis)",
    figsize=(2.2, 10),
    show_text: bool = True,
    text_mode: str = "stars",         # "stars" | "p" | "both"
    p_format: str = ".2g",
    nan_text: str = "",
    xtick_fontsize: int = 13,
    ytick_fontsize: int = 14,
    title_fontsize: int = 18,
    text_fontsize: int = 11,
    use_log: bool = False,            # if True, plot -log10(p)
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Single-column heatmap for Kruskal–Wallis results.

    - Uses SAME colormap + legend conventions as plot_pvalue_heatmap_regions_x_serotypes
      (i.e., truncated_cmap("Blues", 0.2, 0.75), reversed for raw p; not reversed for -log10(p)).
    - Does NOT add a colorbar/legend (so you can reuse the legend from your main heatmap).
    - Region ordering is forced by `regions_order` to match plot_pvalue_heatmap_regions_x_serotypes.

    Star coding (based on raw p-values from p_col):
      *   p < 0.05
      **  p < 0.01
      *** p < 0.001
    """
    if p_col not in kw_df.columns:
        raise ValueError(f"p_col='{p_col}' not found in kw_df columns: {list(kw_df.columns)}")

    # Enforce identical ordering to your main heatmap
    regions_order = list(regions_order)
    missing = [r for r in regions_order if r not in kw_df.index]
    if missing:
        raise ValueError(f"Some regions in regions_order are missing from kw_df.index: {missing}")

    raw_p = kw_df.loc[regions_order, p_col].astype(float).values.reshape(-1, 1)

    # ----- build values to plot (same logic as your main heatmap) -----
    if use_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            p_min = 1e-300
            safe_p = np.clip(raw_p, p_min, 1.0)
            data = -np.log10(safe_p)
        data[~np.isfinite(data)] = np.nan
        cmap = truncated_cmap("Blues", minval=0.2, maxval=0.75)
        # sensible defaults if user did not override
        if vmin == 0.0 and vmax == 1.0:
            vmin, vmax = 0.0, np.nanmax(data)
    else:
        data = raw_p.copy()
        data[~np.isfinite(data)] = np.nan
        cmap = truncated_cmap("Blues", minval=0.2, maxval=0.75).reversed()

    # ----- plotting -----
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # x tick: single column label (keep on top like the other plot)
    ax.set_xticks([0])
    ax.set_xticklabels([p_col_title], fontsize=xtick_fontsize)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", bottom=False)

    # y ticks: regions in the SAME order as the main heatmap
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order, fontsize=ytick_fontsize)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)

    # gridlines (same style)
    ax.set_xticks(np.arange(-0.5, 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(regions_order), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotations (ALWAYS based on raw p-values)
    if show_text:
        for i in range(len(regions_order)):
            p = raw_p[i, 0]

            if not np.isfinite(p):
                txt = nan_text
            else:
                if p < 1e-3:
                    stars = "***"
                elif p < 1e-2:
                    stars = "**"
                elif p < 5e-2:
                    stars = "*"
                else:
                    stars = ""

                if text_mode == "stars":
                    txt = stars
                elif text_mode == "p":
                    txt = format(p, p_format)
                elif text_mode == "both":
                    txt = f"{stars}\n{format(p, p_format)}" if stars else format(p, p_format)
                else:
                    raise ValueError('text_mode must be "stars", "p", or "both"')

            ax.text(
                0, i, txt,
                ha="center",
                va="center",
                fontsize=text_fontsize,
                color="black"
            )

    # IMPORTANT: no legend/colorbar here, by design (reuse the one from your main heatmap)
    fig.tight_layout()

    return fig, ax, im


def _format_effect_size_value(value, value_format=".2f", nan_text=""):
    if value is None or not np.isfinite(value):
        return nan_text
    return format(float(value), value_format)


def plot_effect_size_heatmap_regions_x_serotypes(
    matrix: pd.DataFrame,
    show_title: bool = False,
    title: str = "Pairwise effect sizes by region",
    figsize=(8, 10),
    show_text: bool = True,
    value_format: str = ".2f",
    nan_text: str = "",
    xtick_fontsize: int = 13,
    ytick_fontsize: int = 14,
    title_fontsize: int = 18,
    text_fontsize: int = 10,
    vmin: float = None,
    vmax: float = None,
    cmap_name: str = "Purples",
    cbar_label: str = "Effect size",
    legend_orientation: str = "vertical",  # "horizontal" | "vertical" | "none"
    legend_fraction: float = 0.05,
    legend_pad: float = 0.015,
    legend_label_fontsize: int = 14,
    legend_tick_fontsize: int = 12,
    tight_rect_bottom: float = 0.06,
):
    """
    Heatmap with rows = genomic regions, columns = serotype comparisons,
    values = pairwise effect sizes.

    This mirrors plot_pvalue_heatmap_regions_x_serotypes, but annotations and
    color scale are based directly on effect-size values rather than p-values.
    """
    data = matrix.values.astype(float)
    data[~np.isfinite(data)] = np.nan

    if vmin is None:
        vmin = np.nanmin(data) if np.isfinite(data).any() else 0.0
    if vmax is None:
        vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin == vmax:
        pad = abs(vmin) * 0.05 if vmin != 0 else 0.05
        vmin -= pad
        vmax += pad

    cmap = truncated_cmap(cmap_name, minval=0.2, maxval=0.75)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    xticklabels = [c.replace("vDENV", "v") for c in list(matrix.columns)]
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(xticklabels, fontsize=xtick_fontsize)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", bottom=False)

    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(list(matrix.index), fontsize=ytick_fontsize)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_text:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                txt = _format_effect_size_value(data[i, j], value_format=value_format, nan_text=nan_text)
                ax.text(j, i, txt, ha="center", va="center", fontsize=text_fontsize, color="black")

    if legend_orientation is None or legend_orientation.lower() == "none":
        cbar = None
        fig.tight_layout()
    elif legend_orientation.lower() == "horizontal":
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=legend_fraction, pad=legend_pad)
        cbar.set_label(cbar_label, fontsize=legend_label_fontsize)
        cbar.ax.tick_params(labelsize=legend_tick_fontsize)
        fig.tight_layout(rect=[0, tight_rect_bottom, 1, 1])
    elif legend_orientation.lower() == "vertical":
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=legend_fraction, pad=legend_pad)
        cbar.set_label(cbar_label, fontsize=legend_label_fontsize, rotation=90)
        cbar.ax.tick_params(labelsize=legend_tick_fontsize)
        fig.tight_layout()
    else:
        raise ValueError('legend_orientation must be "horizontal", "vertical", or "none"')

    return fig, ax


def plot_kruskal_effect_size_by_region_column(
    kw_df: pd.DataFrame,
    regions_order,
    effect_col: str = "epsilon_squared_R",
    effect_col_title: str = "Kruskal-Wallis",
    show_title: bool = False,
    title: str = "Global serotype differences by region effect size",
    figsize=(2.2, 10),
    show_text: bool = True,
    value_format: str = ".2f",
    nan_text: str = "",
    xtick_fontsize: int = 13,
    ytick_fontsize: int = 14,
    title_fontsize: int = 18,
    text_fontsize: int = 10,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "Purples",
):
    """
    Single-column heatmap for Kruskal-Wallis effect sizes.

    Mirrors plot_kruskal_pvalues_by_region_column, but values are effect sizes
    such as epsilon_squared_R or eta_squared_H.
    """
    if effect_col not in kw_df.columns:
        raise ValueError(f"effect_col='{effect_col}' not found in kw_df columns: {list(kw_df.columns)}")

    regions_order = list(regions_order)
    missing = [r for r in regions_order if r not in kw_df.index]
    if missing:
        raise ValueError(f"Some regions in regions_order are missing from kw_df.index: {missing}")

    data = kw_df.loc[regions_order, effect_col].astype(float).values.reshape(-1, 1)
    data[~np.isfinite(data)] = np.nan

    cmap = truncated_cmap(cmap_name, minval=0.2, maxval=0.75)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks([0])
    ax.set_xticklabels([effect_col_title], fontsize=xtick_fontsize)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", bottom=False)

    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order, fontsize=ytick_fontsize)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)

    ax.set_xticks(np.arange(-0.5, 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(regions_order), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_text:
        for i in range(len(regions_order)):
            txt = _format_effect_size_value(data[i, 0], value_format=value_format, nan_text=nan_text)
            ax.text(0, i, txt, ha="center", va="center", fontsize=text_fontsize, color="black")

    fig.tight_layout()
    return fig, ax, im
