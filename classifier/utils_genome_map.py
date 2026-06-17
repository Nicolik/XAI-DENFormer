import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from classifier.config import LONGEST_SEQUENCE_LENGTH


def get_gene_boundaries(map_df, gene_name='Proteina', genome_end=11195):
    """Return region boundaries from a coordinate CSV.

    Supports both the new map files, which include a Region column and UTR rows,
    and the older genome_map.py files, which only had Proteina/Start_nt/End_nt.
    """
    preferred_order = [
        "5'UTR", "C", "prM", "E", "NS1", "NS2A", "NS2B",
        "NS3", "NS4A", "2K", "NS4B", "NS5", "3'UTR",
    ]
    product_to_region = {
        "anchored capsid protein ancC": "C",
        "capsid protein C": "C",
        "membrane glycoprotein precursor prM": "prM",
        "premembrane protein prM": "prM",
        "envelope protein E": "E",
        "nonstructural protein NS1": "NS1",
        "nonstructural protein NS2A": "NS2A",
        "nonstructural protein NS2B": "NS2B",
        "nonstructural protein NS3": "NS3",
        "nonstructural protein NS4A": "NS4A",
        "protein 2K": "2K",
        "nonstructural protein NS4B": "NS4B",
        "RNA-dependent RNA polymerase NS5": "NS5",
    }

    boundaries = {}
    for _, row in map_df.iterrows():
        raw_name = str(row.get("Region") or row.get(gene_name) or row.get("Proteina"))
        region = raw_name if raw_name in preferred_order else product_to_region.get(raw_name)
        if region is None:
            continue
        start = int(row["Start_nt"])
        end = int(row["End_nt"])
        if region == "5'UTR":
            start = 0
        boundaries[region] = (start, end)

    if "5'UTR" not in boundaries:
        boundaries["5'UTR"] = (0, 78)
    if "3'UTR" not in boundaries:
        ns5_end = boundaries.get("NS5", (10255, 10255))[1]
        boundaries["3'UTR"] = (ns5_end + 1, genome_end)

    missing = [region for region in preferred_order if region not in boundaries]
    if missing:
        raise ValueError(f"Missing regions in coordinate map: {missing}")

    return {region: boundaries[region] for region in preferred_order}

def _moving_average_ignore_nan(profile, window):
    """Centered moving average that does not let NaNs/invalid padding positions contribute."""
    arr = np.asarray(profile, dtype=float)
    if window is None or int(window) <= 1:
        return arr.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1

    valid = np.isfinite(arr)
    values = np.where(valid, arr, 0.0)
    kernel = np.ones(window, dtype=float)
    num = np.convolve(values, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.divide(num, den, out=np.full_like(arr, np.nan, dtype=float), where=den > 0)
    out[~valid & (den == 0)] = np.nan
    return out


def smooth_profiles(class_profiles, window=51):
    """Return smoothed copies of class profiles using a centered moving average."""
    return {name: _moving_average_ignore_nan(profile, window) for name, profile in class_profiles.items()}


def _minmax_normalize_ignore_nan(profile):
    """Min-max normalize a profile to 0-1 while preserving NaN padded positions."""
    arr = np.asarray(profile, dtype=float)
    out = arr.copy()
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    vmin = float(np.nanmin(arr[valid]))
    vmax = float(np.nanmax(arr[valid]))
    if vmax > vmin:
        out[valid] = (arr[valid] - vmin) / (vmax - vmin)
    else:
        out[valid] = 0.0
    return out


def normalize_profiles_ignore_nan(class_profiles):
    """Return per-profile min-max normalized copies, preserving NaN padded positions."""
    return {
        name: _minmax_normalize_ignore_nan(profile)
        for name, profile in class_profiles.items()
    }


def plot_attention_stacked_with_genes(class_profiles, gene_boundaries, output_dir, prefix="class",
                                      cmap_name="tab20", figsize=(12, 3), dpi=300, xmax=LONGEST_SEQUENCE_LENGTH,
                                      region_name="Gene", ylabel="Attention (%)",
                                      percent_yaxis=True, smooth_window=None, filename_suffix="",
                                      renormalize_after_smoothing=False):
    if not class_profiles:
        print("[WARN] No class profiles to plot.")
        return None

    if smooth_window is not None and int(smooth_window) > 1:
        class_profiles = smooth_profiles(class_profiles, window=int(smooth_window))
        if renormalize_after_smoothing:
            class_profiles = normalize_profiles_ignore_nan(class_profiles)

    classes_sorted = sorted(class_profiles.keys())
    num_classes = len(classes_sorted)

    fig, axes = plt.subplots(
        num_classes, 1,
        figsize=(figsize[0], figsize[1] * num_classes),
        sharex=True
    )
    if num_classes == 1:
        axes = [axes]

    colors = cm.get_cmap(cmap_name, len(gene_boundaries))
    gene_handles, gene_labels = [], []

    for ax, (class_name, profile) in zip(axes, sorted(class_profiles.items())):
        profile = np.asarray(profile, dtype=float)
        x = np.arange(len(profile))
        ax.plot(x, profile, lw=2, color="black")

        for i, (gene, (start, end)) in enumerate(gene_boundaries.items()):
            span = ax.axvspan(start, end, color=colors(i), alpha=0.3, label=gene)
            if gene not in gene_labels:
                gene_handles.append(span)
                gene_labels.append(gene)

        ax.set_title(f"{class_name}", fontsize=20, pad=5)
        ax.set_ylabel(ylabel, fontsize=18, labelpad=6)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, xmax)
        ax.set_yticks(np.linspace(0, 1, 5))
        if percent_yaxis:
            ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 5)], fontsize=16)
        else:
            ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0, 1, 5)], fontsize=16)
        ax.tick_params(axis="x", labelsize=16)

    axes[-1].set_xlabel("Genomic RNA (Position)", fontsize=18, labelpad=6)

    fig.legend(
        gene_handles, gene_labels,
        loc="center left", bbox_to_anchor=(0.915, 0.5),
        ncol=1, fontsize=16,
        title=region_name, title_fontsize=18
    )

    # fig.suptitle(title, fontsize=22, y=0.99)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    out_path = os.path.join(output_dir, f"{prefix}_stacked_with_genes{filename_suffix}.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved stacked subplot figure with improved layout -> {out_path}")
    return out_path
