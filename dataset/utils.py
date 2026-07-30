import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import country_converter as coco

from dataset import config


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_value(value):
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip().strip("{}").strip()
    if value == "" or value.lower() in {"unknown", "nan", "un", "not found", "none"}:
        return "Unknown"
    return value


def normalize_country(country):
    country = normalize_value(country)
    return config.COUNTRY_ALIASES.get(country, country)


def extract_serotype(header, fallback=None):
    """Extract a serotype-like token from an unstructured FASTA header.

    Header-derived values are diagnostic metadata only. The canonical class label
    is always the serotype-specific FASTA assignment supplied by the caller.
    """
    for pattern in (r"DENV(\d)", r"hDenV(\d)"):
        match = re.search(pattern, header, flags=re.IGNORECASE)
        if match:
            return f"DENV{match.group(1)}"
    return fallback


def extract_country_from_meta(header):
    match = re.search(r"META_COUNTRY=\{([^}]+)\}", header)
    if match:
        return normalize_country(match.group(1)), True
    match = re.search(r"META_COUNTRY=([^|;]+?)(?=\s+META_|$|[|;])", header)
    if match:
        return normalize_country(match.group(1)), True
    return "Unknown", False


def extract_year_from_meta(header):
    match = re.search(r"META_YEAR=\{(\d{4})\}", header)
    if match:
        return int(match.group(1)), True
    match = re.search(r"META_YEAR=(\d{4})", header)
    if match:
        return int(match.group(1)), True
    return None, False


def parse_fasta_serotype(filename, serotype_from_filename, start_index=None, include_country=False, include_year=False):
    records = []
    current_index = start_index

    for record in SeqIO.parse(str(filename), "fasta"):
        header = record.description
        header_serotype = extract_serotype(header)
        row = {
            "File": os.path.basename(str(filename)),
            "Header": header,
            # Canonical label: identical to the source used by 01_run_ohe.py.
            "Serotype": serotype_from_filename,
            # Diagnostic fields: never used as class labels.
            "HeaderSerotype": header_serotype if header_serotype is not None else "Unknown",
            "SerotypeHeaderConflict": bool(
                header_serotype is not None and header_serotype != serotype_from_filename
            ),
        }

        if current_index is not None:
            row["Index"] = current_index
            current_index += 1

        if include_country:
            country, has_meta_country = extract_country_from_meta(header)
            row.update({"Country": country, "Has_META_COUNTRY": has_meta_country})

        if include_year:
            year, has_meta_year = extract_year_from_meta(header)
            row.update({"Year": year, "Has_META_YEAR": has_meta_year})

        records.append(row)

    return records, current_index


def load_fasta_records(genomes_dir, include_country=False, include_year=False, with_index=False):
    all_records = []
    next_index = 0 if with_index else None

    for fname, serotype in config.FASTA_FILES.items():
        fpath = Path(genomes_dir) / fname
        if not fpath.exists():
            print(f"WARNING: file not found: {fpath}")
            continue
        records, next_index = parse_fasta_serotype(
            fpath,
            serotype,
            start_index=next_index,
            include_country=include_country,
            include_year=include_year,
        )
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No records loaded. Check genomes_dir and FASTA filenames.")
    return df


def add_continent(df):
    if coco is None:
        raise ImportError("country_converter is required for continent plotting.")

    countries = df["Country"].fillna("Unknown").astype(str).str.strip().tolist()
    continents = coco.convert(names=countries, to="continent_7", not_found="Unknown")
    df = df.copy()
    df["Continent"] = [normalize_value(x) for x in continents]
    df.loc[df["Country"] == "Unknown", "Continent"] = "Unknown"
    return df


def build_serotype_pivot(df, index_col, row_order=None):
    pivot = df.pivot_table(
        index=index_col,
        columns="Serotype",
        values="Header",
        aggfunc="count",
        fill_value=0,
    )
    if row_order is not None:
        pivot = pivot.reindex(row_order)
    pivot = pivot.reindex(columns=config.SEROTYPES, fill_value=0)

    if row_order is None and "Unknown" in pivot.index:
        known = pivot.drop(index="Unknown")
        pivot = pd.concat([known, pivot.loc[["Unknown"]]])
    return pivot


def _stats_bar_figsize(n_bars, requested_height=None):
    height = requested_height if requested_height is not None else config.STATS_BAR_HEIGHT
    width = max(
        config.STATS_BAR_MIN_FIGWIDTH,
        config.STATS_BAR_MARGIN_INCHES + n_bars * config.STATS_BAR_SLOT_INCHES,
    )
    return width, height


def save_pivot_and_bar_plot(
    pivot,
    csv_path,
    plot_path,
    title,
    xlabel,
    figsize=(10, 6),
    xtick_rotation=45,
    no_legend_plot_path=None,
    comparable_bar_width=True,
):
    ensure_dir(Path(csv_path).parent)
    ensure_dir(Path(plot_path).parent)
    if no_legend_plot_path is not None:
        ensure_dir(Path(no_legend_plot_path).parent)

    pivot.to_csv(csv_path, sep=";")
    print(f"Saved pivot table to: {csv_path}")

    requested_height = figsize[1] if figsize is not None else None
    if comparable_bar_width:
        plot_figsize = _stats_bar_figsize(len(pivot.index), requested_height=requested_height)
    else:
        plot_figsize = figsize if figsize is not None else (10, config.STATS_BAR_HEIGHT)

    def draw(output_path, show_legend):
        ax = pivot.plot(
            kind="bar",
            stacked=True,
            figsize=plot_figsize,
            color=config.SEROTYPE_COLORS,
            width=config.STATS_BAR_WIDTH,
        )
        ax.set_ylabel("Genomes", fontsize=config.STATS_AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=config.STATS_TICK_FONTSIZE)

        if config.SHOW_TITLES:
            ax.set_xlabel(xlabel, fontsize=config.STATS_AXIS_LABEL_FONTSIZE)
            ax.set_title(title, fontsize=config.STATS_TITLE_FONTSIZE)
        else:
            ax.set_xlabel("")
            ax.set_title("")

        if show_legend:
            legend = ax.legend(title="Serotype" if config.SHOW_TITLES else None, fontsize=config.STATS_LEGEND_FONTSIZE)
            if legend is not None and not config.SHOW_TITLES:
                legend.set_title(None)
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        plt.xticks(rotation=xtick_rotation)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to: {output_path}")

    draw(plot_path, show_legend=True)
    if no_legend_plot_path is not None:
        draw(no_legend_plot_path, show_legend=False)


def choose_validation_indices(train_pool, val_size=config.VAL_SIZE, random_state=config.RANDOM_STATE):
    if val_size <= 0 or len(train_pool) <= 1:
        return np.array([], dtype=np.int64)

    val_count = int(round(len(train_pool) * val_size)) if val_size < 1 else int(val_size)
    val_count = max(1, min(val_count, len(train_pool) - 1))

    y = train_pool["Serotype"]
    counts = y.value_counts()
    can_stratify = (
        len(counts) > 1
        and counts.min() >= 2
        and val_count >= len(counts)
        and (len(train_pool) - val_count) >= len(counts)
    )

    _, val_df = train_test_split(
        train_pool,
        test_size=val_count,
        random_state=random_state,
        shuffle=True,
        stratify=y if can_stratify else None,
    )
    return val_df["Index"].to_numpy(dtype=np.int64)


def build_leave_one_group_out_split(df, group_col, ordered_groups):
    rows = []
    for group in ordered_groups:
        if group == "Unknown" and not config.INCLUDE_UNKNOWN_SPLIT:
            continue

        test_mask = df[group_col].astype(str) == str(group)
        test_df = df[test_mask]
        train_pool = df[~test_mask]

        if test_df.empty or train_pool.empty:
            print(f"WARNING: skipping split for {group!r}: empty test or train pool")
            continue

        test_indices = set(test_df["Index"].astype(int).tolist())
        val_indices = set(choose_validation_indices(train_pool).tolist())

        for _, row in df.iterrows():
            index = int(row["Index"])
            if index in test_indices:
                split = "test"
            elif index in val_indices:
                split = "val"
            else:
                split = "train"

            rows.append({
                "fold": group,
                "index": index,
                "split": split,
                "group": group,
                "serotype": row["Serotype"],
            })

    return pd.DataFrame(rows, columns=["fold", "index", "split", "group", "serotype"])


def save_split_files(split_df, output_path):
    ensure_dir(Path(output_path).parent)
    split_df.to_csv(output_path, sep=config.SPLIT_SEP, index=False)
    print(f"Saved split file to: {output_path}")

    summary = (
        split_df.groupby(["fold", "split", "serotype"])
        .size()
        .reset_index(name="n")
        .sort_values(["fold", "split", "serotype"])
    )
    summary_path = str(output_path).replace(".csv", ".summary.csv")
    summary.to_csv(summary_path, sep=";", index=False)
    print(f"Saved split summary to: {summary_path}")


def fasta_to_onehot_embeddings(fasta_file):
    base_to_vec = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    seq_vectors = []
    seq_ids = []
    for record in SeqIO.parse(str(fasta_file), "fasta"):
        seq_vectors.append(np.array([base_to_vec.get(base, [0, 0, 0, 0]) for base in str(record.seq).upper()]))
        seq_ids.append(record.id)
    return seq_vectors, seq_ids
