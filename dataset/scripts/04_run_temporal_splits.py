import pandas as pd
import paths

from dataset import config
from dataset.utils import (
    build_leave_one_group_out_split,
    build_serotype_pivot,
    load_fasta_records,
    save_pivot_and_bar_plot,
    save_split_files,
)


def assign_time_bin(year, max_observed_year):
    if pd.isna(year):
        return "Unknown"

    year = int(year)
    if year < config.CUTOFF_YEAR:
        return f"<{config.CUTOFF_YEAR}"

    start = config.CUTOFF_YEAR + ((year - config.CUTOFF_YEAR) // config.BIN_LENGTH) * config.BIN_LENGTH
    end = min(start + config.BIN_LENGTH - 1, max_observed_year)
    return f"{start}-{end}"


def time_bin_sort_key(bin_name):
    if bin_name == f"<{config.CUTOFF_YEAR}":
        return 0
    if bin_name == "Unknown":
        return 9999
    return int(str(bin_name).split("-")[0])


def print_warnings(df):
    missing_year = df[~df["Has_META_YEAR"]]
    if len(missing_year) > 0:
        print("\nWARNING: Some records do not contain META_YEAR.")
        print(f"Records without META_YEAR: {len(missing_year)}")
        print("These entries will be included in the plot as 'Unknown'.")
        print(missing_year[["File", "Header"]].head(20).to_string(index=False))


def main():
    df = load_fasta_records(paths.genomes_dir, include_year=True, with_index=True)
    print(f"\nLoaded records: {len(df)}")
    print_warnings(df)

    valid_years = df["Year"].dropna()
    max_observed_year = config.CUTOFF_YEAR if valid_years.empty else int(valid_years.max())
    if valid_years.empty:
        print("\nWARNING: No valid META_YEAR values found.")

    df["TimeBin"] = df["Year"].apply(lambda y: assign_time_bin(y, max_observed_year))

    suffix = f"c{config.CUTOFF_YEAR}_b{config.BIN_LENGTH}"
    records_path = paths.stats_dir / f"dengue_temporal_metadata_{suffix}.csv"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(records_path, sep=";", index=False)
    print(f"\nSaved metadata table to: {records_path}")

    bins_order = sorted(df["TimeBin"].unique(), key=time_bin_sort_key)
    pivot = build_serotype_pivot(df, "TimeBin", row_order=bins_order)

    print("\nTemporal distribution:")
    print(pivot)

    save_pivot_and_bar_plot(
        pivot,
        paths.stats_dir / f"dengue_temporal_distribution_{suffix}.csv",
        paths.stats_dir / f"dengue_temporal_distribution_{suffix}.png",
        "Temporal distribution of Dengue virus genomes by serotype",
        "Time interval",
        figsize=(10, 6),
        xtick_rotation=45,
        no_legend_plot_path=paths.stats_dir / f"dengue_temporal_distribution_{suffix}_no_legend.png",
    )

    if config.GENERATE_SPLITS:
        split_df = build_leave_one_group_out_split(
            df=df,
            group_col="TimeBin",
            ordered_groups=bins_order,
        )
        save_split_files(split_df, paths.splits_dir / f"dengue_leave_one_timebin_out_{suffix}_splits.csv")

    print("\nCounts by year:")
    print(df["Year"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
