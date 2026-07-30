import argparse

import paths
from dataset import config
from dataset.utils import (
    add_continent,
    build_leave_one_group_out_split,
    build_serotype_pivot,
    load_fasta_records,
    save_pivot_and_bar_plot,
    save_split_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate metadata tables and plots without overwriting existing split files.",
    )
    return parser.parse_args()


def print_warnings(df):
    missing_meta = df[~df["Has_META_COUNTRY"]]
    unknown_country = df[df["Country"] == "Unknown"]
    unknown_continent = df[df["Continent"] == "Unknown"]

    if len(missing_meta) > 0:
        print("\nWARNING: Some records do not contain META_COUNTRY.")
        print(f"Records without META_COUNTRY: {len(missing_meta)}")
        print(missing_meta[["File", "Header"]].head(20).to_string(index=False))

    if len(unknown_country) > 0:
        print("\nWARNING: Some records have Country='Unknown'.")
        print(f"Records with Country='Unknown': {len(unknown_country)}")
        print(unknown_country[["File", "Header", "Country"]].head(20).to_string(index=False))

    if len(unknown_continent) > 0:
        print("\nWARNING: Some records have Continent='Unknown'.")
        print(f"Records with Continent='Unknown': {len(unknown_continent)}")
        print("\nCountries not mapped to a continent:")
        print(unknown_continent["Country"].value_counts())


def main():
    args = parse_args()
    df = load_fasta_records(paths.genomes_dir, include_country=True, with_index=True)
    print(f"\nLoaded records: {len(df)}")

    df = add_continent(df)
    print_warnings(df)

    records_path = paths.stats_dir / "dengue_geographical_distribution_records.csv"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(records_path, sep=";", index=False)
    print(f"\nSaved records table to: {records_path}")

    country_pivot = build_serotype_pivot(df, "Country")
    print("\nCountry distribution:")
    print(country_pivot)
    save_pivot_and_bar_plot(
        country_pivot,
        paths.stats_dir / "dengue_geographical_distribution_by_country.csv",
        paths.stats_dir / "dengue_geographical_distribution_by_country.png",
        "Geographical distribution of Dengue virus genomes by serotype",
        "Country",
        figsize=(12, 6),
        xtick_rotation=90,
        no_legend_plot_path=paths.stats_dir / "dengue_geographical_distribution_by_country_no_legend.png",
        comparable_bar_width=False,
    )

    continent_pivot = build_serotype_pivot(df, "Continent")
    print("\nContinent distribution:")
    print(continent_pivot)
    save_pivot_and_bar_plot(
        continent_pivot,
        paths.stats_dir / "dengue_geographical_distribution_by_continent.csv",
        paths.stats_dir / "dengue_geographical_distribution_by_continent.png",
        "Geographical distribution of Dengue virus genomes by serotype",
        "Continent",
        figsize=(7, 6),
        xtick_rotation=45,
        no_legend_plot_path=paths.stats_dir / "dengue_geographical_distribution_by_continent_no_legend.png",
    )

    if args.plots_only:
        print("\n--plots-only: existing geographical split files were not modified.")
    elif config.GENERATE_SPLITS:
        split_df = build_leave_one_group_out_split(
            df=df,
            group_col="Continent",
            ordered_groups=continent_pivot.index.tolist(),
        )
        save_split_files(split_df, paths.splits_dir / "dengue_leave_one_continent_out_splits.csv")


if __name__ == "__main__":
    main()
