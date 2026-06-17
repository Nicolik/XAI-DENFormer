import csv
import re
from pathlib import Path

from Bio import SeqIO
import country_converter as coco
import pandas as pd

from cdhit.config import desired_serotypes, fasta_files_by_serotype
from cdhit.plotting import continent_order, year_bin_order


COUNTRY_ALIASES = {
    "DRC": "Democratic Republic of the Congo",
}


def normalize_value(value) -> str:
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().strip("{}").strip()

    if value == "" or value.lower() in ["unknown", "nan", "un", "not found", "none"]:
        return "Unknown"

    return value


def normalize_country(country) -> str:
    country = normalize_value(country)
    return COUNTRY_ALIASES.get(country, country)


def extract_country_from_meta(header: str) -> str:
    match = re.search(r"META_COUNTRY=\{([^}]+)\}", header)
    if match:
        return normalize_country(match.group(1))

    match = re.search(r"META_COUNTRY=([^|;]+?)(?=\s+META_|$|[|;])", header)
    if match:
        return normalize_country(match.group(1))

    return "Unknown"


def extract_year_from_meta(header: str) -> int | None:
    match = re.search(r"META_YEAR=\{(\d{4})\}", header)
    if match:
        return int(match.group(1))

    match = re.search(r"META_YEAR=(\d{4})", header)
    if match:
        return int(match.group(1))

    return None


def extract_serotype(header: str, fallback: str) -> str:
    match = re.search(r"DENV(\d)", header, flags=re.IGNORECASE)
    if match:
        return f"DENV{match.group(1)}"

    match = re.search(r"hDenV(\d)", header, flags=re.IGNORECASE)
    if match:
        return f"DENV{match.group(1)}"

    return fallback


def assign_year_bin(year) -> str:
    if year is None or pd.isna(year):
        return "Unknown"

    year = int(year)

    if year < 2000:
        return "<2000"
    if 2000 <= year <= 2009:
        return "2000-2009"
    if 2010 <= year <= 2019:
        return "2010-2019"

    return "2020+"


def pretty_fold_name(fold_name) -> str:
    match = re.search(r"fold[_\s-]*(\d+)", str(fold_name), flags=re.IGNORECASE)
    if match:
        return f"Fold {match.group(1)}"
    return str(fold_name)


def fold_sort_key(fold_name) -> int:
    match = re.search(r"(\d+)", str(fold_name))
    if match:
        return int(match.group(1))
    return 9999


def build_metadata_from_fasta(input_dir: Path) -> dict:
    records = []

    for filename, fallback_serotype in fasta_files_by_serotype.items():
        fasta_path = input_dir / filename

        if not fasta_path.exists():
            print(f"WARNING: FASTA not found: {fasta_path}")
            continue

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            header = record.description
            seq_id = record.id

            country = extract_country_from_meta(header)
            year = extract_year_from_meta(header)
            serotype = extract_serotype(header, fallback_serotype)

            records.append({
                "sequence_id": seq_id,
                "header": header,
                "serotype_from_header": serotype,
                "country": country,
                "year": year,
                "year_bin": assign_year_bin(year),
            })

    if not records:
        raise ValueError("No FASTA metadata loaded. Check input directory and FASTA filenames.")

    df = pd.DataFrame(records)
    countries = df["country"].fillna("Unknown").astype(str).str.strip().tolist()

    continents = coco.convert(
        names=countries,
        to="continent_7",
        not_found="Unknown",
    )

    df["continent"] = [normalize_value(x) for x in continents]
    df.loc[df["country"] == "Unknown", "continent"] = "Unknown"

    metadata = {}

    for _, row in df.iterrows():
        item = row.to_dict()
        metadata[row["sequence_id"]] = item
        metadata[row["header"]] = item

    return metadata


def get_metadata(seq_id: str, metadata: dict) -> dict | None:
    if seq_id in metadata:
        return metadata[seq_id]

    first_token = str(seq_id).split()[0]
    if first_token in metadata:
        return metadata[first_token]

    for key, value in metadata.items():
        if isinstance(key, str) and key.startswith(seq_id):
            return value

    return None


def load_fold_membership(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fold membership file not found: {path}")

    rows = []

    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        required = {"fold", "serotype", "cluster_id", "sequence_id"}
        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(f"Missing columns in fold membership file: {missing}")

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("Fold membership file is empty.")

    return rows


def annotate_fold_rows(fold_rows: list[dict], metadata: dict) -> tuple[pd.DataFrame, list[str]]:
    annotated = []
    missing = []

    for row in fold_rows:
        seq_id = row["sequence_id"]
        meta = get_metadata(seq_id, metadata)
        pretty_fold = pretty_fold_name(row["fold"])

        if meta is None:
            missing.append(seq_id)

            annotated.append({
                "fold": pretty_fold,
                "serotype": row["serotype"],
                "cluster_id": row["cluster_id"],
                "sequence_id": seq_id,
                "country": "Unknown",
                "continent": "Unknown",
                "year": "Unknown",
                "year_bin": "Unknown",
                "metadata_found": 0,
            })
            continue

        annotated.append({
            "fold": pretty_fold,
            "serotype": row["serotype"],
            "cluster_id": row["cluster_id"],
            "sequence_id": seq_id,
            "country": meta["country"],
            "continent": meta["continent"],
            "year": meta["year"] if not pd.isna(meta["year"]) else "Unknown",
            "year_bin": meta["year_bin"],
            "metadata_found": 1,
        })

    return pd.DataFrame(annotated), missing


def warn_unknowns(df: pd.DataFrame) -> None:
    checks = {
        "country": df["country"].eq("Unknown"),
        "continent": df["continent"].eq("Unknown"),
        "year": df["year"].eq("Unknown"),
        "year_bin": df["year_bin"].eq("Unknown"),
        "metadata_found": df["metadata_found"].eq(0),
    }

    for name, mask in checks.items():
        n = int(mask.sum())
        if n > 0:
            print(f"\nWARNING: found {n} sequences with Unknown/missing {name}.")

            cols = ["fold", "serotype", "cluster_id", "sequence_id", "country", "continent", "year", "year_bin"]
            print(df.loc[mask, cols].head(20).to_string(index=False))


def save_fold_summary(df: pd.DataFrame, out_path: Path) -> None:
    folds = sorted(df["fold"].unique(), key=fold_sort_key)
    rows = []

    for fold in folds:
        sub = df[df["fold"] == fold]

        row = {
            "fold": fold,
            "total_sequences": len(sub),
            "metadata_found": int(sub["metadata_found"].sum()),
            "metadata_missing": int((sub["metadata_found"] == 0).sum()),
        }

        for serotype in desired_serotypes:
            row[f"{serotype}_sequences"] = int((sub["serotype"] == serotype).sum())

        for continent in continent_order:
            row[f"{continent}_sequences"] = int((sub["continent"] == continent).sum())

        for year_bin in year_bin_order:
            row[f"{year_bin}_sequences"] = int((sub["year_bin"] == year_bin).sum())

        unknown_continent = int((sub["continent"] == "Unknown").sum())
        unknown_year = int((sub["year_bin"] == "Unknown").sum())

        if unknown_continent:
            row["Unknown_continent_sequences"] = unknown_continent
        if unknown_year:
            row["Unknown_year_sequences"] = unknown_year

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, sep=";", index=False)
    print(f"Saved: {out_path}")


def save_missing_ids(missing_ids: list[str], out_path: Path) -> None:
    with open(out_path, "w") as handle:
        for seq_id in sorted(set(missing_ids)):
            handle.write(seq_id + "\n")

    print(f"Saved: {out_path}")
