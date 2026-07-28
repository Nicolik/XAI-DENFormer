#!/usr/bin/env python3
"""Prepare the merged NCBI Virus + GISAID DENV dataset.

This script consolidates the historical source-preparation workflow into one
portable entry point. For each DENV serotype it:

1. reads one NCBI Virus FASTA and one GISAID FASTA;
2. normalizes sequence letters to uppercase;
3. removes exact whole-sequence duplicates within each source, retaining the
   first header encountered;
4. merges both sources and gives GISAID precedence for exact cross-source
   duplicates;
5. extracts source-specific year/country metadata;
6. discards records without a reliable year or country;
7. writes ``DENV*_merged_meta.fasta`` and reproducibility reports.

No similarity threshold is used during deduplication. IUPAC ambiguity symbols,
``N`` characters, and gap characters are not otherwise normalized: two records
are collapsed only when their complete uppercase sequence strings are exactly
identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from Bio import SeqIO

SEROTYPES = ("DENV1", "DENV2", "DENV3", "DENV4")

# Preserved from the historical NCBI metadata parser. NCBI countries are
# accepted only when an explicit country name occurs in the original header.
COUNTRY_KEYWORDS = (
    "dominican republic",
    "united states",
    "french guiana",
    "new caledonia",
    "french polynesia",
    "saudi arabia",
    "cote d'ivoire",
    "puerto rico",
    "sri lanka",
    "bangladesh",
    "bolivia",
    "india",
    "china",
    "brazil",
    "thailand",
    "vietnam",
    "indonesia",
    "mexico",
    "colombia",
    "peru",
    "ghana",
    "angola",
    "japan",
    "taiwan",
    "nepal",
    "malaysia",
    "singapore",
    "usa",
    "france",
    "germany",
    "italy",
    "spain",
    "guadeloupe",
    "martinique",
    "djibouti",
    "eritrea",
    "somalia",
    "niger",
    "seychelles",
    "uruguay",
    "venezuela",
    "argentina",
    "panama",
    "cambodia",
)

YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
GISAID_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(frozen=True)
class SourceRecord:
    """One retained source record after within-source exact deduplication."""

    sequence: str
    record_id: str
    full_header: str
    source: str


@dataclass
class SourceLoadResult:
    records_by_sequence: dict[str, SourceRecord]
    input_records: int

    @property
    def unique_sequences(self) -> int:
        return len(self.records_by_sequence)

    @property
    def within_source_duplicates(self) -> int:
        return self.input_records - self.unique_sequences


def load_unique_records(path: Path, source: str) -> SourceLoadResult:
    """Read a FASTA and keep the first record for each uppercase sequence."""

    records_by_sequence: dict[str, SourceRecord] = {}
    input_records = 0

    for record in SeqIO.parse(str(path), "fasta"):
        input_records += 1
        sequence = str(record.seq).upper()
        if sequence in records_by_sequence:
            continue
        records_by_sequence[sequence] = SourceRecord(
            sequence=sequence,
            record_id=record.id,
            full_header=record.description,
            source=source,
        )

    return SourceLoadResult(
        records_by_sequence=records_by_sequence,
        input_records=input_records,
    )


def extract_gisaid_year_country(header: str) -> tuple[str | None, str | None]:
    """Extract GISAID country and the last explicit four-digit year."""

    first_field = header.split("|", maxsplit=1)[0]
    parts = first_field.split("/")

    country: str | None = None
    if len(parts) >= 2:
        candidate = parts[1].strip()
        if candidate.lower() not in {"", "un", "unknown", "na", "n/a"}:
            country = candidate.replace("_", " ")

    year_matches = list(GISAID_YEAR_RE.finditer(header))
    year = year_matches[-1].group(0) if year_matches else None
    return year, country


def extract_ncbi_year(header: str) -> str | None:
    """Return the first explicit four-digit year from an NCBI header."""

    match = YEAR_RE.search(header)
    return match.group(0) if match else None


def extract_ncbi_country(header: str) -> str | None:
    """Return an explicitly named country from an NCBI header."""

    normalized = header.lower().replace("_", " ")
    for country in COUNTRY_KEYWORDS:
        if re.search(rf"\b{re.escape(country)}\b", normalized):
            return country.title()
    return None


def format_fasta(sequence: str, width: int = 80) -> Iterator[str]:
    for start in range(0, len(sequence), width):
        yield sequence[start : start + width]


def resolve_input_path(directory: Path, template: str, serotype: str) -> Path:
    path = directory / template.format(serotype=serotype)
    if not path.is_file():
        raise FileNotFoundError(f"Missing input FASTA: {path}")
    return path


def write_merged_fasta(
    path: Path,
    ncbi: SourceLoadResult,
    gisaid: SourceLoadResult,
) -> None:
    """Write the historical intermediate exact-deduplicated merged FASTA."""

    merged: dict[str, SourceRecord] = dict(ncbi.records_by_sequence)
    merged.update(gisaid.records_by_sequence)  # GISAID precedence.

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in merged.values():
            # Historical merging wrote record.id rather than the full NCBI
            # description. The full NCBI header is restored during metadata
            # annotation below.
            handle.write(f">{record.record_id}\n")
            for line in format_fasta(record.sequence):
                handle.write(line + "\n")


def prepare_serotype(
    *,
    serotype: str,
    ncbi_path: Path,
    gisaid_path: Path,
    output_path: Path,
    merged_path: Path | None,
) -> tuple[dict[str, int | str], list[dict[str, str]]]:
    ncbi = load_unique_records(ncbi_path, "NCBI")
    gisaid = load_unique_records(gisaid_path, "GISAID")

    cross_source_duplicates = len(
        set(ncbi.records_by_sequence).intersection(gisaid.records_by_sequence)
    )

    if merged_path is not None:
        write_merged_fasta(merged_path, ncbi, gisaid)

    # Merge in the same order and with the same source precedence as the
    # historical workflow: NCBI first, then GISAID overwrite on exact matches.
    merged: dict[str, SourceRecord] = dict(ncbi.records_by_sequence)
    merged.update(gisaid.records_by_sequence)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    discarded: list[dict[str, str]] = []
    final_retained = 0
    retained_ncbi = 0
    retained_gisaid = 0
    missing_year = 0
    missing_country = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for record in merged.values():
            if record.source == "GISAID":
                # Preserve historical behavior: the final GISAID base header is
                # the FASTA record ID used during merging.
                final_header = record.record_id
                year, country = extract_gisaid_year_country(final_header)
            else:
                # Preserve historical behavior: restore the complete original
                # NCBI description before metadata parsing and output.
                final_header = record.full_header
                year = extract_ncbi_year(final_header)
                country = extract_ncbi_country(final_header)

            if year is None:
                missing_year += 1
            if country is None:
                missing_country += 1

            if year is None or country is None:
                discarded.append(
                    {
                        "serotype": serotype,
                        "source": record.source,
                        "year": year or "",
                        "country": country or "",
                        "header": final_header,
                        "reason": "missing_year_and_country"
                        if year is None and country is None
                        else "missing_year"
                        if year is None
                        else "missing_country",
                    }
                )
                continue

            annotated_header = (
                f"{final_header}"
                f"|META_YEAR={year}"
                f"|META_COUNTRY={country}"
                f"|META_SOURCE={record.source}"
            )
            out.write(f">{annotated_header}\n")
            for line in format_fasta(record.sequence):
                out.write(line + "\n")

            final_retained += 1
            if record.source == "GISAID":
                retained_gisaid += 1
            else:
                retained_ncbi += 1

    summary: dict[str, int | str] = {
        "serotype": serotype,
        "ncbi_input_records": ncbi.input_records,
        "gisaid_input_records": gisaid.input_records,
        "ncbi_unique_sequences": ncbi.unique_sequences,
        "gisaid_unique_sequences": gisaid.unique_sequences,
        "ncbi_within_source_duplicates": ncbi.within_source_duplicates,
        "gisaid_within_source_duplicates": gisaid.within_source_duplicates,
        "cross_source_exact_duplicates": cross_source_duplicates,
        "merged_unique_sequences": len(merged),
        "discarded_missing_year": missing_year,
        "discarded_missing_country": missing_country,
        "discarded_total": len(discarded),
        "retained_from_ncbi": retained_ncbi,
        "retained_from_gisaid": retained_gisaid,
        "final_retained": final_retained,
        "output_fasta": str(output_path),
    }
    return summary, discarded


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge, exact-deduplicate, metadata-annotate, and audit NCBI Virus "
            "and GISAID DENV FASTA files."
        )
    )
    parser.add_argument("--ncbi-dir", type=Path, required=True)
    parser.add_argument("--gisaid-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory receiving DENV*_merged_meta.fasta files.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory receiving CSV/JSON audit reports (default: OUTPUT/preparation_report).",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Optional directory receiving historical DENV*_merged.fasta intermediates.",
    )
    parser.add_argument(
        "--ncbi-template",
        default="{serotype}_ncbi.fasta",
        help="NCBI input filename template.",
    )
    parser.add_argument(
        "--gisaid-template",
        default="{serotype}_gisaid.fasta",
        help="GISAID input filename template.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help="Fail if the total retained cohort differs from this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir or (args.output_dir / "preparation_report")

    summaries: list[dict[str, int | str]] = []
    discarded_rows: list[dict[str, str]] = []

    for serotype in SEROTYPES:
        ncbi_path = resolve_input_path(args.ncbi_dir, args.ncbi_template, serotype)
        gisaid_path = resolve_input_path(args.gisaid_dir, args.gisaid_template, serotype)
        output_path = args.output_dir / f"{serotype}_merged_meta.fasta"
        merged_path = (
            args.merged_dir / f"{serotype}_merged.fasta"
            if args.merged_dir is not None
            else None
        )

        summary, discarded = prepare_serotype(
            serotype=serotype,
            ncbi_path=ncbi_path,
            gisaid_path=gisaid_path,
            output_path=output_path,
            merged_path=merged_path,
        )
        summaries.append(summary)
        discarded_rows.extend(discarded)

        print(
            f"{serotype}: final={summary['final_retained']} "
            f"discarded={summary['discarded_total']} "
            f"cross_source_duplicates={summary['cross_source_exact_duplicates']}"
        )

    summary_fields = list(summaries[0].keys())
    write_csv(report_dir / "preparation_summary.csv", summaries, summary_fields)

    discarded_fields = ["serotype", "source", "year", "country", "reason", "header"]
    write_csv(
        report_dir / "discarded_records.csv",
        discarded_rows,
        discarded_fields,
    )

    total_retained = sum(int(row["final_retained"]) for row in summaries)
    parameters = {
        "deduplication": "exact identity of complete uppercase sequence string",
        "within_source_rule": "retain first header encountered",
        "cross_source_rule": "GISAID takes precedence over NCBI",
        "ambiguity_normalization": "none beyond uppercase conversion",
        "metadata_exclusion_rule": "discard if year or country cannot be extracted",
        "serotypes": list(SEROTYPES),
        "total_retained": total_retained,
        "expected_total": args.expected_total,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preparation_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Total retained: {total_retained}")
    print(f"Summary: {report_dir / 'preparation_summary.csv'}")
    print(f"Discarded: {report_dir / 'discarded_records.csv'}")

    if args.expected_total is not None and total_retained != args.expected_total:
        raise SystemExit(
            f"Expected {args.expected_total} retained records, obtained {total_retained}."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
