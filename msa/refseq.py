from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO, pairwise2
from Bio.SeqRecord import SeqRecord

from .config import FEATURE_TYPES, PRODUCT_TO_REGION, REFSEQ_FILES, REGION_ORDER, REGION_TO_PRODUCT, SEROTYPES


def get_product_name(feature) -> str:
    for key in ("product", "gene", "note"):
        values = feature.qualifiers.get(key)
        if values:
            return str(values[0])
    return "unknown"


def normalize_region_name(product_name: str) -> str | None:
    key = " ".join(product_name.lower().replace("-", " ").split())
    if key in PRODUCT_TO_REGION:
        return PRODUCT_TO_REGION[key]
    for known, region in PRODUCT_TO_REGION.items():
        if known in key:
            return region
    return None


def load_refseq_records(refseq_dir: Path) -> Dict[str, SeqRecord]:
    records = {}
    for denv, filename in REFSEQ_FILES.items():
        path = refseq_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing RefSeq GenBank file: {path}")
        records[denv] = SeqIO.read(path, "genbank")
    return records


def extract_region_rows(record: SeqRecord) -> List[dict]:
    rows = []
    seen = set()

    for feat in record.features:
        if feat.type not in FEATURE_TYPES:
            continue
        product = get_product_name(feat)
        region = normalize_region_name(product)
        if region is None or region in seen:
            continue
        start = int(feat.location.start) + 1
        end = int(feat.location.end)
        rows.append({
            "Region": region,
            "Proteina": REGION_TO_PRODUCT.get(region, product),
            "Start_nt": start,
            "End_nt": end,
            "Lunghezza_nt": end - start + 1,
        })
        seen.add(region)

    rows = sorted(rows, key=lambda r: r["Start_nt"])
    if not rows:
        raise ValueError(f"No mature protein features found in {record.id}")

    first_start = rows[0]["Start_nt"]
    last_end = rows[-1]["End_nt"]
    genome_len = len(record.seq)

    if first_start > 1:
        rows.insert(0, {
            "Region": "5'UTR",
            "Proteina": "5'UTR",
            "Start_nt": 1,
            "End_nt": first_start - 1,
            "Lunghezza_nt": first_start - 1,
        })
    if last_end < genome_len:
        rows.append({
            "Region": "3'UTR",
            "Proteina": "3'UTR",
            "Start_nt": last_end + 1,
            "End_nt": genome_len,
            "Lunghezza_nt": genome_len - last_end,
        })

    order = {name: i for i, name in enumerate(REGION_ORDER)}
    return sorted(rows, key=lambda r: order.get(r["Region"], 999))


def _write_coordinate_rows(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Region", "Proteina", "Start_nt", "End_nt", "Lunghezza_nt"])
        writer.writeheader()
        writer.writerows(rows)


def export_refseq_coordinates(refseq_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_refseq_records(refseq_dir)
    for denv, record in records.items():
        rows = extract_region_rows(record)
        out_path = output_dir / f"coordinates_dengue_{denv}.csv"
        _write_coordinate_rows(rows, out_path)
        print(f"[OK] {denv}: {out_path}")


def _clean_sequence(seq: str) -> str:
    return str(seq).upper().replace("U", "T").replace("-", "")


def _find_genome_fasta(genomes_dir: Path, denv: str) -> Path:
    candidates = [
        genomes_dir / f"{denv}_merged_meta.fasta",
        genomes_dir / f"{denv}_merged.fasta",
        genomes_dir / f"{denv}.fasta",
        genomes_dir / f"{denv}.fa",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(genomes_dir.glob(f"{denv}*.fa*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing genome FASTA for {denv} under {genomes_dir}")


def _load_longest_record(genomes_dir: Path, denv: str) -> Tuple[SeqRecord, Path]:
    fasta_path = _find_genome_fasta(genomes_dir, denv)
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    longest = max(records, key=lambda r: len(_clean_sequence(str(r.seq))))
    return longest, fasta_path


def _map_ref_to_target_positions(ref_seq: str, target_seq: str) -> Tuple[Dict[int, int], float]:
    alignment = pairwise2.align.globalms(
        ref_seq,
        target_seq,
        2,
        -1,
        -5,
        -0.5,
        one_alignment_only=True,
    )[0]

    mapping: Dict[int, int] = {}
    ref_pos = 0
    target_pos = 0
    for ref_base, target_base in zip(alignment.seqA, alignment.seqB):
        if ref_base != "-":
            ref_pos += 1
        if target_base != "-":
            target_pos += 1
        if ref_base != "-" and ref_pos not in mapping:
            mapping[ref_pos] = target_pos
    return mapping, float(alignment.score)


def _map_rows_to_longest(ref_rows: List[dict], mapping: Dict[int, int]) -> List[dict]:
    mapped_rows = []
    for row in ref_rows:
        start = int(row["Start_nt"])
        end = int(row["End_nt"])
        mapped_start = mapping.get(start)
        mapped_end = mapping.get(end)
        if mapped_start is None or mapped_end is None:
            print(f"[WARN] Missing coordinate mapping for {row['Region']}: {start}-{end}; skipping")
            continue
        if mapped_start > mapped_end:
            mapped_start, mapped_end = mapped_end, mapped_start
        mapped = dict(row)
        mapped["Start_nt"] = mapped_start
        mapped["End_nt"] = mapped_end
        mapped["Lunghezza_nt"] = mapped_end - mapped_start + 1
        mapped_rows.append(mapped)
    return mapped_rows


def export_longest_coordinates(
    genomes_dir: Path,
    refseq_dir: Path,
    output_dir: Path,
    longest_filename: str = "coordinates_dengue_LONGEST.csv",
) -> None:
    """Export coordinates mapped from each RefSeq to the longest available genome.

    The per-serotype files are written as ``coordinates_dengue_<DENV>_LONGEST.csv``.
    The globally longest genome among DENV1-DENV4 is also copied to
    ``coordinates_dengue_LONGEST.csv`` for attention plots that use the common
    padded/truncated model length.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_records = load_refseq_records(Path(refseq_dir))

    summary_rows = []
    global_longest: Optional[dict] = None

    for denv in SEROTYPES:
        longest_record, fasta_path = _load_longest_record(Path(genomes_dir), denv)
        target_seq = _clean_sequence(str(longest_record.seq))
        ref_seq = _clean_sequence(str(ref_records[denv].seq))
        print(f"[INFO] {denv}: longest={longest_record.id} length={len(target_seq)} source={fasta_path}")

        mapping, score = _map_ref_to_target_positions(ref_seq, target_seq)
        ref_rows = extract_region_rows(ref_records[denv])
        mapped_rows = _map_rows_to_longest(ref_rows, mapping)
        out_path = output_dir / f"coordinates_dengue_{denv}_LONGEST.csv"
        _write_coordinate_rows(mapped_rows, out_path)
        print(f"[OK] {denv}: {out_path} alignment_score={score:.1f}")

        row = {
            "Serotype": denv,
            "Record_ID": longest_record.id,
            "Length_nt": len(target_seq),
            "Source_FASTA": str(fasta_path),
            "Coordinates_CSV": str(out_path),
            "Alignment_score": score,
        }
        summary_rows.append(row)
        if global_longest is None or row["Length_nt"] > global_longest["Length_nt"]:
            global_longest = {**row, "mapped_rows": mapped_rows}

    summary_path = output_dir / "longest_sequences.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Serotype", "Record_ID", "Length_nt", "Source_FASTA", "Coordinates_CSV", "Alignment_score"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] longest summary: {summary_path}")

    if global_longest is None:
        raise RuntimeError("Could not determine global longest genome")
    longest_path = output_dir / longest_filename
    _write_coordinate_rows(global_longest["mapped_rows"], longest_path)
    print(
        f"[OK] global longest map: {longest_path} "
        f"({global_longest['Serotype']}, {global_longest['Record_ID']}, {global_longest['Length_nt']} nt)"
    )


def load_coordinates(coordinates_dir: Path) -> Dict[str, Dict[str, tuple[int, int]]]:
    all_coords = {}
    for denv in SEROTYPES:
        path = coordinates_dir / f"coordinates_dengue_{denv}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing coordinates file: {path}")
        coords = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                region = row.get("Region") or row.get("Proteina")
                coords[region] = (int(row["Start_nt"]), int(row["End_nt"]))
        all_coords[denv] = coords
    return all_coords


def write_region_fastas(refseq_dir: Path, coordinates_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_refseq_records(refseq_dir)
    coords = load_coordinates(coordinates_dir)

    for region in REGION_ORDER:
        safe_region = region.replace("'", "").replace("/", "_")
        out_path = output_dir / f"{safe_region}.fasta"
        written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for denv in SEROTYPES:
                if region not in coords[denv]:
                    print(f"[WARN] {denv}: region not found, skipping {region}")
                    continue
                start, end = coords[denv][region]
                seq = records[denv].seq[start - 1:end]
                f.write(f">{denv}|{records[denv].id}|{region}|{start}-{end}\n")
                f.write(str(seq).upper() + "\n")
                written += 1
        print(f"[OK] {region}: wrote {written} sequences -> {out_path}")
