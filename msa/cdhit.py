from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .config import REGION_ORDER, SEROTYPES
from .refseq import load_coordinates, load_refseq_records

FASTA_EXTENSIONS = ("*.fasta", "*.fa", "*.fna")


def infer_serotype(text: str) -> Optional[str]:
    t = str(text).upper()
    patterns = [
        r"\bDENV[-_ ]?([1-4])\b",
        r"\bDEN[-_ ]?V[-_ ]?([1-4])\b",
        r"\bDENGUE[-_ ]?VIRUS[-_ ]?([1-4])\b",
        r"\bH?DENV([1-4])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            return f"DENV{m.group(1)}"
    return None


def expected_cdhit_fasta(input_dir: Path, denv: str) -> Path:
    return Path(input_dir) / denv / f"{denv}_cdhit.fasta"


def find_cdhit_fasta_files(input_dir: Path) -> List[Path]:
    input_dir = Path(input_dir)
    expected = [expected_cdhit_fasta(input_dir, denv) for denv in SEROTYPES]
    if all(p.exists() for p in expected):
        return expected

    files: List[Path] = []
    for ext in FASTA_EXTENSIONS:
        files.extend(sorted(input_dir.rglob(ext)))
    files = [p for p in files if "cdhit" in p.name.lower()]
    return sorted(set(files))


def clean_sequence(seq: str) -> str:
    seq = str(seq).upper().replace("U", "T").replace("-", "")
    return re.sub(r"[^ACGTN]", "N", seq)


def load_existing_cdhit_records(input_dir: Path) -> Dict[str, List[SeqRecord]]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"CD-HIT directory not found: {input_dir}")

    fasta_files = find_cdhit_fasta_files(input_dir)
    if not fasta_files:
        raise FileNotFoundError(f"No CD-HIT FASTA files found under {input_dir}")

    records_by_serotype: Dict[str, List[SeqRecord]] = {s: [] for s in SEROTYPES}

    for fasta_path in fasta_files:
        filename_serotype = infer_serotype(fasta_path.name) or infer_serotype(str(fasta_path.parent))
        for rec in SeqIO.parse(fasta_path, "fasta"):
            denv = infer_serotype(rec.description) or infer_serotype(rec.id) or filename_serotype
            if denv not in records_by_serotype:
                print(f"[WARN] Could not infer serotype, skipping: {rec.id}")
                continue
            rec.seq = Seq(clean_sequence(str(rec.seq)))
            records_by_serotype[denv].append(rec)

    for denv in SEROTYPES:
        print(f"[INFO] Loaded {len(records_by_serotype[denv])} existing CD-HIT representatives for {denv}")
        if len(records_by_serotype[denv]) == 0:
            raise ValueError(f"No CD-HIT representatives loaded for {denv} from {input_dir}")

    return records_by_serotype


def subsample_records(
    records_by_serotype: Dict[str, List[SeqRecord]],
    max_per_serotype: Optional[int] = None,
    seed: int = 13,
) -> Dict[str, List[SeqRecord]]:
    if max_per_serotype is None or max_per_serotype <= 0:
        return records_by_serotype

    rng = random.Random(seed)
    sampled: Dict[str, List[SeqRecord]] = {}
    for denv, records in records_by_serotype.items():
        if len(records) <= max_per_serotype:
            sampled[denv] = records
        else:
            sampled[denv] = rng.sample(records, max_per_serotype)
        print(f"[INFO] Using {len(sampled[denv])}/{len(records)} records for {denv}")
    return sampled


def build_ref_to_target_alignment_map(ref_seq: str, target_seq: str) -> Tuple[Dict[int, List[str]], float]:
    """Map each 1-based RefSeq position to target characters from a global alignment.

    Target insertions are attached to the previous RefSeq position. This lets us
    extract every viral region from one alignment per representative genome.
    """
    alignment = pairwise2.align.globalms(
        ref_seq,
        target_seq,
        2,
        -1,
        -5,
        -0.5,
        one_alignment_only=True,
    )[0]

    ref_pos = 0
    last_ref_pos = 0
    mapping: Dict[int, List[str]] = {}

    for r, t in zip(alignment.seqA, alignment.seqB):
        if r != "-":
            ref_pos += 1
            last_ref_pos = ref_pos
            mapping.setdefault(ref_pos, [])
            if t != "-":
                mapping[ref_pos].append(t.upper())
        elif t != "-" and last_ref_pos > 0:
            mapping.setdefault(last_ref_pos, []).append(t.upper())

    return mapping, float(alignment.score)


def extract_region_from_mapping(mapping: Dict[int, List[str]], start_1based: int, end_1based: int) -> str:
    chars: List[str] = []
    for pos in range(start_1based, end_1based + 1):
        chars.extend(mapping.get(pos, []))
    return "".join(chars).replace("U", "T")


def sanitize_record_id(record_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(record_id))[:180]


def write_existing_cdhit_region_fastas(
    cdhit_dir: Path,
    refseq_dir: Path,
    coordinates_dir: Path,
    output_dir: Path,
    max_per_serotype: Optional[int] = None,
    seed: int = 13,
    min_region_length_fraction: float = 0.65,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cdhit_records = load_existing_cdhit_records(Path(cdhit_dir))
    cdhit_records = subsample_records(cdhit_records, max_per_serotype, seed)
    ref_records = load_refseq_records(Path(refseq_dir))
    coords = load_coordinates(Path(coordinates_dir))

    handles = {}
    counts = {region: 0 for region in REGION_ORDER}
    skipped_short = {region: 0 for region in REGION_ORDER}

    try:
        for region in REGION_ORDER:
            safe_region = region.replace("'", "").replace("/", "_")
            handles[region] = (output_dir / f"{safe_region}.fasta").open("w", encoding="utf-8")

        for denv in SEROTYPES:
            ref_seq = clean_sequence(str(ref_records[denv].seq))
            records = cdhit_records[denv]
            for rec_idx, record in enumerate(records, start=1):
                target_seq = clean_sequence(str(record.seq))
                if not target_seq:
                    continue

                mapping, score = build_ref_to_target_alignment_map(ref_seq, target_seq)
                clean_id = sanitize_record_id(record.id)

                for region in REGION_ORDER:
                    if region not in coords[denv]:
                        continue
                    start, end = coords[denv][region]
                    expected_len = end - start + 1
                    region_seq = extract_region_from_mapping(mapping, start, end)
                    if len(region_seq) < expected_len * min_region_length_fraction:
                        skipped_short[region] += 1
                        continue
                    handles[region].write(f">{denv}|{clean_id}|{region}|ref:{start}-{end}\n")
                    handles[region].write(region_seq + "\n")
                    counts[region] += 1

                if rec_idx % 25 == 0 or rec_idx == len(records):
                    print(f"[INFO] {denv}: processed {rec_idx}/{len(records)} representatives")

        for region in REGION_ORDER:
            print(
                f"[OK] {region}: wrote {counts[region]} sequences; "
                f"skipped_short={skipped_short[region]}"
            )
    finally:
        for handle in handles.values():
            handle.close()


# Backward-compatible alias used by earlier scripts.
def write_cdhit_region_fastas(*args, **kwargs) -> None:
    write_existing_cdhit_region_fastas(*args, **kwargs)
