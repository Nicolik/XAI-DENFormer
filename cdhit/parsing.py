import re
from pathlib import Path

from Bio import SeqIO
from typing import Dict, List, Tuple

from cdhit.config import fasta_files_by_serotype
from cdhit.utils import normalize_seq_id


def choose_word_size(identity: float) -> str:
    if identity >= 0.98:
        return "10"
    if identity >= 0.95:
        return "9"
    if identity >= 0.90:
        return "8"
    if identity >= 0.88:
        return "7"
    if identity >= 0.85:
        return "6"
    if identity >= 0.80:
        return "5"
    if identity >= 0.75:
        return "4"
    raise ValueError("Identity threshold too low for cd-hit-est word size.")


def infer_serotype_name(filename: str) -> str:
    name = filename.upper()
    patterns = [
        r"DENV[\-_ ]?([1-4])",
        r"SEROTYPE[\-_ ]?([1-4])",
        r"TYPE[\-_ ]?([1-4])",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return f"DENV{match.group(1)}"

    raise ValueError(f"Could not infer serotype from filename: {filename}")


def count_fasta_records(fasta_path: Path) -> int:
    count = 0
    with open(fasta_path, "r") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def parse_clstr(clstr_path: Path) -> List[dict]:
    clusters = []
    current_cluster_id = None
    current_members = []
    representative = None

    with open(clstr_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">Cluster"):
                if current_cluster_id is not None:
                    clusters.append({
                        "cluster_id": current_cluster_id,
                        "members": current_members,
                        "representative": representative,
                        "size": len(current_members),
                    })

                current_cluster_id = int(line.split()[1])
                current_members = []
                representative = None
                continue

            if ">" not in line or "..." not in line:
                continue

            seq_id = line.split(">", 1)[1].split("...", 1)[0]
            current_members.append(seq_id)

            if line.endswith("*"):
                representative = seq_id

    if current_cluster_id is not None:
        clusters.append({
            "cluster_id": current_cluster_id,
            "members": current_members,
            "representative": representative,
            "size": len(current_members),
        })

    return clusters


def build_dataset_index_map(input_dir: Path) -> Tuple[Dict[str, int], List[str]]:
    seq_to_index = {}
    all_fasta_ids = []
    index = 0

    for filename in fasta_files_by_serotype:
        fasta_path = input_dir / filename

        if not fasta_path.exists():
            print(f"WARNING: FASTA file not found: {fasta_path}")
            continue

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            candidates = {
                normalize_seq_id(record.id),
                normalize_seq_id(record.name),
                normalize_seq_id(record.description),
            }

            for key in candidates:
                if key:
                    seq_to_index[key] = index
                    all_fasta_ids.append(key)

            index += 1

    print(f"Built dataset index map with {len(seq_to_index)} sequence identifiers")
    print(f"Total indexed FASTA records: {index}")

    return seq_to_index, all_fasta_ids


def find_dataset_index(seq, seq_to_index: Dict[str, int], all_fasta_ids: List[str]):
    seq_norm = normalize_seq_id(seq)

    if seq_norm in seq_to_index:
        return seq_to_index[seq_norm]

    matches = [
        fasta_id
        for fasta_id in all_fasta_ids
        if fasta_id.startswith(seq_norm) or seq_norm.startswith(fasta_id)
    ]
    matches = sorted(set(matches))

    if len(matches) == 1:
        print("Resolved truncated CD-HIT id by prefix match:")
        print(f"  CD-HIT: {seq_norm}")
        print(f"  FASTA : {matches[0]}")
        return seq_to_index[matches[0]]

    if len(matches) > 1:
        matched_indices = sorted(set(seq_to_index[m] for m in matches))

        if len(matched_indices) == 1:
            print("Resolved truncated CD-HIT id by prefix match with duplicated FASTA aliases:")
            print(f"  CD-HIT: {seq_norm}")
            print(f"  FASTA matches: {len(matches)}")
            print(f"  Dataset index: {matched_indices[0]}")
            return matched_indices[0]

        print("\nAmbiguous prefix match for CD-HIT id:")
        print(seq_norm)
        print("\nFirst 20 FASTA matches:")
        for match in matches[:20]:
            print(f"  index={seq_to_index[match]} | {match}")

        raise ValueError(
            f"Ambiguous prefix match for CD-HIT id {seq_norm!r}: "
            f"{len(matches)} FASTA ids matched across {len(matched_indices)} dataset indices."
        )

    return None
