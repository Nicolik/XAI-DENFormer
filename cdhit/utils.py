def normalize_seq_id(seq_id) -> str:
    return str(seq_id).strip().lstrip(">")
