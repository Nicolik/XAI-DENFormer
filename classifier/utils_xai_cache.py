import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ATTN_NPZ = 'attention_test.npz'
GXI_NPZ = 'gxi_test.npz'


def _sorted_npy_files(numpy_dir: str, suffix: str) -> List[str]:
    if not os.path.isdir(numpy_dir):
        return []
    return sorted(f for f in os.listdir(numpy_dir) if f.endswith(suffix))


def convert_attention_split(split_dir: str, subset: str = 'test', overwrite: bool = False) -> Optional[str]:
    """Convert old per-sample attention .npy files into one split-level .npz."""
    split_dir = str(split_dir)
    numpy_dir = os.path.join(split_dir, 'numpy')
    out_path = os.path.join(split_dir, f'attention_{subset}.npz')
    if os.path.exists(out_path) and not overwrite:
        print(f'[SKIP] Exists: {out_path}')
        return out_path

    rx = re.compile(rf'{re.escape(subset)}_batch(\d+)_sample(\d+)_class(\d+)_attn\.npy$')
    files = []
    labels = []
    batch_idx = []
    sample_idx = []
    scores = []

    for fname in _sorted_npy_files(numpy_dir, '_attn.npy'):
        m = rx.match(fname)
        if not m:
            continue
        b, s, c = map(int, m.groups())
        arr = np.load(os.path.join(numpy_dir, fname)).astype(np.float32, copy=False)
        if arr.ndim != 1:
            raise ValueError(f'{fname}: expected 1D profile, got {arr.shape}')
        files.append(fname)
        labels.append(c)
        batch_idx.append(b)
        sample_idx.append(s)
        scores.append(arr)

    if not scores:
        print(f'[WARN] No attention .npy files found in {numpy_dir}')
        return None

    os.makedirs(split_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        scores=np.stack(scores, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        batch_idx=np.asarray(batch_idx, dtype=np.int64),
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        files=np.asarray(files),
        subset=np.asarray(subset),
        score_type=np.asarray('attention'),
    )
    print(f'[OK] attention: {len(scores)} profiles -> {out_path}')
    return out_path


def convert_gxi_split(split_dir: str, subset: str = 'test', overwrite: bool = False) -> Optional[str]:
    """Convert old per-sample GxI .npy files into one split-level .npz."""
    split_dir = str(split_dir)
    numpy_dir = os.path.join(split_dir, 'numpy')
    out_path = os.path.join(split_dir, f'gxi_{subset}.npz')
    if os.path.exists(out_path) and not overwrite:
        print(f'[SKIP] Exists: {out_path}')
        return out_path

    rx = re.compile(rf'{re.escape(subset)}_batch(\d+)_sample(\d+)_class(\d+)_target(\d+)_pred(\d+)_gxi\.npy$')
    files = []
    labels = []
    targets = []
    preds = []
    batch_idx = []
    sample_idx = []
    scores = []

    for fname in _sorted_npy_files(numpy_dir, '_gxi.npy'):
        m = rx.match(fname)
        if not m:
            continue
        b, s, c, t, p = map(int, m.groups())
        arr = np.load(os.path.join(numpy_dir, fname)).astype(np.float32, copy=False)
        if arr.ndim != 1:
            raise ValueError(f'{fname}: expected 1D profile, got {arr.shape}')
        files.append(fname)
        labels.append(c)
        targets.append(t)
        preds.append(p)
        batch_idx.append(b)
        sample_idx.append(s)
        scores.append(arr)

    if not scores:
        print(f'[WARN] No GxI .npy files found in {numpy_dir}')
        return None

    os.makedirs(split_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        scores=np.stack(scores, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        preds=np.asarray(preds, dtype=np.int64),
        batch_idx=np.asarray(batch_idx, dtype=np.int64),
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        files=np.asarray(files),
        subset=np.asarray(subset),
        score_type=np.asarray('gxi'),
    )
    print(f'[OK] gxi: {len(scores)} profiles -> {out_path}')
    return out_path


def load_split_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=False)
    scores = data['scores']
    labels = data['labels']
    meta = {k: data[k] for k in data.files if k not in {'scores', 'labels'}}
    return scores, labels, meta


def find_split_npz(split_dir: str, score_type: str, subset: str = 'test') -> Optional[str]:
    fname = f'{score_type}_{subset}.npz'
    path = os.path.join(split_dir, fname)
    return path if os.path.exists(path) else None



def infer_valid_mask_from_samples(samples_subset: np.ndarray) -> np.ndarray:
    """Infer real (non-padding) nucleotide positions from encoded samples.

    Padding is assumed to be encoded as all-zero vectors. For 3D arrays
    [N, L, C], a position is valid when any channel is non-zero. For 2D
    arrays [N, L], a position is valid when the value is non-zero.
    """
    arr = np.asarray(samples_subset)
    if arr.ndim == 3:
        return np.any(arr != 0, axis=-1)
    if arr.ndim == 2:
        return arr != 0
    raise ValueError(f'Expected samples with shape [N,L] or [N,L,C], got {arr.shape}')


def _align_valid_mask(valid_mask: Optional[np.ndarray], scores: np.ndarray, npz_path: str) -> Optional[np.ndarray]:
    if valid_mask is None:
        return None
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f'{npz_path}: valid_mask must be 2D [N,L], got {valid_mask.shape}')
    if valid_mask.shape != scores.shape:
        n = min(valid_mask.shape[0], scores.shape[0])
        l = min(valid_mask.shape[1], scores.shape[1])
        print(
            f'[WARN] {npz_path}: valid_mask shape {valid_mask.shape} != scores shape {scores.shape}; '
            f'using aligned slice ({n}, {l})'
        )
        scores[:] = scores[:scores.shape[0], :scores.shape[1]]
        valid_mask = valid_mask[:n, :l]
    return valid_mask


def _normalize_scores_per_sample(scores: np.ndarray, normalize_per_sample: Optional[str], valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    scores = scores.astype(float, copy=False)
    if normalize_per_sample in {'sum1', 'abs_sum1'}:
        work = np.abs(scores) if normalize_per_sample == 'abs_sum1' else scores.copy()
        if valid_mask is not None:
            work = np.where(valid_mask, work, 0.0)
        denom = work.sum(axis=1, keepdims=True)
        return np.divide(work, denom, out=np.zeros_like(work, dtype=float), where=denom != 0)
    if normalize_per_sample not in {None, 'none', 'no'}:
        raise ValueError(f'Unknown normalize_per_sample={normalize_per_sample!r}')
    if valid_mask is not None:
        return np.where(valid_mask, scores, 0.0)
    return scores


def profiles_by_class_from_npz(npz_path: str, class_dict=None, normalize_per_sample: Optional[str] = None,
                               valid_mask: Optional[np.ndarray] = None):
    scores, labels, meta = load_split_npz(npz_path)
    scores = scores.astype(float, copy=False)
    labels = labels.astype(int, copy=False)
    if valid_mask is None and 'valid_mask' in meta:
        valid_mask = meta['valid_mask']
    valid_mask = _align_valid_mask(valid_mask, scores, npz_path)
    if valid_mask is not None:
        n = min(scores.shape[0], valid_mask.shape[0])
        l = min(scores.shape[1], valid_mask.shape[1])
        scores = scores[:n, :l]
        labels = labels[:n]
        valid_mask = valid_mask[:n, :l]

    scores = _normalize_scores_per_sample(scores, normalize_per_sample, valid_mask=valid_mask)

    class_groups = {}
    class_masks = {}
    for class_idx in sorted(np.unique(labels)):
        mask = labels == class_idx
        class_name = f'class {class_idx}' if class_dict is None else class_dict[int(class_idx)]
        class_groups[class_name] = scores[mask]
        class_masks[class_name] = valid_mask[mask] if valid_mask is not None else np.ones_like(scores[mask], dtype=bool)
    return class_groups, class_masks


def normalize_class_profiles(raw_profiles, normalize='class'):
    if not raw_profiles or normalize in {None, 'no'}:
        return raw_profiles
    profiles = {k: np.asarray(v, dtype=float) for k, v in raw_profiles.items()}
    if normalize == 'global':
        all_values = np.concatenate([v[np.isfinite(v)] for v in profiles.values() if np.any(np.isfinite(v))])
        if all_values.size == 0:
            return {k: v for k, v in profiles.items()}
        vmin, vmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        if vmax > vmin:
            return {k: np.where(np.isfinite(v), (v - vmin) / (vmax - vmin), np.nan) for k, v in profiles.items()}
        return {k: np.where(np.isfinite(v), 0.0, np.nan) for k, v in profiles.items()}
    if normalize == 'class':
        out = {}
        for k, v in profiles.items():
            finite = np.isfinite(v)
            if not np.any(finite):
                out[k] = v
                continue
            vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
            out[k] = np.where(finite, (v - vmin) / (vmax - vmin), np.nan) if vmax > vmin else np.where(finite, 0.0, np.nan)
        return out
    raise ValueError(f'Unknown normalize={normalize!r}. Use no, global, or class.')


def aggregate_profiles_from_npz(npz_paths: Iterable[str], output_dir: str, prefix: str, class_dict=None,
                                normalize: str = 'class', divide: bool = True,
                                normalize_per_sample: Optional[str] = None,
                                valid_masks: Optional[Iterable[np.ndarray]] = None):
    os.makedirs(output_dir, exist_ok=True)
    class_sums = {}
    class_valid_counts = {}
    class_sample_counts = {}
    npz_paths = list(npz_paths)
    masks_iter = list(valid_masks) if valid_masks is not None else [None] * len(npz_paths)
    if len(masks_iter) != len(npz_paths):
        raise ValueError(f'valid_masks length ({len(masks_iter)}) must match npz_paths length ({len(npz_paths)})')

    for npz_path, valid_mask in zip(npz_paths, masks_iter):
        if not npz_path or not os.path.exists(npz_path):
            continue
        groups, group_valid_masks = profiles_by_class_from_npz(
            npz_path,
            class_dict=class_dict,
            normalize_per_sample=normalize_per_sample,
            valid_mask=valid_mask,
        )
        for class_name, stack in groups.items():
            if stack.size == 0:
                continue
            mask_stack = group_valid_masks[class_name]
            if class_name not in class_sums:
                class_sums[class_name] = np.zeros(stack.shape[1], dtype=float)
                class_valid_counts[class_name] = np.zeros(stack.shape[1], dtype=np.int64)
                class_sample_counts[class_name] = 0
            class_sums[class_name] += np.where(mask_stack, stack, 0.0).sum(axis=0)
            class_valid_counts[class_name] += mask_stack.sum(axis=0)
            class_sample_counts[class_name] += stack.shape[0]

    if not class_sums:
        return {}, {}

    raw = {}
    rows = []
    for class_name in sorted(class_sums):
        if divide:
            counts = class_valid_counts[class_name]
            profile = np.divide(
                class_sums[class_name],
                counts,
                out=np.full_like(class_sums[class_name], np.nan, dtype=float),
                where=counts > 0,
            )
        else:
            profile = class_sums[class_name]
        raw[class_name] = profile
        rows.append({'class_name': class_name, 'n_samples': int(class_sample_counts[class_name])})
        print(
            f'[aggregate npz] {class_name}: n={class_sample_counts[class_name]} | '
            f'valid positions={int(np.sum(class_valid_counts[class_name] > 0))} | '
            f'min={np.nanmin(profile):.6g} | max={np.nanmax(profile):.6g}'
        )

    profiles = normalize_class_profiles(raw, normalize=normalize)
    for class_name, profile in profiles.items():
        out_path = os.path.join(output_dir, f'{prefix}_{class_name}_sum.npy')
        np.save(out_path, profile)
        valid_out_path = os.path.join(output_dir, f'{prefix}_{class_name}_valid_counts.npy')
        np.save(valid_out_path, class_valid_counts[class_name])
        print(f'[INFO] Saved aggregate profile for {class_name} -> {out_path}')

    counts_path = os.path.join(output_dir, f'{prefix}_counts.csv')
    pd.DataFrame(rows).to_csv(counts_path, index=False)
    return profiles, {row['class_name']: row['n_samples'] for row in rows}


def aggregate_overall_profile_from_npz(npz_paths: Iterable[str], output_dir: str, prefix: str,
                                       normalize: str = 'class', divide: bool = True,
                                       normalize_per_sample: Optional[str] = None,
                                       valid_masks: Optional[Iterable[np.ndarray]] = None,
                                       label: str = 'Overall'):
    """Aggregate all samples together, independent of serotype."""
    os.makedirs(output_dir, exist_ok=True)
    npz_paths = list(npz_paths)
    masks_iter = list(valid_masks) if valid_masks is not None else [None] * len(npz_paths)
    if len(masks_iter) != len(npz_paths):
        raise ValueError(f'valid_masks length ({len(masks_iter)}) must match npz_paths length ({len(npz_paths)})')

    total_sum = None
    total_valid_counts = None
    n_samples = 0

    for npz_path, valid_mask in zip(npz_paths, masks_iter):
        if not npz_path or not os.path.exists(npz_path):
            continue
        scores, labels, meta = load_split_npz(npz_path)
        scores = scores.astype(float, copy=False)
        if valid_mask is None and 'valid_mask' in meta:
            valid_mask = meta['valid_mask']
        valid_mask = _align_valid_mask(valid_mask, scores, npz_path)
        if valid_mask is not None:
            n = min(scores.shape[0], valid_mask.shape[0])
            l = min(scores.shape[1], valid_mask.shape[1])
            scores = scores[:n, :l]
            valid_mask = valid_mask[:n, :l]
        else:
            valid_mask = np.ones_like(scores, dtype=bool)
        scores = _normalize_scores_per_sample(scores, normalize_per_sample, valid_mask=valid_mask)

        if total_sum is None:
            total_sum = np.zeros(scores.shape[1], dtype=float)
            total_valid_counts = np.zeros(scores.shape[1], dtype=np.int64)
        total_sum[:scores.shape[1]] += np.where(valid_mask, scores, 0.0).sum(axis=0)
        total_valid_counts[:scores.shape[1]] += valid_mask.sum(axis=0)
        n_samples += scores.shape[0]

    if total_sum is None:
        return {}, {'n_samples': 0}

    if divide:
        profile = np.divide(
            total_sum,
            total_valid_counts,
            out=np.full_like(total_sum, np.nan, dtype=float),
            where=total_valid_counts > 0,
        )
    else:
        profile = total_sum
    profiles = normalize_class_profiles({label: profile}, normalize=normalize)
    profile = profiles[label]

    out_path = os.path.join(output_dir, f'{prefix}_{safe_filename(label)}_sum.npy')
    np.save(out_path, profile)
    np.save(os.path.join(output_dir, f'{prefix}_{safe_filename(label)}_valid_counts.npy'), total_valid_counts)
    pd.DataFrame([{'profile': label, 'n_samples': int(n_samples)}]).to_csv(
        os.path.join(output_dir, f'{prefix}_counts.csv'), index=False
    )
    print(f'[INFO] Saved overall aggregate profile -> {out_path}')
    return {label: profile}, {'n_samples': int(n_samples)}


def safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'profile'


def build_region_long_df_from_npz(npz_path: str, regions: dict, class_dict: dict, value_col: str,
                                  region_reduce: str = 'mean', normalize_per_sample: bool = False):
    reducers = {'mean': np.mean, 'sum': np.sum, 'median': np.median}
    if region_reduce not in reducers:
        raise ValueError(f'region_reduce must be one of {list(reducers)}')
    reduce_fn = reducers[region_reduce]

    scores, labels, meta = load_split_npz(npz_path)
    scores = scores.astype(float, copy=False)
    labels = labels.astype(int, copy=False)
    files = meta.get('files', np.asarray([f'sample_{i}' for i in range(scores.shape[0])]))
    batch_idx = meta.get('batch_idx', np.full(scores.shape[0], -1))
    sample_idx = meta.get('sample_idx', np.arange(scores.shape[0]))

    rows = []
    for i in range(scores.shape[0]):
        arr = scores[i]
        if normalize_per_sample:
            denom = np.abs(arr).sum()
            if denom > 0:
                arr = np.abs(arr) / denom
        class_idx = int(labels[i])
        serotype = class_dict.get(class_idx, f'class {class_idx}')
        for region_name, (start, end) in regions.items():
            start = max(0, int(start))
            end = min(len(arr), int(end))
            if end <= start:
                continue
            rows.append({
                'batch': int(batch_idx[i]),
                'sample': int(sample_idx[i]),
                'class': class_idx,
                'serotype': serotype,
                'region': region_name,
                value_col: float(reduce_fn(arr[start:end])),
                'file': str(files[i]),
            })
    return pd.DataFrame(rows)


def delete_single_npy(split_dir: str, score_type: str, subset: str = 'test', require_npz: bool = True) -> int:
    npz_path = find_split_npz(split_dir, score_type, subset=subset)
    if require_npz and not npz_path:
        print(f'[SKIP] Missing {score_type}_{subset}.npz, not deleting: {split_dir}')
        return 0
    numpy_dir = os.path.join(split_dir, 'numpy')
    suffix = '_attn.npy' if score_type == 'attention' else '_gxi.npy'
    count = 0
    for fname in _sorted_npy_files(numpy_dir, suffix):
        os.remove(os.path.join(numpy_dir, fname))
        count += 1
    try:
        if os.path.isdir(numpy_dir) and not os.listdir(numpy_dir):
            os.rmdir(numpy_dir)
    except OSError:
        pass
    print(f'[OK] deleted {count} {suffix} files from {numpy_dir}')
    return count
