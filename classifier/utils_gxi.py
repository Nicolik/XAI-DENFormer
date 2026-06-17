import os
import re
import numpy as np
import torch


def gradient_x_input(model, inputs, labels=None, target_mode='pred', score_mode='signed'):
    """
    Compute Gradient x Input (GxI) contribution profiles for a classifier batch.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier returning logits [B, C].
    inputs : torch.Tensor
        Input tensor [B, L, D]. It can be OHE or embedding-based.
    labels : torch.Tensor or None
        Ground-truth labels [B]. Required when target_mode='true'.
    target_mode : {'pred', 'true'}
        Class score used as attribution target.
    score_mode : {'signed', 'abs', 'positive'}
        How to reduce channel-wise input*gradient contributions.

    Returns
    -------
    scores : torch.Tensor
        Per-token contribution profile [B, L].
    targets : torch.Tensor
        Class index used as gradient target [B].
    preds : torch.Tensor
        Predicted class [B].
    """
    if target_mode not in {'pred', 'true'}:
        raise ValueError("target_mode must be 'pred' or 'true'.")
    if score_mode not in {'signed', 'abs', 'positive'}:
        raise ValueError("score_mode must be 'signed', 'abs', or 'positive'.")

    x = inputs.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = torch.argmax(logits.detach(), dim=1)
    if target_mode == 'pred':
        targets = preds
    else:
        if labels is None:
            raise ValueError("labels are required when target_mode='true'.")
        targets = labels.detach().long()

    selected = logits.gather(1, targets.view(-1, 1)).sum()
    selected.backward()

    contrib = x.grad * x
    if score_mode == 'signed':
        scores = contrib.sum(dim=-1)
    elif score_mode == 'abs':
        scores = contrib.abs().sum(dim=-1)
    else:
        scores = torch.clamp(contrib, min=0.0).sum(dim=-1)

    return scores.detach(), targets.detach(), preds.detach()


def save_gxi(scores, out_dir, prefix='example', labels=None, targets=None, preds=None, save_npy=True):
    """Save per-sample GxI .npy files using labels in filenames for downstream aggregation."""
    os.makedirs(out_dir, exist_ok=True)
    npy_out_dir = os.path.join(out_dir, 'numpy')
    if save_npy:
        os.makedirs(npy_out_dir, exist_ok=True)

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    if labels is not None and torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if targets is not None and torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if preds is not None and torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    for i in range(scores.shape[0]):
        sample_prefix = f'{prefix}_sample{i}'
        if labels is not None:
            sample_prefix += f'_class{int(labels[i])}'
        if targets is not None:
            sample_prefix += f'_target{int(targets[i])}'
        if preds is not None:
            sample_prefix += f'_pred{int(preds[i])}'
        if save_npy:
            np.save(os.path.join(npy_out_dir, f'{sample_prefix}_gxi.npy'), scores[i])


def normalize_profiles(raw_profiles, normalize='class'):
    if not raw_profiles or normalize == 'no':
        return raw_profiles
    profiles = {k: np.asarray(v, dtype=float) for k, v in raw_profiles.items()}
    if normalize == 'global':
        all_values = np.concatenate(list(profiles.values()))
        vmin, vmax = float(all_values.min()), float(all_values.max())
        if vmax > vmin:
            return {k: (v - vmin) / (vmax - vmin) for k, v in profiles.items()}
        return {k: np.zeros_like(v) for k, v in profiles.items()}
    if normalize == 'class':
        out = {}
        for k, v in profiles.items():
            vmin, vmax = float(v.min()), float(v.max())
            out[k] = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
        return out
    raise ValueError(f'Unknown normalize={normalize!r}. Use no, global, or class.')


def sum_gxi_by_class(input_dir, output_dir, prefix='class', class_dict=None, normalize='class', divide=True):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r'batch(\d+)_sample(\d+)_class(\d+).*_gxi\.npy')
    class_groups = {}
    for fname in os.listdir(input_dir):
        if not fname.endswith('_gxi.npy'):
            continue
        match = pattern.search(fname)
        if not match:
            continue
        class_idx = int(match.group(3))
        arr = np.load(os.path.join(input_dir, fname))
        class_groups.setdefault(class_idx, []).append(arr)

    if not class_groups:
        print(f'[WARN] No GxI .npy files found in {input_dir}')
        return {}

    raw_profiles = {}
    for class_idx, arrays in sorted(class_groups.items()):
        class_name = f'class {class_idx}' if class_dict is None else class_dict[class_idx]
        stack = np.stack(arrays, axis=0)
        profile = stack.mean(axis=0) if divide else stack.sum(axis=0)
        raw_profiles[class_name] = profile
        print(f'[sum_gxi_by_class] {class_name}: n={len(arrays)} | min={profile.min():.6g} | max={profile.max():.6g}')

    profiles = normalize_profiles(raw_profiles, normalize=normalize)
    for class_name, profile in profiles.items():
        out_path = os.path.join(output_dir, f'{prefix}_{class_name}_sum.npy')
        np.save(out_path, profile)
        print(f'[INFO] Saved GxI profile for {class_name} -> {out_path}')
    return profiles


def sum_gxi_by_class_from_dirs(input_dirs, output_dir, prefix='class', class_dict=None, normalize='class', divide=False):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r'batch(\d+)_sample(\d+)_class(\d+).*_gxi\.npy')
    class_sums = {}
    class_counts = {}
    total_files = 0

    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f'[WARN] Missing GxI numpy dir, skipping: {input_dir}')
            continue
        for fname in os.listdir(input_dir):
            if not fname.endswith('_gxi.npy'):
                continue
            match = pattern.search(fname)
            if not match:
                continue
            class_idx = int(match.group(3))
            arr = np.load(os.path.join(input_dir, fname))
            if class_idx not in class_sums:
                class_sums[class_idx] = np.zeros_like(arr, dtype=float)
                class_counts[class_idx] = 0
            class_sums[class_idx] += arr
            class_counts[class_idx] += 1
            total_files += 1

    if total_files == 0:
        print('[WARN] No GxI .npy files found for dataset-level aggregation.')
        return {}, {}

    raw_profiles = {}
    count_rows = []
    for class_idx in sorted(class_sums):
        class_name = f'class {class_idx}' if class_dict is None else class_dict[class_idx]
        profile = class_sums[class_idx]
        if divide:
            profile = profile / max(class_counts[class_idx], 1)
        raw_profiles[class_name] = profile
        count_rows.append({'class_idx': class_idx, 'class_name': class_name, 'n_samples': int(class_counts[class_idx])})
        print(f'[dataset aggregate GxI] {class_name}: n={class_counts[class_idx]} | min={profile.min():.6g} | max={profile.max():.6g}')

    profiles = normalize_profiles(raw_profiles, normalize=normalize)
    for class_name, profile in profiles.items():
        out_path = os.path.join(output_dir, f'{prefix}_{class_name}_sum.npy')
        np.save(out_path, profile)
        print(f'[INFO] Saved dataset-level GxI for {class_name} -> {out_path}')

    return profiles, {row['class_name']: row['n_samples'] for row in count_rows}
