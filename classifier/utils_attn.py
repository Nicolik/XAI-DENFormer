import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Reduce attention per chunk
# ----------------------------
def reduce_attention(attn_layer, reduction="cls", debug=False):
    """
    Reduce one attention layer while staying in torch until the final output.

    Parameters
    ----------
    attn_layer : list[torch.Tensor]
        One layer of chunk-level attentions. Each element has shape [B, H, Lc, Lc].
    reduction : str
        "cls" uses CLS-to-token attention: mean over heads, row 0 -> [B, Lc].
        "mean_query" averages over heads and query positions -> [B, Lc].
    debug : bool
        If True, prints compact tensor diagnostics.

    Returns
    -------
    list[torch.Tensor]
        One tensor per chunk, each [B, Lc], still on the same device as attention.
    """
    if not attn_layer:
        raise ValueError("Empty attention layer received.")

    reduced_chunks = []
    if debug:
        print(
            f"[reduce_attention] chunks={len(attn_layer)} | "
            f"first_shape={tuple(attn_layer[0].shape)} | device={attn_layer[0].device}"
        )

    for chunk_attn in attn_layer:
        # chunk_attn: [B, H, Lc, Lc]
        if reduction == "cls":
            # Average heads, then take CLS/query-0 attention to all tokens.
            chunk_scores = chunk_attn.mean(dim=1)[:, 0, :]  # [B, Lc]
        elif reduction == "mean_query":
            # Average heads and all query positions. This is more global but less CLS-specific.
            chunk_scores = chunk_attn.mean(dim=1).mean(dim=1)  # [B, Lc]
        else:
            raise ValueError(f"Unknown attention reduction: {reduction}")

        reduced_chunks.append(chunk_scores)

    if debug:
        print(
            f"[reduce_attention] reduced_chunks={len(reduced_chunks)} | "
            f"first_reduced_shape={tuple(reduced_chunks[0].shape)}"
        )

    return reduced_chunks


# ----------------------------
# Collate chunks into full sequence
# ----------------------------
def collate_attention(reduced_chunks, seq_len=None, debug=False):
    """
    Concatenate reduced chunk scores while staying in torch.

    Parameters
    ----------
    reduced_chunks : list[torch.Tensor]
        One tensor per chunk, each [B, Lc].
    seq_len : int or None
        If provided, final tensor is truncated to this length.
    debug : bool
        If True, prints compact diagnostics.

    Returns
    -------
    torch.Tensor
        Attention scores [B, seq_len or total_chunk_len].
    """
    if not reduced_chunks:
        raise ValueError("No reduced chunks received.")

    scores = torch.cat(reduced_chunks, dim=1)  # [B, total_len]
    if seq_len is not None:
        scores = scores[:, :seq_len]

    if debug:
        print(f"[collate_attention] scores={tuple(scores.shape)} | device={scores.device}")

    return scores


# ----------------------------
# Map scores to genomic coordinates (optional)
# ----------------------------
def map_to_genome(scores, token_to_coord):
    """
    scores: [seq_len]
    token_to_coord: list of (chrom, start, end) for each token index
    Returns: list of tuples (chrom, start, end, score)
    """
    mapped = []
    for idx, score in enumerate(scores):
        chrom, start, end = token_to_coord[idx]
        mapped.append((chrom, start, end, float(score)))
    return mapped


# ----------------------------
# Save and plot scores
# ----------------------------
def save_attention(scores, out_dir, prefix="example", token_to_coord=None,
                   save_npy=True, save_plot=False, save_tsv=False, labels=None, debug=False):
    """
    Save already-reduced attention scores.

    Important for speed: by default this saves only .npy files. PNG/TSV generation is
    intentionally disabled by default because plotting/writing per sample during
    inference is often slower than the attention reduction itself.
    """
    os.makedirs(out_dir, exist_ok=True)

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    if labels is not None and torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    B, seq_len = scores.shape

    plot_out_dir = os.path.join(out_dir, "plot")
    npy_out_dir = os.path.join(out_dir, "numpy")
    tsv_out_dir = os.path.join(out_dir, "text")

    if save_plot:
        os.makedirs(plot_out_dir, exist_ok=True)
    if save_npy:
        os.makedirs(npy_out_dir, exist_ok=True)
    if save_tsv:
        os.makedirs(tsv_out_dir, exist_ok=True)

    for i in range(B):
        sample_prefix = f"{prefix}_sample{i}"
        if labels is not None:
            sample_prefix = f"{sample_prefix}_class{int(labels[i])}"

        if save_npy:
            npy_path = os.path.join(npy_out_dir, f"{sample_prefix}_attn.npy")
            np.save(npy_path, scores[i])
            if debug:
                print(f"[INFO] Saved attention scores to {npy_path}")

        if save_plot:
            plt.figure(figsize=(12, 4))
            plt.plot(scores[i])
            plt.title(f"Attention scores across sequence (sample {i})")
            plt.xlabel("Position (token index)")
            plt.ylabel("Attention (avg heads)")
            plt.tight_layout()
            plot_path = os.path.join(plot_out_dir, f"{sample_prefix}_attn.png")
            plt.savefig(plot_path)
            plt.close()
            if debug:
                print(f"[INFO] Saved attention plot to {plot_path}")

        if save_tsv and token_to_coord is not None:
            mapped = map_to_genome(scores[i], token_to_coord)
            tsv_path = os.path.join(tsv_out_dir, f"{sample_prefix}_attn.tsv")
            with open(tsv_path, "w") as f:
                f.write("chrom\tstart\tend\tscore\n")
                for chrom, start, end, score in mapped:
                    f.write(f"{chrom}\t{start}\t{end}\t{score:.6f}\n")
            if debug:
                print(f"[INFO] Saved genomic mapping to {tsv_path}")


# ----------------------------
# Main function to extract attention for one batch
# ----------------------------
def extract_attention_from_batch(model=None, inputs=None,
                                 out_dir="./attn_outputs", prefix="example0",
                                 chrom="DENV", k=3, return_output=True, labels=None,
                                 attn=None, reduction="cls", layer="last",
                                 save_plot=False, save_tsv=False, debug=False):
    """
    Extract reduced attention for one batch, keeping all reduction operations in torch.

    Preferred fast path
    -------------------
    Pass precomputed `attn` from the same forward pass used for logits. This avoids
    running the model twice per batch.

    Backward-compatible path
    ------------------------
    If `attn` is None, `model` and `inputs` are used to run one forward pass.

    Returns
    -------
    np.ndarray or None
        If return_output=True, returns scores [B, seq_len] as numpy array.
        Otherwise saves them to disk and returns None.
    """
    if attn is None:
        if model is None or inputs is None:
            raise ValueError("Provide either precomputed attn, or both model and inputs.")
        model.eval()
        with torch.no_grad():
            _, attn = model(inputs, return_attn=True)

    if inputs is not None:
        seq_len = inputs.shape[1]
    else:
        seq_len = None

    if debug:
        print(f"[extract_attention_from_batch] inputs.shape: {None if inputs is None else tuple(inputs.shape)}")
        print(f"[extract_attention_from_batch] layers={len(attn)} | chunks={len(attn[0])}")
        print(f"[extract_attention_from_batch] first_attn_shape={tuple(attn[0][0].shape)}")

    if layer == "last":
        selected_layer = attn[-1]
    elif layer == "first":
        selected_layer = attn[0]
    elif isinstance(layer, int):
        selected_layer = attn[layer]
    else:
        raise ValueError("layer must be 'last', 'first', or an integer index.")

    reduced_chunks = reduce_attention(selected_layer, reduction=reduction, debug=debug)
    scores_torch = collate_attention(reduced_chunks, seq_len=seq_len, debug=debug)

    # One CPU transfer only, after all reductions/collation are complete.
    scores_np = scores_torch.detach().cpu().numpy()

    if return_output:
        return scores_np

    token_to_coord = None
    if seq_len is not None and save_tsv:
        token_to_coord = [(chrom, i, i + k) for i in range(seq_len)]

    save_attention(
        scores_np,
        out_dir=out_dir,
        prefix=prefix,
        token_to_coord=token_to_coord,
        labels=labels,
        save_plot=save_plot,
        save_tsv=save_tsv,
        debug=debug,
    )
    return None


def sum_attention_by_class(input_dir, output_dir, prefix="class", class_dict=None, normalize='class', divide=True):
    """
    Compute summed attention profiles grouped by class.
    Optionally apply global normalization across all classes.

    Parameters
    ----------
    input_dir : str
        Directory containing attention .npy files.
    output_dir : str
        Directory to save summed profiles.
    prefix : str
        Prefix for output filenames.
    class_dict : dict
        Mapping from class index -> class name.
    normalize : str ['no', 'global', 'class']
        Whether to normalize the summed profiles (0–1) globally or per-class.
    divide : bool
        Whether to calculate mean (if True) or sum (if False)

    Returns
    -------
    dict
        Dictionary {class_name: summed_profile (np.ndarray)}.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"batch(\d+)_sample(\d+)_class(\d+)_attn\.npy")

    # collect files
    files = [f for f in os.listdir(input_dir) if f.endswith("_attn.npy")]

    # group by class
    class_groups = {}
    for fname in files:
        m = pattern.search(fname)
        if not m:
            continue
        _, _, c = map(int, m.groups())
        arr = np.load(os.path.join(input_dir, fname))
        class_groups.setdefault(c, []).append(arr)

    print(f"[sum_attention_by_class] class_groups.keys: {class_groups.keys()}")
    key = list(class_groups.keys())[0]
    print(f"[sum_attention_by_class] class_groups[key].len: {len(class_groups[key])}")
    idx = 0
    print(f"[sum_attention_by_class] class_groups[key][idx].shape: {class_groups[key][idx].shape}")

    # Step 1: sum across samples per class (no normalization yet)
    raw_profiles = {}
    for c, arrays in class_groups.items():
        class_name = f"class {c}" if class_dict is None else class_dict[c]
        op = np.mean if divide else np.sum
        summed = op(arrays, axis=0)
        raw_profiles[class_name] = summed
        print(f"[sum_attention_by_class] class: {class_name},  "
              f"raw_profiles[{class_name}]: {raw_profiles[class_name].shape}, "
              f"min: {np.min(raw_profiles[class_name]):.3f}, max: {np.max(raw_profiles[class_name]):.3f}")

    # Step 2: global normalization if requested
    if raw_profiles:
        if normalize == 'global':
            all_values = np.concatenate(list(raw_profiles.values()))
            global_min, global_max = all_values.min(), all_values.max()
            print(f"[sum_attention_by_class] global_min: {global_min:.3f}, global_max: {global_max:.3f}")
            if global_max > global_min:
                for class_name in raw_profiles:
                    raw_profiles[class_name] = (
                        (raw_profiles[class_name] - global_min) / (global_max - global_min)
                    )
                    print(f"[sum_attention_by_class] [global-norm] class: {class_name}, "
                          f"raw_profiles[{class_name}]: {raw_profiles[class_name].shape}, "
                          f"min: {np.min(raw_profiles[class_name]):.3f}, max: {np.max(raw_profiles[class_name]):.3f}")
        elif normalize == 'class':
            for class_name in raw_profiles:
                class_values = raw_profiles[class_name]
                class_min, class_max = class_values.min(), class_values.max()
                raw_profiles[class_name] = (
                        (raw_profiles[class_name] - class_min) / (class_max - class_min)
                )
                print(f"[sum_attention_by_class] [class-norm] class: {class_name}, "
                      f"raw_profiles[{class_name}]: {raw_profiles[class_name].shape}, "
                      f"min: {np.min(raw_profiles[class_name]):.3f}, max: {np.max(raw_profiles[class_name]):.3f}")

    # Step 3: save profiles
    for class_name, profile in raw_profiles.items():
        out_path = os.path.join(output_dir, f"{prefix}_{class_name}_sum.npy")
        np.save(out_path, profile)
        print(f"[INFO] Saved summed attention for {class_name} -> {out_path}")

    return raw_profiles



def plot_multi_class_profiles(class_profiles, output_dir, prefix="class"):
    """
    Plot multiple summed attention profiles (subplot and scatter).
    """
    if not class_profiles:
        print("[WARN] No class profiles to plot.")
        return

    classes_sorted = sorted(class_profiles.keys())
    num_classes = len(classes_sorted)

    # --------- Subplot figure ---------
    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 3 * num_classes), sharex=True)
    if num_classes == 1:
        axes = [axes]  # ensure iterable

    for ax, class_name in zip(axes, classes_sorted):
        ax.plot(class_profiles[class_name])
        ax.set_title(f"Normalized Aggregate Attention Scores ({class_name})")
        ax.set_ylabel("Attention Score")

    axes[-1].set_xlabel("Genomic Position")
    plt.tight_layout()
    multi_path = os.path.join(output_dir, f"{prefix}_subplots.png")
    plt.savefig(multi_path)
    plt.close()
    print(f"[INFO] Saved multi-class subplot figure -> {multi_path}")

    # --------- Scatter plot ---------
    plt.figure(figsize=(12, 6))
    for class_name, profile in class_profiles.items():
        x = np.arange(len(profile))
        plt.scatter(x, profile, s=5, alpha=0.6, label=class_name)

    plt.title("Normalized Aggregate Attention Scores (All Classes)")
    plt.xlabel("Genomic Position")
    plt.ylabel("Attention Score")
    plt.legend()
    plt.tight_layout()

    scatter_path = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"[INFO] Saved scatter plot across classes -> {scatter_path}")
