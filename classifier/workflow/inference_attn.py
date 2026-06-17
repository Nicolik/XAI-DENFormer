# python .\classifier\workflow\scripts\03_run_inference_attn.py --split_file path/to/splits.csv --epochs 1 --model_type denformer_attn --run_name continent --one-hot
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import paths
from classifier.workflow import config
from classifier.workflow.utils import (
    build_classifier_model,
    build_model_dir,
    get_latest_model_path,
    get_run_suffix,
    load_and_validate_folds,
    parse_run_args,
    resolve_k_type_and_emb_dim,
    safe_name,
)
from classifier.utils import get_args, print_args
from classifier.utils_data import get_dataset, collect_shapes
from classifier.utils_attn import extract_attention_from_batch
from classifier.data import DengueDataset


# Keep progress visible, but avoid printing tensor diagnostics at every batch.
PROGRESS_EVERY = 1
DEBUG_ATTENTION_FIRST_BATCH = False

# Fast default: save only reduced .npy profiles during inference.
# Generate PNG/TSV later from saved .npy files if needed.
SAVE_ATTENTION_PLOTS = False
SAVE_ATTENTION_TSV = False

# Attention reduction strategy:
# - "cls": CLS-to-token attention, same intent as the previous implementation.
# - "mean_query": average attention over all query positions.
ATTENTION_REDUCTION = "cls"
ATTENTION_LAYER = "last"


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def load_fold_checkpoint(model, fold_model_dir):
    latest_model = get_latest_model_path(fold_model_dir)
    if latest_model is None:
        raise FileNotFoundError(f"No checkpoint found in {fold_model_dir}")
    print(f"Loading checkpoint: {latest_model}")
    model.load_state_dict(torch.load(latest_model, map_location=config.DEVICE))
    model.eval()
    return latest_model


def format_seconds(seconds):
    seconds = float(max(seconds, 0.0))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(sec)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(sec)}s"


def run_inference_on_loader(model, loader, subset, out_dir, attn_dir, k, save_attention=False):
    all_labels = []
    all_preds = []
    all_probs = []
    attn_scores = []
    attn_labels = []
    attn_batch_idx = []
    attn_sample_idx = []
    attn_valid_masks = []

    total_batches = len(loader)
    subset_start = time.time()

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            iter_start = time.time()
            inputs, labels = batch
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            # Important: when saving attention, do a single forward pass returning both
            # logits and attention. The previous implementation did a second forward
            # inside extract_attention_from_batch, doubling the expensive attention cost.
            if save_attention:
                outputs, attn = model(inputs, return_attn=True)
            else:
                model_out = model(inputs)
                outputs = model_out[0] if isinstance(model_out, tuple) else model_out
                attn = None

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            labels_np = labels.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            all_labels.extend(labels_np)
            all_preds.extend(preds_np)
            all_probs.extend(probs_np)

            if save_attention:
                batch_scores = extract_attention_from_batch(
                    inputs=inputs,
                    attn=attn,
                    labels=labels,
                    out_dir=attn_dir,
                    prefix=f"{subset}_batch{bidx}",
                    k=k,
                    return_output=True,
                    reduction=ATTENTION_REDUCTION,
                    layer=ATTENTION_LAYER,
                    save_plot=False,
                    save_tsv=False,
                    debug=(DEBUG_ATTENTION_FIRST_BATCH and bidx == 0),
                ).astype(np.float32, copy=False)
                attn_scores.append(batch_scores)
                attn_labels.append(labels_np.astype(np.int64, copy=False))
                attn_batch_idx.append(np.full(batch_scores.shape[0], bidx, dtype=np.int64))
                attn_sample_idx.append(np.arange(batch_scores.shape[0], dtype=np.int64))
                valid_mask_np = (inputs.detach().cpu().numpy() != 0).any(axis=-1)
                attn_valid_masks.append(valid_mask_np.astype(bool, copy=False))

            iter_elapsed = time.time() - iter_start
            done = bidx + 1
            elapsed_total = time.time() - subset_start
            avg_iter = elapsed_total / done
            eta = avg_iter * (total_batches - done)

            if done == 1 or done == total_batches or done % PROGRESS_EVERY == 0:
                print(
                    f"[{subset}] [{done} / {total_batches}] "
                    f"iter: {iter_elapsed:.2f} sec | avg: {avg_iter:.2f} sec | "
                    f"elapsed: {format_seconds(elapsed_total)} | eta: {format_seconds(eta)}"
                )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    npz_path = os.path.join(out_dir, f"predictions_{subset}.npz")
    np.savez(npz_path, labels=all_labels, preds=all_preds, probs=all_probs)
    print(f"Saved arrays to {npz_path}")

    pred_df = pd.DataFrame({"y_true": all_labels, "y_pred": all_preds})
    for cls_idx in range(all_probs.shape[1]):
        pred_df[f"prob_{cls_idx}"] = all_probs[:, cls_idx]

    csv_path = os.path.join(out_dir, f"predictions_{subset}.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"Saved table to {csv_path}")

    if save_attention and attn_scores:
        os.makedirs(attn_dir, exist_ok=True)
        scores = np.concatenate(attn_scores, axis=0)
        labels_np = np.concatenate(attn_labels, axis=0)
        batch_idx = np.concatenate(attn_batch_idx, axis=0)
        sample_idx = np.concatenate(attn_sample_idx, axis=0)
        valid_mask = np.concatenate(attn_valid_masks, axis=0) if attn_valid_masks else None
        files = np.asarray([
            f'{subset}_batch{int(b)}_sample{int(s)}_class{int(c)}_attn.npy'
            for b, s, c in zip(batch_idx, sample_idx, labels_np)
        ])
        attn_npz_path = os.path.join(attn_dir, f'attention_{subset}.npz')
        np.savez_compressed(
            attn_npz_path,
            scores=scores,
            labels=labels_np,
            batch_idx=batch_idx,
            sample_idx=sample_idx,
            files=files,
            subset=np.asarray(subset),
            score_type=np.asarray('attention'),
            reduction=np.asarray(ATTENTION_REDUCTION),
            layer=np.asarray(str(ATTENTION_LAYER)),
            valid_mask=valid_mask,
        )
        print(f"Saved split-level attention archive to {attn_npz_path}")


def inference_single_fold(args, fold, samples, targets, emb_dim, k_type, base_model_dir, base_metrics_dir):
    fold_id_safe = safe_name(fold["fold_id"])
    fold_dir_name = f"split_{fold_id_safe}"

    fold_model_dir = os.path.join(base_model_dir, fold_dir_name)
    fold_metrics_dir = os.path.join(base_metrics_dir, fold_dir_name)
    fold_attn_dir = os.path.join(base_model_dir, "attention", fold_dir_name)

    os.makedirs(fold_metrics_dir, exist_ok=True)
    os.makedirs(fold_attn_dir, exist_ok=True)

    print(f"\n==== FOLD {fold['fold_id']} ====")
    print(f"Model type: {args.model_type} | loading checkpoints from denformer_{args.pooling}")

    split_map = {"train": fold["train_idx"], "val": fold["val_idx"], "test": fold["test_idx"]}
    datasets = {}
    loaders = {}

    for subset, indices in split_map.items():
        if len(indices) == 0:
            print(f"Skipping {subset}: empty split")
            continue
        datasets[subset] = DengueDataset(samples, targets, indices=indices)
        loaders[subset] = DataLoader(
            datasets[subset],
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"{subset} len: {len(datasets[subset])}")

    collect_shapes(datasets=list(datasets.values()), names=list(datasets.keys()), output_dir=fold_metrics_dir)

    model = build_classifier_model(args, emb_dim, config=config, device=config.DEVICE, attn=True)
    checkpoint_path = load_fold_checkpoint(model, fold_model_dir)

    save_attention_for = getattr(args, "save_attention_for", "test")
    if save_attention_for is None:
        save_attention_for = "test"

    print(f"Attention will be saved for subset: {save_attention_for}")
    print(f"Attention reduction: layer={ATTENTION_LAYER}, reduction={ATTENTION_REDUCTION}")
    print(f"Attention PNG plots: {SAVE_ATTENTION_PLOTS} | TSV: {SAVE_ATTENTION_TSV}")

    for subset, loader in loaders.items():
        print(f"\n==== INFERENCE ON {subset.upper()} SET ====")
        run_inference_on_loader(
            model=model,
            loader=loader,
            subset=subset,
            out_dir=fold_metrics_dir,
            attn_dir=fold_attn_dir,
            k=args.k,
            save_attention=(subset == save_attention_for),
        )

    return {
        "fold": fold["fold_id"],
        "model_type": args.model_type,
        "pooling": args.pooling,
        "checkpoint_path": checkpoint_path,
        "train_size": int(len(fold["train_idx"])),
        "val_size": int(len(fold["val_idx"])),
        "test_size": int(len(fold["test_idx"])),
    }


def main():
    args = parse_args()
    print_args(args)
    print(f"Model type: {args.model_type}")
    print(f"Pooling strategy: {args.pooling}")

    if not hasattr(args, "split_file") or args.split_file is None:
        raise ValueError("Missing --split_file")

    k_type, emb_dim = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)

    # Attention model shares checkpoints/output root with the trained denformer run.
    model_dir = build_model_dir(paths.logs_dir, "denformer", args.pooling, k_type, run_suffix, args.epochs)
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Model dir: {model_dir}")
    print(f"Metrics dir: {metrics_dir}")

    print("==== LETTURA FILE ====")
    start_time = time.time()
    samples, targets = get_dataset(paths.embeddings_dir, k_type)
    print(f"Dataset read in {time.time() - start_time:.2f} seconds")

    print("\n==== LETTURA SPLIT PRECOMPUTATI ====")
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, "fold", None))
    print(f"Loaded {len(folds)} fold(s) from {args.split_file}")

    results = []
    for fold in folds:
        results.append(inference_single_fold(args, fold, samples, targets, emb_dim, k_type, model_dir, metrics_dir))

    summary_path = os.path.join(metrics_dir, "inference_attn_summary.json")
    pd.DataFrame(results).to_json(summary_path, orient="records", indent=4)
    print(f"\nSaved attention inference summary to {summary_path}")


if __name__ == "__main__":
    main()
