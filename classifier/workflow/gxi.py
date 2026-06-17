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
from classifier.data import DengueDataset
from classifier.utils_gxi import gradient_x_input

PROGRESS_EVERY = 1
GXI_TARGET_MODE = 'true'      # 'true' for class-specific biological profiles; 'pred' for model-decision profiles
GXI_SCORE_MODE = 'signed'     # 'signed', 'abs', or 'positive'
SAVE_GXI_FOR = 'test'


def parse_args():
    return parse_run_args(get_args, allow_attn=True)


def load_fold_checkpoint(model, fold_model_dir):
    latest_model = get_latest_model_path(fold_model_dir)
    if latest_model is None:
        raise FileNotFoundError(f'No checkpoint found in {fold_model_dir}')
    print(f'Loading checkpoint: {latest_model}')
    model.load_state_dict(torch.load(latest_model, map_location=config.DEVICE))
    model.eval()
    return latest_model


def format_seconds(seconds):
    seconds = float(max(seconds, 0.0))
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m {int(sec)}s'
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h {int(minutes)}m {int(sec)}s'


def run_gxi_on_loader(model, loader, subset, out_dir, gxi_dir, save_gxi_for_subset=False):
    all_labels = []
    all_preds = []
    all_probs = []
    gxi_scores = []
    gxi_labels = []
    gxi_targets = []
    gxi_preds = []
    gxi_batch_idx = []
    gxi_sample_idx = []
    gxi_valid_masks = []
    total_batches = len(loader)
    subset_start = time.time()

    for bidx, batch in enumerate(loader):
        iter_start = time.time()
        inputs, labels = batch
        inputs = inputs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        if save_gxi_for_subset:
            scores, targets, preds = gradient_x_input(
                model=model,
                inputs=inputs,
                labels=labels,
                target_mode=GXI_TARGET_MODE,
                score_mode=GXI_SCORE_MODE,
            )
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = torch.softmax(outputs, dim=1)

            scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
            labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            targets_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
            preds_np = preds.detach().cpu().numpy().astype(np.int64, copy=False)
            gxi_scores.append(scores_np)
            gxi_labels.append(labels_np)
            gxi_targets.append(targets_np)
            gxi_preds.append(preds_np)
            gxi_batch_idx.append(np.full(scores_np.shape[0], bidx, dtype=np.int64))
            gxi_sample_idx.append(np.arange(scores_np.shape[0], dtype=np.int64))
            valid_mask_np = (inputs.detach().cpu().numpy() != 0).any(axis=-1)
            gxi_valid_masks.append(valid_mask_np.astype(bool, copy=False))
        else:
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        done = bidx + 1
        elapsed_total = time.time() - subset_start
        avg_iter = elapsed_total / done
        eta = avg_iter * (total_batches - done)
        if done == 1 or done == total_batches or done % PROGRESS_EVERY == 0:
            print(
                f'[{subset}] [{done} / {total_batches}] '
                f'iter: {time.time() - iter_start:.2f} sec | avg: {avg_iter:.2f} sec | '
                f'elapsed: {format_seconds(elapsed_total)} | eta: {format_seconds(eta)}'
            )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    npz_path = os.path.join(out_dir, f'predictions_{subset}.npz')
    np.savez(npz_path, labels=all_labels, preds=all_preds, probs=all_probs)

    pred_df = pd.DataFrame({'y_true': all_labels, 'y_pred': all_preds})
    for cls_idx in range(all_probs.shape[1]):
        pred_df[f'prob_{cls_idx}'] = all_probs[:, cls_idx]
    csv_path = os.path.join(out_dir, f'predictions_{subset}.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f'Saved predictions to {csv_path}')

    if save_gxi_for_subset and gxi_scores:
        os.makedirs(gxi_dir, exist_ok=True)
        scores = np.concatenate(gxi_scores, axis=0)
        labels_np = np.concatenate(gxi_labels, axis=0)
        targets_np = np.concatenate(gxi_targets, axis=0)
        preds_np = np.concatenate(gxi_preds, axis=0)
        batch_idx = np.concatenate(gxi_batch_idx, axis=0)
        sample_idx = np.concatenate(gxi_sample_idx, axis=0)
        valid_mask = np.concatenate(gxi_valid_masks, axis=0) if gxi_valid_masks else None
        files = np.asarray([
            f'{subset}_batch{int(b)}_sample{int(s)}_class{int(c)}_target{int(t)}_pred{int(p)}_gxi.npy'
            for b, s, c, t, p in zip(batch_idx, sample_idx, labels_np, targets_np, preds_np)
        ])
        gxi_npz_path = os.path.join(gxi_dir, f'gxi_{subset}.npz')
        np.savez_compressed(
            gxi_npz_path,
            scores=scores,
            labels=labels_np,
            targets=targets_np,
            preds=preds_np,
            batch_idx=batch_idx,
            sample_idx=sample_idx,
            files=files,
            subset=np.asarray(subset),
            score_type=np.asarray('gxi'),
            target_mode=np.asarray(GXI_TARGET_MODE),
            score_mode=np.asarray(GXI_SCORE_MODE),
            valid_mask=valid_mask,
        )
        print(f'Saved split-level GxI archive to {gxi_npz_path}')


def gxi_single_fold(args, fold, samples, targets, emb_dim, k_type, base_model_dir, base_metrics_dir):
    fold_id_safe = safe_name(fold['fold_id'])
    fold_dir_name = f'split_{fold_id_safe}'
    fold_model_dir = os.path.join(base_model_dir, fold_dir_name)
    fold_metrics_dir = os.path.join(base_metrics_dir, fold_dir_name)
    fold_gxi_dir = os.path.join(base_model_dir, 'gxi', fold_dir_name)

    os.makedirs(fold_metrics_dir, exist_ok=True)
    os.makedirs(fold_gxi_dir, exist_ok=True)

    print(f'\n==== GxI | FOLD {fold["fold_id"]} ====')
    print(f'Model type: {args.model_type} | loading checkpoints from denformer_{args.pooling}')
    print(f'GxI target_mode={GXI_TARGET_MODE} | score_mode={GXI_SCORE_MODE} | subset={SAVE_GXI_FOR}')

    split_map = {'train': fold['train_idx'], 'val': fold['val_idx'], 'test': fold['test_idx']}
    datasets = {}
    loaders = {}
    for subset, indices in split_map.items():
        if len(indices) == 0:
            print(f'Skipping {subset}: empty split')
            continue
        datasets[subset] = DengueDataset(samples, targets, indices=indices)
        loaders[subset] = DataLoader(
            datasets[subset],
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        print(f'{subset} len: {len(datasets[subset])}')

    collect_shapes(datasets=list(datasets.values()), names=list(datasets.keys()), output_dir=fold_metrics_dir)

    model = build_classifier_model(args, emb_dim, config=config, device=config.DEVICE, attn=True)
    checkpoint_path = load_fold_checkpoint(model, fold_model_dir)

    for subset, loader in loaders.items():
        print(f'\n==== INFERENCE/GxI ON {subset.upper()} SET ====')
        run_gxi_on_loader(
            model=model,
            loader=loader,
            subset=subset,
            out_dir=fold_metrics_dir,
            gxi_dir=fold_gxi_dir,
            save_gxi_for_subset=(subset == SAVE_GXI_FOR),
        )

    return {
        'fold': fold['fold_id'],
        'model_type': args.model_type,
        'pooling': args.pooling,
        'checkpoint_path': checkpoint_path,
        'gxi_target_mode': GXI_TARGET_MODE,
        'gxi_score_mode': GXI_SCORE_MODE,
        'test_size': int(len(fold['test_idx'])),
    }


def main():
    args = parse_args()
    print_args(args)
    if not hasattr(args, 'split_file') or args.split_file is None:
        raise ValueError('Missing --split_file')

    k_type, emb_dim = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)
    model_dir = build_model_dir(paths.logs_dir, 'denformer', args.pooling, k_type, run_suffix, args.epochs)
    metrics_dir = os.path.join(model_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    print(f'Model dir: {model_dir}')
    samples, targets = get_dataset(paths.embeddings_dir, k_type)
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))
    print(f'Loaded {len(folds)} fold(s) from {args.split_file}')

    results = []
    for fold in folds:
        results.append(gxi_single_fold(args, fold, samples, targets, emb_dim, k_type, model_dir, metrics_dir))

    summary_path = os.path.join(metrics_dir, 'gxi_summary.json')
    pd.DataFrame(results).to_json(summary_path, orient='records', indent=4)
    print(f'\nSaved GxI summary to {summary_path}')


if __name__ == '__main__':
    main()
