import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import paths
from classifier.data import DengueDataset
from classifier.workflow import config
from classifier.workflow.utils import (
    build_classifier_model,
    build_model_dir,
    build_xai_embeddings_data_dir,
    get_latest_model_path,
    get_run_suffix,
    load_and_validate_folds,
    parse_run_args,
    resolve_k_type_and_emb_dim,
    safe_name,
)
from classifier.utils import get_args, print_args
from classifier.utils_data import get_dataset


def parse_args():
    return parse_run_args(get_args, allow_attn=False)


def load_fold_checkpoint(model, fold_model_dir):
    latest_model = get_latest_model_path(fold_model_dir)
    if latest_model is None:
        raise FileNotFoundError(f'No checkpoint found in {fold_model_dir}')
    print(f'Loading checkpoint: {latest_model}')
    model.load_state_dict(torch.load(latest_model, map_location=config.DEVICE))
    model.eval()
    return latest_model


def extract_embeddings(model, loader):
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for bidx, (batch_x, batch_y) in enumerate(loader):
            start = time.time()
            batch_x = batch_x.to(config.DEVICE)
            emb = model.get_embedding(batch_x)
            all_embeddings.append(emb.detach().cpu())
            all_labels.append(batch_y.detach().cpu())
            elapsed = time.time() - start
            eta = elapsed * (len(loader) - bidx - 1)
            print(f'[{bidx + 1} / {len(loader)}] iter: {elapsed:.2f} sec | eta: {eta:.2f} sec')

    X = torch.cat(all_embeddings).numpy()
    y = torch.cat(all_labels).numpy()
    return X, y


def run_fold(args, fold, samples, targets, emb_dim, k_type, model_dir, xai_data_root):
    fold_id_safe = safe_name(fold['fold_id'])
    fold_dir_name = f'split_{fold_id_safe}'
    fold_model_dir = os.path.join(model_dir, fold_dir_name)
    xai_dir = os.path.join(xai_data_root, fold_dir_name)
    os.makedirs(xai_dir, exist_ok=True)

    print(f'\n==== EMBEDDINGS | FOLD {fold["fold_id"]} ====')
    model = build_classifier_model(args, emb_dim, config=config, device=config.DEVICE, attn=False)
    checkpoint_path = load_fold_checkpoint(model, fold_model_dir)

    split_map = {'train': fold['train_idx'], 'test': fold['test_idx']}
    if len(fold['test_idx']) == 0:
        split_map = {'train': fold['train_idx'], 'val': fold['val_idx']}

    rows = []
    for subset, indices in split_map.items():
        if len(indices) == 0:
            print(f'Skipping {subset}: empty split')
            continue
        dataset = DengueDataset(samples, targets, indices=indices)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        print(f'\n==== EXTRACT {subset.upper()} EMBEDDINGS | n={len(dataset)} ====')
        X, y = extract_embeddings(model, loader)
        out_path = os.path.join(xai_dir, f'embeddings_{subset}.npz')
        np.savez(out_path, embeddings=X, labels=y, indices=np.asarray(indices, dtype=np.int64))
        print(f'Saved {X.shape}, {y.shape} -> {out_path}')
        rows.append({'fold': fold['fold_id'], 'subset': subset, 'n': len(y), 'dim': X.shape[1], 'path': out_path})

    return {'fold': fold['fold_id'], 'checkpoint_path': checkpoint_path, 'xai_dir': xai_dir, 'rows': rows}


def main():
    args = parse_args()
    print_args(args)

    k_type, emb_dim = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)
    model_dir = build_model_dir(paths.logs_dir, args.model_type, args.pooling, k_type, run_suffix, args.epochs)

    xai_data_root = build_xai_embeddings_data_dir(paths.logs_dir, args.model_type, args.pooling, k_type, run_suffix, args.epochs)

    print(f'Model dir: {model_dir}')
    print(f'XAI embedding data dir: {xai_data_root}')
    samples, targets = get_dataset(paths.embeddings_dir, k_type)
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))

    summary_rows = []
    for fold in folds:
        result = run_fold(args, fold, samples, targets, emb_dim, k_type, model_dir, xai_data_root)
        summary_rows.extend(result['rows'])

    summary_path = os.path.join(xai_data_root, 'embeddings_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')


if __name__ == '__main__':
    main()
