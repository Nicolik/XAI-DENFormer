# python .\classifier\workflow\run_train_from_split.py --split_file path/to/splits.csv --model_type denformer --epochs 1 --run_name continent --one-hot
import json
import os
import sys
import time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import paths
from classifier.workflow import config
from classifier.workflow.utils import (
    build_classifier_model,
    build_model_dir,
    get_run_suffix,
    load_and_validate_folds,
    parse_run_args,
    resolve_k_type_and_emb_dim,
    safe_name,
)
from classifier.utils_train import train_one_epoch
from classifier.utils import get_args, print_args, plot_training_curves
from classifier.utils_data import get_dataset
from classifier.data import DengueDataset, build_2d_weighted_dataloader, get_targets_dataset


def parse_args():
    return parse_run_args(get_args, allow_attn=False)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    labels = []
    preds = []
    probs_all = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, batch_labels = data
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, batch_labels)
            running_loss += loss.cpu().item()

            probs = torch.softmax(outputs, dim=1)
            batch_preds = torch.argmax(probs, dim=1)

            labels.extend(batch_labels.cpu().numpy())
            preds.extend(batch_preds.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    avg_loss = running_loss / (i + 1)
    acc = accuracy_score(labels, preds)
    return avg_loss, acc, np.array(labels), np.array(preds), np.array(probs_all)


def train_single_fold(args, fold, samples, targets, emb_dim, k_type, base_model_dir, base_metrics_dir, device):
    fold_id_safe = safe_name(fold['fold_id'])
    fold_dir_name = f'split_{fold_id_safe}'
    fold_model_dir = os.path.join(base_model_dir, fold_dir_name)
    fold_metrics_dir = os.path.join(base_metrics_dir, fold_dir_name)
    os.makedirs(fold_model_dir, exist_ok=True)
    os.makedirs(fold_metrics_dir, exist_ok=True)

    train_idx = fold['train_idx']
    val_idx = fold['val_idx']
    test_idx = fold['test_idx']

    print(f"\n==== FOLD {fold['fold_id']} ====")
    print(f"Model type: {args.model_type} | pooling: {args.pooling}")
    print(f"Train Indices: {train_idx.shape}, Validation Indices: {val_idx.shape}, Test Indices: {test_idx.shape}")
    print(f"Samples: {samples.shape}, Targets: {targets.shape}")

    print("\n==== CREAZIONE DATASET E DATALOADER ====")
    train_dataset = DengueDataset(samples, targets, indices=train_idx)
    val_dataset = DengueDataset(samples, targets, indices=val_idx)
    test_dataset = DengueDataset(samples, targets, indices=test_idx) if len(test_idx) else None
    print(f"Train len: {len(train_dataset)}, Validation len: {len(val_dataset)}, Test len: {len(test_dataset) if test_dataset else 0}")

    print(f"Train Targets: {Counter(get_targets_dataset(train_dataset))}")
    print(f"Val   Targets: {Counter(get_targets_dataset(val_dataset))}")
    if test_dataset is not None:
        print(f"Test  Targets: {Counter(get_targets_dataset(test_dataset))}")

    train_loader = build_2d_weighted_dataloader(train_dataset, batch_size=config.BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False) if test_dataset else None

    print(f"Creating model: {args.model_type}")
    model = build_classifier_model(args, emb_dim, config=config, device=device, attn=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter(os.path.join(fold_model_dir, 'runs', f'denv_trainer_{timestamp}'))

    best_vloss = float('inf')
    best_model_path = None
    train_stats = []

    print("Starting training!")
    for epoch in range(args.epochs):
        epoch_number = epoch + 1
        print(f'EPOCH {epoch_number}:')

        model.train(True)
        all_train_labels, all_train_preds, avg_loss = train_one_epoch(
            model, train_loader, epoch, writer, device, optimizer, loss_fn
        )
        train_acc = accuracy_score(all_train_labels, all_train_preds)

        avg_vloss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn, device)

        train_stats.append({
            'fold': fold['fold_id'],
            'model_type': args.model_type,
            'pooling': args.pooling,
            'epoch': epoch_number,
            'train_loss': float(avg_loss),
            'train_acc': float(train_acc),
            'val_loss': float(avg_vloss),
            'val_acc': float(val_acc),
        })

        print(f"LOSS train {avg_loss:.4f} valid {avg_vloss:.4f} | ACC train {train_acc:.4f} valid {val_acc:.4f}")
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training': avg_loss, 'Validation': avg_vloss},
            epoch_number,
        )
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            c_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            best_model_path = os.path.join(
                fold_model_dir,
                f'model-{k_type}_fold-{fold_id_safe}_epoch-{epoch_number}_{c_timestamp}.pt',
            )
            torch.save(model.state_dict(), best_model_path)

    json_path = os.path.join(fold_metrics_dir, 'train_trend_stats.json')
    with open(json_path, 'w') as f:
        json.dump(train_stats, f, indent=4)
    print(f"Saved training stats to {json_path}")
    plot_training_curves(train_stats, fold_metrics_dir)

    result = {
        'fold': fold['fold_id'],
        'model_type': args.model_type,
        'pooling': args.pooling,
        'best_val_loss': float(best_vloss),
        'best_model_path': best_model_path,
        'train_size': int(len(train_idx)),
        'val_size': int(len(val_idx)),
        'test_size': int(len(test_idx)),
        'last_val_acc': float(train_stats[-1]['val_acc']),
    }

    if test_loader and best_model_path:
        print("\n==== TEST BEST MODEL ====")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, loss_fn, device)
        result.update({
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        })
        print(f"LOSS test {test_loss:.4f} | ACC test {test_acc:.4f}")

        test_metrics_path = os.path.join(fold_metrics_dir, 'test_metrics.json')
        with open(test_metrics_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Saved test metrics to {test_metrics_path}")

        if getattr(args, 'save_test_predictions', False):
            pred_df = pd.DataFrame({
                'index': test_idx,
                'y_true': y_true,
                'y_pred': y_pred,
            })
            for cls_idx in range(y_prob.shape[1]):
                pred_df[f'prob_{cls_idx}'] = y_prob[:, cls_idx]
            pred_path = os.path.join(fold_metrics_dir, 'test_predictions.csv')
            pred_df.to_csv(pred_path, index=False)
            print(f"Saved test predictions to {pred_path}")

    writer.close()
    return result


def main():
    args = parse_args()
    print_args(args)
    print(f"Model type: {args.model_type}")
    print(f"Pooling strategy: {args.pooling}")

    if not hasattr(args, 'split_file') or args.split_file is None:
        raise ValueError('Missing --split_file')

    torch.manual_seed(0)
    device = config.DEVICE

    k_type, emb_dim = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)
    model_dir = build_model_dir(paths.logs_dir, args.model_type, args.pooling, k_type, run_suffix, args.epochs)
    metrics_dir = os.path.join(model_dir, 'metrics')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Model dir: {model_dir}")
    print(f"Metrics dir: {metrics_dir}")

    print("==== LETTURA FILE ====")
    start_time = time.time()
    samples, targets = get_dataset(paths.embeddings_dir, k_type)
    print(f"Dataset read in {time.time() - start_time:.2f} seconds")

    print("\n==== LETTURA SPLIT PRECOMPUTATI ====")
    start_time = time.time()
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))
    print(f"Loaded {len(folds)} fold(s) from {args.split_file} in {time.time() - start_time:.2f} seconds")

    all_results = []
    for fold in folds:
        all_results.append(train_single_fold(args, fold, samples, targets, emb_dim, k_type, model_dir, metrics_dir, device))

    summary_path = os.path.join(metrics_dir, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved CV/holdout summary to {summary_path}")


if __name__ == '__main__':
    main()
