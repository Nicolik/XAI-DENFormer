# python .\classifier\workflow\run_inference.py --split_file path/to/splits.csv --model_type denformer --epochs 1 --run_name continent --one-hot
import json
import os
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    save_subset_reports,
    plot_confusion_matrices_hconcat
)
from classifier.utils import get_args, print_args
from classifier.utils_data import get_dataset, collect_shapes
from classifier.data import DengueDataset

RUN_MODEL_INFERENCE = True
RUN_REPORTS = True
OVERWRITE_REPORTS = True
EVAL_SUBSETS = ['train', 'val', 'test']

# Timing warm-up.
# This is excluded from measured inference time and reduces the CUDA/cuDNN
# first-pass overhead that can make the first evaluated model look slower.
ENABLE_TIMING_WARMUP = True
WARMUP_BATCHES = 10


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


def prediction_path(fold_metrics_dir, subset):
    return os.path.join(fold_metrics_dir, f'predictions_{subset}.npz')


def predictions_exist(fold_metrics_dir, subsets):
    return all(os.path.exists(prediction_path(fold_metrics_dir, subset)) for subset in subsets)


def sync_if_cuda():
    device = str(config.DEVICE)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()


def get_device_info():
    info = {
        "device": str(config.DEVICE),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if torch.cuda.is_available() and str(config.DEVICE).startswith("cuda"):
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "gpu_index": int(idx),
            "gpu_name": torch.cuda.get_device_name(idx),
            "gpu_total_memory_gb": float(props.total_memory / (1024 ** 3)),
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "gpu_multiprocessor_count": int(props.multi_processor_count),
        })

    return info


def run_timing_warmup(model, loader, warmup_batches=WARMUP_BATCHES):
    if not ENABLE_TIMING_WARMUP or warmup_batches <= 0 or len(loader) == 0:
        return {
            "warmup_enabled": bool(ENABLE_TIMING_WARMUP),
            "warmup_batches_requested": int(warmup_batches),
            "warmup_batches_completed": 0,
            "warmup_time_sec": 0.0,
        }

    completed = 0
    print(f'Warm-up: running up to {warmup_batches} batch(es), excluded from timing')

    sync_if_cuda()
    warmup_start = time.perf_counter()

    with torch.no_grad():
        for bidx, (inputs, _) in enumerate(loader):
            if bidx >= warmup_batches:
                break

            inputs = inputs.to(config.DEVICE)
            _ = model(inputs)
            completed += 1

    sync_if_cuda()
    warmup_elapsed = time.perf_counter() - warmup_start
    print(f'Warm-up completed: {completed} batch(es) in {warmup_elapsed:.2f} sec')

    return {
        "warmup_enabled": True,
        "warmup_batches_requested": int(warmup_batches),
        "warmup_batches_completed": int(completed),
        "warmup_time_sec": float(warmup_elapsed),
    }


def run_inference_on_loader(model, loader):
    all_labels = []
    all_preds = []
    all_probs = []
    batch_times = []

    warmup_info = run_timing_warmup(model, loader)

    sync_if_cuda()
    total_start = time.perf_counter()

    with torch.no_grad():
        for bidx, (inputs, labels) in enumerate(loader):
            sync_if_cuda()
            batch_start = time.perf_counter()

            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            sync_if_cuda()
            elapsed = time.perf_counter() - batch_start
            batch_times.append(elapsed)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            eta = elapsed * (len(loader) - bidx - 1)
            print(f'[{bidx + 1} / {len(loader)}] iter: {elapsed:.2f} sec | eta: {eta:.2f} sec')

    sync_if_cuda()
    total_elapsed = time.perf_counter() - total_start

    n_samples = len(all_labels)
    timing = {
        "num_samples": int(n_samples),
        "num_batches": int(len(loader)),
        "batch_size": int(loader.batch_size) if loader.batch_size is not None else None,
        "total_time_sec": float(total_elapsed),
        "mean_batch_time_sec": float(np.mean(batch_times)) if batch_times else None,
        "std_batch_time_sec": float(np.std(batch_times)) if batch_times else None,
        "samples_per_sec": float(n_samples / total_elapsed) if total_elapsed > 0 else None,
        "timing_mode": "post_warmup_full_loader_with_cuda_synchronize",
    }
    timing.update(warmup_info)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), timing


def save_predictions(out_dir, subset, indices, y_true, y_pred, y_prob):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = prediction_path(out_dir, subset)
    np.savez(npz_path, indices=np.array(indices), labels=y_true, preds=y_pred, probs=y_prob)
    print(f'Saved arrays to {npz_path}')

    pred_df = pd.DataFrame({'index': indices, 'y_true': y_true, 'y_pred': y_pred})
    for cls_idx in range(y_prob.shape[1]):
        pred_df[f'prob_{cls_idx}'] = y_prob[:, cls_idx]

    csv_path = os.path.join(out_dir, f'predictions_{subset}.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f'Saved table to {csv_path}')


def load_predictions(fold_metrics_dir, subset):
    path = prediction_path(fold_metrics_dir, subset)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing cached prediction file: {path}')
    data = np.load(path)
    indices = data['indices'] if 'indices' in data.files else None
    return indices, data['labels'], data['preds'], data['probs']


def build_datasets_and_loaders(fold, samples, targets):
    split_map = {'train': fold['train_idx'], 'val': fold['val_idx'], 'test': fold['test_idx']}
    datasets = {}
    loaders = {}
    for subset, indices in split_map.items():
        if len(indices) == 0:
            print(f'Skipping {subset}: empty split')
            continue
        datasets[subset] = DengueDataset(samples, targets, indices=indices)
        loaders[subset] = DataLoader(datasets[subset], batch_size=config.BATCH_SIZE, shuffle=False)
        print(f'{subset} len: {len(datasets[subset])}')
    return split_map, datasets, loaders


def inference_single_fold(args, fold, samples, targets, emb_dim, k_type, base_model_dir, base_metrics_dir):
    fold_id = fold['fold_id']
    fold_id_safe = safe_name(fold_id)
    fold_dir_name = f'split_{fold_id_safe}'
    fold_model_dir = os.path.join(base_model_dir, fold_dir_name)
    fold_metrics_dir = os.path.join(base_metrics_dir, fold_dir_name)
    os.makedirs(fold_metrics_dir, exist_ok=True)

    print(f'\n==== FOLD {fold_id} ====')
    print(f'Model type: {args.model_type} | pooling: {args.pooling}')

    split_map, datasets, loaders = build_datasets_and_loaders(fold, samples, targets)
    if datasets:
        collect_shapes(datasets=list(datasets.values()), names=list(datasets.keys()), output_dir=fold_metrics_dir)

    available_subsets = list(loaders.keys())
    fold_result = {
        'fold': fold_id,
        'model_type': args.model_type,
        'pooling': args.pooling,
        'train_size': int(len(fold['train_idx'])),
        'val_size': int(len(fold['val_idx'])),
        'test_size': int(len(fold['test_idx'])),
        'device_info': get_device_info(),
    }

    if RUN_MODEL_INFERENCE:
        model = build_classifier_model(args, emb_dim, config=config, device=config.DEVICE, attn=False)
        checkpoint_path = load_fold_checkpoint(model, fold_model_dir)
        fold_result['checkpoint_path'] = checkpoint_path

        for subset, loader in loaders.items():
            print(f'\n==== MODEL INFERENCE ON {subset.upper()} SET ====')
            y_true, y_pred, y_prob, timing = run_inference_on_loader(model, loader)
            fold_result[f'{subset}_timing'] = timing
            save_predictions(fold_metrics_dir, subset, split_map[subset], y_true, y_pred, y_prob)
    else:
        if not predictions_exist(fold_metrics_dir, available_subsets):
            missing = [prediction_path(fold_metrics_dir, subset) for subset in available_subsets if not os.path.exists(prediction_path(fold_metrics_dir, subset))]
            raise FileNotFoundError('RUN_MODEL_INFERENCE is False, but cached predictions are missing:\n' + '\n'.join(missing))
        print('Using cached predictions. Skipping model inference.')

    test_confmat = None
    if RUN_REPORTS:
        for subset in available_subsets:
            print(f'\n==== REPORTS FOR {subset.upper()} SET ====')
            _, y_true, y_pred, _ = load_predictions(fold_metrics_dir, subset)
            subset_result, cm = save_subset_reports(fold_metrics_dir, fold_id, subset, y_true, y_pred, config.NUM_CLASSES)
            fold_result[subset] = subset_result
            if subset == 'test':
                test_confmat = cm
                print(f"Test accuracy: {subset_result['accuracy']:.4f}")
            else:
                print(f"{subset} accuracy: {subset_result['accuracy']:.4f}")

        metrics_path = os.path.join(fold_metrics_dir, 'inference_metrics.json')
        if OVERWRITE_REPORTS or not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                json.dump(fold_result, f, indent=4)
            print(f'Saved fold inference metrics to {metrics_path}')

    return fold_result, test_confmat


def main():
    args = parse_args()
    print_args(args)
    print(f'Model type: {args.model_type}')
    print(f'Pooling strategy: {args.pooling}')

    if not hasattr(args, 'split_file') or args.split_file is None:
        raise ValueError('Missing --split_file')

    k_type, emb_dim = resolve_k_type_and_emb_dim(args, config.EMB_DIM, config.EMB_DIM_OHE)
    run_suffix = get_run_suffix(args, args.split_file)
    model_dir = build_model_dir(paths.logs_dir, args.model_type, args.pooling, k_type, run_suffix, args.epochs)
    metrics_dir = os.path.join(model_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    print(f'Model dir: {model_dir}')
    print(f'Metrics dir: {metrics_dir}')
    print(f'RUN_MODEL_INFERENCE: {RUN_MODEL_INFERENCE}')
    print(f'RUN_REPORTS: {RUN_REPORTS}')

    print('\n==== LETTURA FILE ====')
    start_time = time.time()
    samples, targets = get_dataset(paths.embeddings_dir, k_type)
    print(f'Dataset read in {time.time() - start_time:.2f} seconds')

    print('\n==== LETTURA SPLIT PRECOMPUTATI ====')
    folds = load_and_validate_folds(args.split_file, len(samples), getattr(args, 'fold', None))
    print(f'Loaded {len(folds)} fold(s) from {args.split_file}')

    all_results = []
    test_confmats = []
    test_titles = []
    test_accuracies = []

    for fold in folds:
        result, test_cm = inference_single_fold(args, fold, samples, targets, emb_dim, k_type, model_dir, metrics_dir)
        all_results.append(result)
        if test_cm is not None:
            test_confmats.append(test_cm)
            test_titles.append(str(result['fold']))
            test_accuracies.append(result['test']['accuracy'])

    summary_path = os.path.join(metrics_dir, 'inference_summary.json')
    if RUN_REPORTS and (OVERWRITE_REPORTS or not os.path.exists(summary_path)):
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f'\nSaved inference summary to {summary_path}')

    if RUN_REPORTS and test_confmats:
        for normalize, suffix in [('none', 'counts'), ('row', 'row_normalized'), ('all', 'all_normalized')]:
            path = os.path.join(metrics_dir, f'confusion_matrices_test_hconcat_{suffix}.png')
            plot_confusion_matrices_hconcat(test_confmats, test_titles, path, num_classes=config.NUM_CLASSES, normalize=normalize, accuracies=test_accuracies)
            print(f'Saved hconcat {suffix} confusion matrices to {path}')


if __name__ == '__main__':
    main()
