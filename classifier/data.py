import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter


# Creating a Custom Dataset
class DengueDataset(Dataset):
    def __init__(self, samples, targets, indices=None):
        self.samples = samples
        self.targets = targets
        self.indices = indices if indices is not None else np.arange(len(samples))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_idx = self.indices[index]
        x = self.samples[real_idx]
        y = self.targets[real_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def get_targets_dataset(dataset, use_subset_targets=True):
    if use_subset_targets:
        local_to_global = np.array(dataset.indices, dtype=np.int64)
        targets = dataset.targets[local_to_global].tolist()
    else:
        targets = dataset.targets.tolist()
    return targets


def build_2d_weighted_dataloader(dataset, batch_size, use_subset_targets=True):
    targets = get_targets_dataset(dataset, use_subset_targets)
    cnt_targets = Counter(targets)
    unique_targets = set(targets)
    print(f"Counter Targets: {cnt_targets}")

    num_elements = len(dataset)
    class_weights = []
    for t in unique_targets:
        val = 1 - cnt_targets[t] / len(dataset)
        class_weights.append(val)

    print(f"Weights: {class_weights}")
    element_weights = []

    for label in targets:
        element_weights.append(class_weights[label])

    element_weights = torch.Tensor(element_weights)
    sampler = WeightedRandomSampler(element_weights, num_elements, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=sampler)
    return dataloader
