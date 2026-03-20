# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/data/compound_dataset.py

import torch
import numpy as np


class CompoundDataset(torch.utils.data.Dataset):

    def __init__(self, datasets, common_keys=None):
        self.datasets = datasets
        self.dataset_lens = [len(ds) for ds in self.datasets]
        self.dataset_start_indices = [0] + list(np.cumsum(self.dataset_lens))[:-1]
        self.common_keys = common_keys

    @property
    def name(self):
        return "+".join([dataset.name for dataset in self.datasets])

    def full_name(self):
        return "+".join([dataset.full_name for dataset in self.datasets])

    def __str__(self):
        return self.name()

    def __getitem__(self, index):
        for dataset_idx, dataset_start in enumerate(self.dataset_start_indices):
            if (dataset_idx == len(self.datasets)-1) or (self.dataset_start_indices[dataset_idx+1] > index):
                sample = self.datasets[dataset_idx][index - dataset_start]
                break

        if self.common_keys is not None:
            sample = {k: sample[k] for k in self.common_keys}

        return sample

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    @classmethod
    def init_as_loader(cls, datasets, common_keys=None, batch_size=1, shuffle=False, num_workers=0, collate_fn=None,
                       pin_memory=False, drop_last=False, worker_init_fn=None, indices=None):
        compound_dataset = cls(datasets, common_keys)
        return compound_dataset.get_loader(batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                           num_workers=num_workers, collate_fn=collate_fn,
                                           drop_last=drop_last, worker_init_fn=worker_init_fn, indices=indices)

    def get_loader(self, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False,
                   worker_init_fn=None, indices=None):

        for dataset in self.datasets:
            dataset.to_torch = False

        compound_dataset = torch.utils.data.Subset(self, indices) if indices is not None else self

        return torch.utils.data.DataLoader(compound_dataset, batch_size=batch_size, shuffle=shuffle,
                                           pin_memory=pin_memory, num_workers=num_workers, collate_fn=collate_fn,
                                           drop_last=drop_last, worker_init_fn=worker_init_fn)
