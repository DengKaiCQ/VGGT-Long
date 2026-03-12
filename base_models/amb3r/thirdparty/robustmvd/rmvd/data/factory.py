# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/factory.py

import os.path as osp
from glob import glob

from .registry import get_dataset, has_dataset
from .compound_dataset import CompoundDataset
from .dataset import Dataset


def create_dataset(dataset_name_or_path, dataset_type=None, split=None, **kwargs):
    """Create a dataset by its name or from a dataset config file.

    Args:
        dataset_name_or_path (str): The name of the dataset to create or the path to a dataset config file.
            If it is the name of a dataset, it can optionally contain the dataset_type and split in the format
            base_dataset_name.split.dataset_type.
        dataset_type (str): The type of the dataset to create. Only used when dataset_name_or_path is a dataset name.
            Can optionally be provided within the dataset_name_or_path.
        split (str): The split of the dataset to create. Only used when dataset_name_or_path is a dataset name.
            Can optionally be provided within the dataset_name_or_path.

    Keyword Args:
        **kwargs: Arguments for the dataset.
    """
    if has_dataset(dataset_name=dataset_name_or_path, dataset_type=dataset_type, split=split):
        return _create_dataset_from_registry(dataset_name=dataset_name_or_path, dataset_type=dataset_type,
                                             split=split, **kwargs)
    else:
        if len(kwargs) > 0:
            print(f"Warning: arguments {', '.join(kwargs.keys())} were ignored when creating the dataset {dataset_name_or_path}.")
        return _create_dataset_from_cfg(path=dataset_name_or_path)


def _create_dataset_from_registry(dataset_name, dataset_type, split, **kwargs):
    dataset_cls = get_dataset(dataset_name=dataset_name, dataset_type=dataset_type, split=split)
    dataset = dataset_cls(**kwargs)
    return dataset


def _create_dataset_from_cfg(path):
    if not osp.split(path)[1] == 'dataset.cfg':
        paths = glob(f'{path}/**/dataset.cfg', recursive=True)
        assert len(paths) > 0, f"No dataset.cfg file found in {path} or its subdirectories."
        path = paths[0]
    dataset = Dataset.from_config(path)
    return dataset


def create_dataloader(dataset_name, dataset_type=None, split=None, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=None, pin_memory=False, drop_last=False, worker_init_fn=None, indices=None, **kwargs):
    """Create a dataloader.

    Args:
        dataset_name (str): The name of the dataset to create. Can optionally contain the dataset_type and split in the
            format base_dataset_name.split.dataset_type.
        dataset_type (str): The type of the dataset to create. Can optionally be provided within the dataset_name.
        split (str): The split of the dataset to create. Can optionally be provided within the dataset_name.

    Keyword Args:
        **kwargs: Arguments for the dataset.
    """
    # TODO: take dataset_path_or_name as input and build on create_dataset function
    dataset_cls = get_dataset(dataset_name=dataset_name, dataset_type=dataset_type, split=split)
    dataloader = dataset_cls.init_as_loader(batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                            num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last,
                                            worker_init_fn=worker_init_fn, indices=indices, **kwargs)
    return dataloader


def create_compound_dataset(datasets, common_keys=None):
    """Create a compound dataset.

    Args:
        datasets (list): List of datasets to be combined.
        common_keys (list): List of keys that need to be present in all datasets and will be used to filter the data
            in each sample.
    """
    compound_dataset = CompoundDataset(datasets=datasets, common_keys=common_keys)
    return compound_dataset


def create_compound_dataloader(datasets, common_keys=None, batch_size=1, shuffle=False, num_workers=0, collate_fn=None,
                               pin_memory=False, drop_last=False, worker_init_fn=None, indices=None):
    dataloader = CompoundDataset.init_as_loader(datasets=datasets, common_keys=common_keys, batch_size=batch_size,
                                                shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers,
                                                collate_fn=collate_fn, drop_last=drop_last,
                                                worker_init_fn=worker_init_fn, indices=indices)
    return dataloader
