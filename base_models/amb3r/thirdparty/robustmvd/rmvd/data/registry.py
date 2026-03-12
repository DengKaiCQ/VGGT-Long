# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/registry.py

"""Dataset Registry.

Maintains a registry of datasets.
Based on the model registry from the timm package ( https://github.com/rwightman/pytorch-image-models ).
"""


_datasets = {}  # keys are (base_dataset, dataset_type, split)
_default_splits = {}
_aug_fcts = {}
_batch_aug_fcts = {}


def register_dataset(dataset_cls):
    """Register a dataset."""
    base_dataset = dataset_cls.base_dataset.lower()
    dataset_type = dataset_cls.dataset_type.lower()
    split = dataset_cls.split.lower()
    assert (base_dataset, dataset_type, split) not in _datasets, \
        f"Dataset {str((base_dataset, dataset_type, split))} is already registered."
    _datasets[(base_dataset, dataset_type, split)] = dataset_cls
    return dataset_cls


def register_default_dataset(dataset_cls):
    """Register the default split of a dataset."""
    register_dataset(dataset_cls)

    base_dataset = dataset_cls.base_dataset.lower()
    dataset_type = dataset_cls.dataset_type.lower()
    split = dataset_cls.split.lower()
    assert (base_dataset, dataset_type) not in _default_splits, \
        f"Dataset {str((base_dataset, dataset_type))} has already a default split."
    _default_splits[(base_dataset, dataset_type)] = split
    return dataset_cls


def list_datasets(base_dataset=None, dataset_type=None, split=None, no_dataset_type=False, no_split=False):
    """Get list of available datasets.

    Args:
        base_dataset (str): name of the original dataset.
        dataset_type (str): dataset type.
        split (str): dataset split.
        no_dataset_type: do not include dataset type in the dataset name.
        no_split: do not include split in the dataset name.
    """
    datasets = _filter_datasets(base_dataset=base_dataset, dataset_type=dataset_type, split=split)
    datasets = [_build_dataset_name(*k, no_dataset_type=no_dataset_type, no_split=no_split) for k in datasets]
    return list(sorted(datasets))


def list_base_datasets(dataset_type=None, split=None):
    """Get list of all base datasets."""
    datasets = _filter_datasets(dataset_type=dataset_type, split=split)
    base_datasets = sorted(list(set([k[0] for k in datasets])))
    return base_datasets


def list_dataset_types(base_dataset=None, split=None):
    """Get list of all dataset types."""
    datasets = _filter_datasets(base_dataset=base_dataset, split=split)
    dataset_types = sorted(list(set([k[1] for k in datasets])))
    return dataset_types


def list_splits(base_dataset=None, dataset_type=None):
    """Get list of all dataset splits."""
    datasets = _filter_datasets(base_dataset=base_dataset, dataset_type=dataset_type)
    splits = sorted(list(set([k[2] for k in datasets])))
    return splits


def _filter_datasets(base_dataset=None, dataset_type=None, split=None):
    """Get filtered list of available datasets.

    Args:
        base_dataset (str): name of the original dataset.
        dataset_type (str): dataset type.
        split (str): dataset split.
    """
    base_dataset = base_dataset.lower() if base_dataset is not None else None
    dataset_type = dataset_type.lower() if dataset_type is not None else None
    split = split.lower() if split is not None else None

    datasets = _datasets.keys()
    datasets = [k for k in datasets if base_dataset is None or k[0] == base_dataset]
    datasets = [k for k in datasets if dataset_type is None or k[1] == dataset_type]
    datasets = [k for k in datasets if split is None or k[2] == split]

    return datasets


def _split_dataset_name(dataset_name, dataset_type=None, split=None):
    dataset_name = dataset_name.lower()
    dataset_type = dataset_type.lower() if dataset_type is not None else None
    split = split.lower() if split is not None else None

    s = dataset_name.split(".")

    if s[-1] in list_dataset_types():
        if dataset_type is not None:
            assert s[-1] == dataset_type, "The given dataset name conflicts with the given dataset type."
        else:
            dataset_type = s[-1]
        s = s[:-1]

    assert dataset_type is not None, \
        f"Dataset type must be provided. Available types are: {','.join(list_dataset_types())}"

    if split is None and dataset_type is not None and ('.'.join(s), dataset_type) in _default_splits:
        split = _default_splits[('.'.join(s), dataset_type)]
    if split is not None and split in s:
        s.remove(split)
    if split is None:
        s, split = s[:-1], s[-1]

    base_dataset = '.'.join(s)
    return base_dataset, dataset_type, split


def _build_dataset_name(dataset_name, dataset_type=None, split=None, no_dataset_type=False, no_split=False):
    dataset_name = dataset_name.lower()
    dataset_type = dataset_type.lower() if dataset_type is not None else None
    split = split.lower() if split is not None else None

    s = dataset_name.split(".")

    if s[-1] in list_dataset_types():
        if dataset_type is not None:
            assert s[-1] == dataset_type, "The given dataset name conflicts with the given dataset type."
        else:
            dataset_type = s[-1]
        s = s[:-1]

    if split is None and dataset_type is not None and ('.'.join(s), dataset_type) in _default_splits:
        split = _default_splits[('.'.join(s), dataset_type)]
    if split is not None and split in s:
        s.remove(split)

    s = s + [split] if (split is not None and not no_split) else s
    s = s + [dataset_type] if (dataset_type is not None and not no_dataset_type) else s
    dataset_name = '.'.join(s)
    return dataset_name


def has_dataset(dataset_name, dataset_type=None, split=None):
    """Check if dataset is registered."""
    try:
        base_dataset, dataset_type, split = _split_dataset_name(dataset_name=dataset_name,
                                                                dataset_type=dataset_type,
                                                                split=split)
    except AssertionError:
        return False

    return (base_dataset, dataset_type, split) in _datasets


def get_dataset(dataset_name, dataset_type=None, split=None):
    """Get dataset entrypoint by name."""
    base_dataset, dataset_type, split = _split_dataset_name(dataset_name=dataset_name,
                                                            dataset_type=dataset_type,
                                                            split=split)
    
    # TODO: add assert that dataset is registered

    return _datasets[(base_dataset, dataset_type, split)]


def register_augmentation(augmentation_factory):
    """Register augmentation by name."""
    aug_fct_name = augmentation_factory.__name__
    _aug_fcts[aug_fct_name] = augmentation_factory
    return augmentation_factory


def list_augmentations():
    """Get list of all augmentations."""
    return list(sorted(_aug_fcts.keys()))


def has_augmentation(augmentation_name):
    """Check if augmentation is registered."""
    return augmentation_name in _aug_fcts


def create_augmentation(augmentation_name, **kwargs):
    """Get augmentation by name."""
    assert has_augmentation(augmentation_name), f'The requested augmentation function "{augmentation_name}" does not exist. Available augmentation functions are: {" ".join(list_augmentations())}'
    return _aug_fcts[augmentation_name](**kwargs)


def register_batch_augmentation(batch_augmentation_factory):
    """Register batch augmentation by name."""
    batch_aug_fct_name = batch_augmentation_factory.__name__
    _batch_aug_fcts[batch_aug_fct_name] = batch_augmentation_factory
    return batch_augmentation_factory


def list_batch_augmentations():
    """Get list of all batch augmentations."""
    return list(sorted(_batch_aug_fcts.keys()))


def has_batch_augmentation(batch_augmentation_name):
    """Check if batch augmentation is registered."""
    return batch_augmentation_name in _batch_aug_fcts


def create_batch_augmentation(batch_augmentation_name, **kwargs):
    """Get batch augmentation by name."""
    assert has_batch_augmentation(batch_augmentation_name), f'The requested batch augmentation function "{batch_augmentation_name}" does not exist. Available batch augmentation functions are: {" ".join(list_batch_augmentations())}'
    return _batch_aug_fcts[batch_augmentation_name](**kwargs)
