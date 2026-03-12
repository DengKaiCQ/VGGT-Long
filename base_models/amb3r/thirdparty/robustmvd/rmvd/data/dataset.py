# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/dataset.py

import os.path as osp
import random
import abc
import pickle
import time

import torch
import numpy as np
import pytoml

import rmvd.utils as utils
from .transforms import ResizeInputs, ResizeTargets
from .updates import Updates, PickledUpdates
from .layout import Layout
from .registry import create_augmentation


class Sample(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, root):
        return


class Dataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):

    def __init__(self, root, augmentations=None, input_size=None, target_size=None, to_torch=False, updates=None, update_strict=False,
                 layouts=None, verbose=True, **kwargs):

        augmentations = [] if augmentations is None else augmentations
        augmentations = [augmentations] if not isinstance(augmentations, list) else augmentations
        self.verbose = verbose

        self.root = None
        self._init_root(root)

        if self.verbose:
            print(f"Initializing dataset {self.name} from {self.root}")

        self._seed_initialized = False
        self.input_resize = ResizeInputs(size=input_size) if input_size is not None else None
        self.target_resize = ResizeTargets(size=target_size) if target_size is not None else None
        self.augmentations = []
        self._init_augmentations(augmentations)
        self.to_torch = to_torch

        self.samples = []
        self._init_samples(**kwargs)
        self._layouts = {}
        self._init_layouts(layouts)
        self._allowed_indices = []
        self.updates = []
        self._init_updates(updates, update_strict)

        if self.verbose:
            print(f"\tNumber of samples: {len(self)}")
            if len(self.updates) > 0:
                print(f"\tUpdates: {', '.join([update.name for update in self.updates])}")
            if self.input_resize is not None:
                print(f"\tImage resolution (height, width): ({input_size[0]}, {input_size[1]})")
            if self.target_resize is not None:
                print(f"\Target resolution (height, width): ({target_size[0]}, {target_size[1]})")
            print(f"Finished initializing dataset {self.name}.")
            print()

    @property
    def name(self):
        if hasattr(self, "base_dataset"):
            name = self.base_dataset
            name = f"{name}.{self.split}" if hasattr(self, "split") else name
            name = f"{name}.{self.dataset_type}" if hasattr(self, "dataset_type") else name
        else:
            name = type(self).__name__
        return name

    @property
    def full_name(self):
        name = self.name
        for update in self.updates:
            name += f"+{update.name}"
        return name

    def _init_root(self, root):
        if isinstance(root, str):
            self.root = root
        elif isinstance(root, list):
            self.root = [path for path in root if osp.isdir(path)][0]

    def _init_augmentations(self, augmentations):
        for augmentation in augmentations:
            if isinstance(augmentation, str):
                augmentation = create_augmentation(augmentation)
            self.augmentations.append(augmentation)

    def _init_samples(self):
        self._init_samples_from_list()
        
    def _init_samples_from_list(self):
        sample_list_path = _get_sample_list_path(self.name)
        if self.verbose:
            print("\tInitializing samples from list at {}".format(sample_list_path))
        with open(sample_list_path, 'rb') as sample_list:
            self.samples = pickle.load(sample_list)

    def _write_samples_list(self, path=None):
        path = _get_sample_list_path(self.name) if path is None else path
        if osp.isdir(osp.split(path)[0]):
            if self.verbose:
                print(f"Writing sample list to {path}")
            with open(path, 'wb') as file:
                pickle.dump(self.samples, file)
        elif self.verbose:
            print(f"Could not write sample list to {path}")

    def _init_updates(self, updates, update_strict=False):

        if updates is not None:
            for update in updates:
                if isinstance(update, Updates):
                    update = update
                elif isinstance(update, str):
                    update = PickledUpdates(path=update, verbose=False)
                self.updates.append(update)

        if update_strict:
            self._allowed_indices = [i for i in range(len(self.samples)) if all([i in update for update in self.updates])]
        else:
            self._allowed_indices = list(range(len(self.samples)))

    def _init_layouts(self, layouts):
        if layouts is not None:
            for layout in layouts:
                layout = layout if isinstance(layout, Layout) else Layout.from_file(layout)
                self.add_layout(layout)

    def add_layout(self, layout):
        self._layouts[layout.name.lower()] = layout

    def get_layout_names(self):
        return list(self._layouts.keys())

    def get_layout(self, layout_name=None):
        layout_name = layout_name if layout_name is not None else 'default'
        return self._layouts[layout_name.lower()]

    def __len__(self):
        return len(self._allowed_indices)

    def __getitem__(self, index):

        index = self._allowed_indices[index]
        sample = self.samples[index]

        sample_dict = sample.load(root=self.root)
        sample_dict['_index'] = index
        sample_dict['_dataset'] = self.full_name

        _preprocess_sample(sample_dict)

        for update in self.updates:
            update.apply_update(sample_dict, index=index)

        for augmentation in self.augmentations:
            augmentation(sample_dict)

        if self.input_resize is not None:
            self.input_resize(sample_dict)
        if self.target_resize is not None:
            self.target_resize(sample_dict)

        if self.to_torch:
            sample_dict = utils.torch_collate([sample_dict])

        return sample_dict

    def __str__(self):
        return self.name

    def _get_paths(self):
        return _get_paths()

    def _get_path(self, *keys):
        return _get_path(*keys)

    @classmethod
    def init_as_loader(cls, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False,
                       drop_last=False, worker_init_fn=None, indices=None, **kwargs):
        kwargs.pop("to_torch", None)  # create dataset with to_torch=False to avoid adding batch dimension twice
        dataset = cls(**kwargs)
        dataset = torch.utils.data.Subset(dataset, indices) if indices is not None else dataset
        return dataset.get_loader(batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                  num_workers=num_workers, collate_fn=collate_fn,
                                  drop_last=drop_last, worker_init_fn=worker_init_fn)

    def get_loader(self, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False,
                   drop_last=False, worker_init_fn=None, indices=None):
        dataset = torch.utils.data.Subset(self, indices) if indices is not None else self
        dataset.to_torch = False  # create dataset with to_torch=False to avoid adding batch dimension twice
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                                 num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last,
                                                 worker_init_fn=worker_init_fn)
        return dataloader

    def timeit(self, num_batches=100, batch_size=1, num_workers=0):
        start = time.time()

        loader = self.get_loader(batch_size=batch_size, num_workers=num_workers)
        for idx, data_blob in enumerate(loader):
            if idx >= num_batches-1:
                break

        end = time.time()
        print("Total time for loading {} batches: %1.4fs.".format(num_batches) % (end - start))
        print("Mean time per batch: %1.4fs." % ((end - start)/num_batches))

    @classmethod
    def write_config(cls, path, dataset_cls_name, augmentations=None, input_size=None, to_torch=False, updates=None,
                     update_strict=False, layouts=None):

        config = {'dataset_cls_name': dataset_cls_name,
                  'augmentations': augmentations,
                  'input_size': input_size,
                  'to_torch': to_torch,
                  'updates': updates,
                  'update_strict': update_strict,
                  'layouts': layouts}

        with open(path, 'wb') as file:
            pickle.dump(config, file)

    @classmethod
    def from_config(cls, path, more_updates=None, more_layouts=None, verbose=None):
        with open(path, 'rb') as file:
            config = pickle.load(file)

        if more_updates is not None:
            more_updates = [more_updates] if not isinstance(more_updates, list) else more_updates
            updates = config['updates'] if 'updates' in config else []
            updates += more_updates
            config['updates'] = updates

        if more_layouts is not None:
            more_layouts = [more_layouts] if not isinstance(more_layouts, list) else more_layouts
            layouts = config['layouts'] if 'layouts' in config else []
            layouts += more_layouts
            config['layouts'] = layouts

        if verbose is not None:
            config['verbose'] = verbose

        dataset_cls_name = config['dataset_cls_name']
        del config['dataset_cls_name']
        dataset_cls = utils.get_class(dataset_cls_name)
        return dataset_cls(**config)


def _get_paths():
    paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
    with open(paths_file, 'r') as paths_file:
        return pytoml.load(paths_file)


def _get_sample_list_path(name):
    sample_lists_path = osp.join(osp.dirname(osp.realpath(__file__)), 'sample_lists')
    return osp.join(sample_lists_path, "{}.pickle".format(name))


def _get_path(*keys):
    paths = _get_paths()
    path = None

    for idx, key in enumerate(keys):
        if key in paths:
            if key in paths and (isinstance(paths[key], str) or isinstance(paths[key], list)) and idx == len(keys) - 1:
                path = paths[key]
            else:
                paths = paths[key]

    return path


def _preprocess_sample(sample):
    assert ('depth' in sample or 'invdepth' in sample) and (not ('depth' in sample and 'invdepth' in sample))

    if 'depth' in sample:
        with np.errstate(divide='ignore', invalid='ignore'):
            sample['depth'] = sample['depth'].astype(np.float32)
            sample['depth'][sample['depth'] <= 0] = 0
            sample['depth'][~np.isfinite(sample['depth'])] = 0
            sample['invdepth'] = np.nan_to_num(1 / sample['depth'], copy=False, nan=0, posinf=0, neginf=0)

    elif 'invdepth' in sample:
        sample['invdepth'] = sample['invdepth'].astype(np.float32)
        sample['invdepth'][sample['invdepth'] <= 0] = 0
        sample['invdepth'][~np.isfinite(sample['invdepth'])] = 0
        sample['depth'] = np.nan_to_num(1 / sample['invdepth'], copy=False, nan=0, posinf=0, neginf=0)

    if 'depth_range' not in sample:
        sample['depth_range'] = utils.compute_depth_range(depth=sample['depth'])

    key_idx = sample['keyview_idx'] if 'keyview_idx' in sample else 0
    key_to_ref_transform = sample['poses'][key_idx]
    ref_to_key_transform = utils.invert_transform(key_to_ref_transform)
    for idx, to_ref_transform in enumerate(sample['poses']):
        to_key_transform = np.dot(to_ref_transform, ref_to_key_transform)
        sample['poses'][idx] = to_key_transform
