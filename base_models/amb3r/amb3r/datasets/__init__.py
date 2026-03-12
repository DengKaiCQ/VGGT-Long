# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/__init__.py

import os
import sys
import torch 
import numpy as np
from .demo import Demo
from .seven_scenes import SevenScenes   
from .eth3d import Eth3d
from .dtu import Dtu

from .sintel import Sintel
from .bonn import Bonn
from .kitti import Kitti

from .tnt import Tnt
from .imc import Imc


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))
from thirdparty.dust3r.datasets.utils.transforms import *

def safe_collate(batch):
    try:
        return default_collate(batch)
    except Exception as e:
        for i, sample in enumerate(batch):
            views, views_all = sample

            for view in views:
                if isinstance(view, dict):
                    for key, value in view.items():
                        if isinstance(value, np.ndarray):
                            print(f"Sample {i}, view {view['label']} has key '{key}' with shape {value.shape} and dtype {value.dtype}")
                        elif isinstance(value, torch.Tensor):
                            print(f"Sample {i}, view {view['label']} has key '{key}' with shape {value.shape} and dtype {value.dtype}")
                        else:
                            print(f"Sample {i}, view {view['label']} has key '{key}' with unsupported type: {type(value)}")

                
                
        raise e


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from thirdparty.croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    # sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
    #                                    rank=rank, drop_last=drop_last)


    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        # collate_fn=safe_collate,
    )

    return data_loader


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def collate_fn(batch):
    """
    Collate function to handle batches of `views`, each possibly containing TSDF fields of varying lengths.
    Concatenates all tsdf-related fields and generates `tsdf_batch_ids`.
    """
    from collections import defaultdict
    import torch

    batch_size = len(batch)
    num_views = len(batch[0])  # assuming each item returns a list of views

    collated_views = [[] for _ in range(num_views)]

    for b in range(batch_size):
        views = batch[b]
        for v in range(num_views):
            collated_views[v].append(views[v])

    batched_output = []
    for v_views in collated_views:
        out = defaultdict(list)
        tsdf_lens = []

        for batch_idx, view in enumerate(v_views):
            for key, val in view.items():
                # if isinstance(val, np.ndarray):
                #     raise ValueError(f"Unexpected numpy array in view: {key}")

                if key in {'tsdf', 'tsdf_xyz', 'tsdf_indices'}:
                    out[key].append(torch.from_numpy(val) if isinstance(val, np.ndarray) else val)
                    if key == 'tsdf':
                        tsdf_lens.append(len(val))  # assume tsdf is 1D
                elif isinstance(val, torch.Tensor):
                    out[key].append(val.unsqueeze(0))
                elif isinstance(val, np.ndarray) and val.ndim > 0:
                    out[key].append(torch.from_numpy(val).unsqueeze(0))
                else:
                    out[key].append(val)

        # Stack what can be stacked
        for key in out:
            if isinstance(out[key][0], torch.Tensor) and key not in {'tsdf', 'tsdf_xyz', 'tsdf_indices'}:
                out[key] = torch.cat(out[key], dim=0)

        # Concatenate tsdf fields and add batch ids
        if 'tsdf' in out:
            out['tsdf'] = torch.cat(out['tsdf'], dim=0)
            out['tsdf_xyz'] = torch.cat(out['tsdf_xyz'], dim=0)
            out['tsdf_indices'] = torch.cat(out['tsdf_indices'], dim=0)

            batch_ids = []
            for i, l in enumerate(tsdf_lens):
                batch_ids.append(torch.full((l,), i, dtype=torch.long))
            out['tsdf_batch_ids'] = torch.cat(batch_ids, dim=0)

        batched_output.append(dict(out))

    return batched_output
