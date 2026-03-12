# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/helpers.py

"""Helper functions.

Based on and partially copied from the model registry from the
timm package ( https://github.com/rwightman/pytorch-image-models ).
"""
from copy import deepcopy
import collections

import torch
from torch.hub import load_state_dict_from_url

from typing import Callable, Optional, Dict, Callable

from rmvd.utils import numpy_collate


_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False


def add_batch_dim(images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
    images, keyview_idx, poses, intrinsics, depth_range = \
        numpy_collate([(images, keyview_idx, poses, intrinsics, depth_range)])
    return images, keyview_idx, poses, intrinsics, depth_range


def remove_batch_dim(batch):
    err_msg_format = "remove_batch_dim: batch must contain numpy arrays, dicts or lists; found {}"

    batch_type = type(batch)

    if batch_type.__module__ == 'numpy' and batch_type.__name__ != 'str_' and batch_type.__name__ != 'string_':
        if batch_type.__name__ == 'ndarray' or batch_type.__name__ == 'memmap':
            return batch[0]

    elif isinstance(batch, collections.abc.Mapping):
        try:
            return batch_type({key: remove_batch_dim(batch[key]) for key in batch})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: remove_batch_dim(batch[key]) for key in batch}

    elif isinstance(batch, tuple) and hasattr(batch, '_fields'):  # namedtuple
        return batch_type(*(remove_batch_dim(x) for x in batch))

    elif isinstance(batch, collections.abc.Sequence):
        try:
            return batch_type([remove_batch_dim(x) for x in batch])
        except TypeError:
            return [remove_batch_dim(x) for x in batch]

    elif batch is None:
        return None

    raise TypeError(err_msg_format.format(batch_type))


def add_run_function(model):
    @torch.no_grad()
    def run(images, keyview_idx, poses=None, intrinsics=None, depth_range=None, **_):
        no_batch_dim = (images[0].ndim == 3)
        if no_batch_dim:
            images, keyview_idx, poses, intrinsics, depth_range = \
                add_batch_dim(images, keyview_idx, poses, intrinsics, depth_range)

        sample = model.input_adapter(images=images, keyview_idx=keyview_idx, poses=poses,
                                     intrinsics=intrinsics, depth_range=depth_range)
        model_output = model(**sample)
        pred, aux = model.output_adapter(model_output)

        if no_batch_dim:
            pred, aux = remove_batch_dim((pred, aux))

        return pred, aux

    model.run = run


def set_download_progress(enable=True):
    """ Set download progress for pretrained weights on/off (globally). """
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable


def set_check_hash(enable=True):
    """ Set hash checking for pretrained weights on/off (globally). """
    global _CHECK_HASH
    _CHECK_HASH = enable


def build_model_with_cfg(
        model_cls: Callable,
        cfg: Optional[Dict] = None,
        weights: Optional[str] = None,
        train: bool = False,
        num_gpus: int = 1,
        preprocess_weights_fct: Optional[Callable] = None,
        **kwargs):
    """Builds a model with a given config and restores weights.

    Args:
        model_cls: The model class.
        cfg: Dictionary with config parameters.
        weights: Path to model weights.
        train: Whether to put the model in train mode.
        num_gpus: Number of GPUs to be used from the model.
        preprocess_weights_fct: Function that is applied to the weights before loading them with the model.

    Returns:
        Model.
    """
    if cfg is not None:
        cfg = deepcopy(cfg)  # avoid changing the input cfg dict
        cfg.update(kwargs)
        kwargs = cfg
    model = model_cls(**kwargs)

    if weights is not None:
        load_from_url = weights.startswith('http')
        if not load_from_url:
            print(f'Using model weights from file {weights}.')
            state_dict = torch.load(weights, map_location='cpu')
        else:
            print(f'Using model weights from url {weights}.')
            state_dict = load_state_dict_from_url(weights, map_location='cpu', progress=_DOWNLOAD_PROGRESS,
                                                  check_hash=_CHECK_HASH)

        if preprocess_weights_fct is not None:
            state_dict = preprocess_weights_fct(state_dict)

        model.load_state_dict(state_dict, strict=True)

    if train:
        model.train()
    else:
        model.eval()

    if num_gpus == 1:
        model = model.cuda()
    elif num_gpus > 1:
        parallel_model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus))).cuda()
        parallel_model.input_adapter = model.input_adapter
        parallel_model.output_adapter = model.output_adapter
        model = parallel_model
    # elif num_gpus < 1: use CPU, nothing to be done

    return model
