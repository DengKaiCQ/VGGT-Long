# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/utils/utils.py

import importlib
import collections
import re
import random

import numpy as np
import torch
# from torch import string_classes
string_classes = str
import pytoml


def set_random_seed(seed=None, use_gpu=True):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_depth_range(depth=None, invdepth=None, default_min_depth=0.1, default_max_depth=100.):
    if depth is not None:
        mask = depth > 0
        if mask.any():
            min_depth = np.min(depth[mask])
            max_depth = np.max(depth[mask])
            depth_range = (min_depth, max_depth)
            return depth_range
        
    if invdepth is not None:
        mask = invdepth > 0
        if mask.any():
            min_invdepth = np.min(invdepth[mask])
            max_invdepth = np.max(invdepth[mask])
            depth_range = (1./max_invdepth, 1./min_invdepth)
            return depth_range
        
    return (default_min_depth, default_max_depth)


def get_function(name):  # from https://github.com/aschampion/diluvian/blob/master/diluvian/util.py
    mod_name, func_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def get_class(name):
    return get_function(name)


def module_exists(name):
    spec = importlib.util.find_spec(name)
    return (spec is not None)


def function_exists(name):
    mod_name, fct_name = name.rsplit('.', 1)
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        return False
    else:
        mod = importlib.import_module(mod_name)
        return hasattr(mod, fct_name)


def class_exists(name):
    return function_exists(name)


def get_full_class_name(obj):
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + obj.__class__.__name__


def trans_from_transform(T):
    return T[0:3, 3]


def rot_from_transform(T):
    return T[0:3, 0:3]


def transform_from_rot_trans(R, t):
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1])).astype(np.float32)


def invert_transform(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = np.dot(-R.T, t)
    return transform_from_rot_trans(R_inv, t_inv)


def to_cuda(data, device=None):
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cuda(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.cuda(device=device)
    else:
        return data


def torch_collate(batch):
    if batch is None:
        return None

    return torch.utils.data._utils.collate.default_collate(batch)


def to_torch(data, device=None):
    # adapted from torch.utils.data._utils.collate.default_convert
    np_str_obj_array_pattern = re.compile(r'[SaUO]')

    if data is None:
        return None

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data, device=device)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: to_torch(data[key], device=device) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`...
            return {key: to_torch(data[key], device=device) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(to_torch(d, device=device) for d in data))
    elif isinstance(data, tuple):
        return [to_torch(d, device=device) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([to_torch(d, device=device) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [to_torch(d, device=device) for d in data]
    else:
        return data


def numpy_collate(batch):
    # adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    err_msg_format = "numpy_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"

    if batch is None:
        return None

    elem = batch[0]
    elem_type = type(elem)

    if elem is None:
        assert all(elem is None for elem in batch)
        return None

    elif isinstance(elem, torch.Tensor):
        return numpy_collate([b.cpu().detach().numpy() for b in batch])

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return np.stack(batch, 0)
        elif elem.shape == ():  # scalars
            return np.array(batch)

    elif isinstance(elem, float):
        return np.array(batch)

    elif isinstance(elem, int):
        return np.array(batch)

    elif isinstance(elem, string_classes):
        return batch

    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: numpy_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: numpy_collate([d[key] for d in batch]) for key in elem}

    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(numpy_collate(samples) for samples in zip(*batch)))

    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [numpy_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([numpy_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [numpy_collate(samples) for samples in transposed]

    raise TypeError(err_msg_format.format(elem_type))


def to_numpy(data):
    if data is None:
        return None

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: to_numpy(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: to_numpy(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(to_numpy(d) for d in data))
    elif isinstance(data, tuple):
        return [to_numpy(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([to_numpy(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [to_numpy(d) for d in data]
    else:
        return data


def get_torch_model_device(model):
    # make sure that all parameters are on the same device:
    it = iter(model.parameters())
    is_cuda = next(it).is_cuda
    device = next(it).device
    if not all((elem.device == device) for elem in it):
        raise RuntimeError('All model parameters need to be on the same device')
    return device


def check_torch_model_cuda(model):
    # make sure that model parameters are all on the GPU or all on the CPU:
    it = iter(model.parameters())
    is_cuda = next(it).is_cuda
    if not all((elem.is_cuda == is_cuda) for elem in it):
        raise RuntimeError('All model parameters need to be on the same device')
    return is_cuda


def count_torch_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_by_index(l, idx):
    """Select an element from a list by an index. Supports data batches with different indices.

    Args:
        l (list): List with potentially batched data items.
        idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
            Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
    """
    if isinstance(idx, int):
        ret = l[idx]
    else:
        indices = idx
        ret = []
        for batch_idx, idx in enumerate(indices):
            ret.append(l[idx][batch_idx])

        if isinstance(ret[0], np.ndarray):
            ret = np.stack(ret, 0)
        else:
            ret = torch.stack(ret, 0)

    return ret


def exclude_index(l, exclude_idx):
    """Selects all element from a list, excluding a specific index. Supports data batches with different indices.

    Args:
        l (list): List with potentially batched data items.
        idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
            Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
    """
    if isinstance(exclude_idx, int):
        ret = [ele for idx, ele in enumerate(l) if idx != exclude_idx]
    else:
        exclude_indices = exclude_idx
        ret = []
        for batch_idx, exclude_idx in enumerate(exclude_indices):
            ret.append([ele[batch_idx] for idx, ele in enumerate(l) if idx != exclude_idx])

        transposed = list(zip(*ret))
        if isinstance(transposed[0][0], np.ndarray):
            ret = [np.stack(ele, 0) for ele in transposed]
        else:
            ret = [torch.stack(ele, 0) for ele in transposed]

    return ret


def batched_index(l, eles):
    """Gets the indices of a batch of elements in a list.

    This is basically the equivalent of l.index(ele) in a batched version.

    Args:
        l (list): List with batched data items.
        eles: Iterable that contains elemenets for each sample in the batch.
    """
    ret = []
    for batch_idx, ele in enumerate(eles):
        batch_l = [x[batch_idx] for x in l]
        ret.append(batch_l.index(ele))

    if isinstance(ret[0], np.ndarray):
        ret = np.stack(ret, 0)
    else:
        ret = torch.stack(ret, 0)

    return ret


def get_paths(paths_file):
    with open(paths_file, 'r') as paths_file:
        return pytoml.load(paths_file)


def get_path(paths_file, *keys):
    paths = get_paths(paths_file)
    path = None

    for idx, key in enumerate(keys):
        if key in paths:
            if key in paths and (isinstance(paths[key], str) or isinstance(paths[key], list)) and idx == len(keys) - 1:
                path = paths[key]
            else:
                paths = paths[key]

    return path


def normalize(x, dim=None, eps=1e-9):
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    return x / (norm + eps)
