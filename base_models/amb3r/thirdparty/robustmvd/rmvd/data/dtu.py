# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/dtu.py

import os.path as osp
import re

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_dataset, register_default_dataset
from .layouts import MVDUnstructuredDefaultLayout, AllImagesLayout


def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data


class DTUImage:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        path = osp.join(root, self.path)
        img = np.array(Image.open(path), dtype=np.float32).transpose(2, 0, 1)
        return img


class DTUDepth:
    def __init__(self, path, format=None):
        self.path = path

    def load(self, root):
        depth = readPFM(osp.join(root, self.path)) / 1000
        depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
        depth = np.expand_dims(depth, 0)  # 1HW
        return depth


class DTUSample(Sample):

    def __init__(self, name, base):
        self.name = name
        self.base = base
        self.data = {}

    def load(self, root):

        base = osp.join(root, self.base)
        out_dict = {'_base': base, '_name': self.name}

        for key, val in self.data.items():
            if not isinstance(val, list):
                if getattr(val, "load", False):
                    out_dict[key] = val.load(base)
                else:
                    out_dict[key] = val
            else:
                out_dict[key] = [ele if isinstance(ele, np.ndarray) else ele.load(base) for ele in val]

        return out_dict


@register_default_dataset
class DTURobustMVD(Dataset):

    base_dataset = 'dtu'
    split = 'robustmvd'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("dtu", "root")

        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=11, max_views=4),
            AllImagesLayout("all_images", num_views=11),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(root=root, layouts=layouts, **kwargs)
