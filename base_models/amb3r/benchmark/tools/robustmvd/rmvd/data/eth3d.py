# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/data/eth3d.py

import os.path as osp

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_dataset, register_default_dataset
from .layouts import MVDUnstructuredDefaultLayout, AllImagesLayout


class ETH3DImage:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        img = np.array(Image.open(osp.join(root, self.path)))
        img = img.transpose([2, 0, 1]).astype(np.float32)
        return img  # 3, H, W, float32


class ETH3DDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        height, width = 4032, 6048
        depth = np.fromfile(osp.join(root, self.path), dtype=np.float32).reshape(height, width)
        depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
        depth = np.expand_dims(depth, 0)  # 1HW
        return depth


class ETH3DSample(Sample):

    def __init__(self, base, name):
        self.base = base
        self.name = name
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
class ETH3DTrainRobustMVD(Dataset):

    base_dataset = 'eth3d'
    split = 'robustmvd'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("eth3d", "root")

        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=11, max_views=4),
            AllImagesLayout("all_images", num_views=11),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(root=root, layouts=layouts, **kwargs)
