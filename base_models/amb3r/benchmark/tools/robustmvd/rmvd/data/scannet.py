# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/data/scannet.py

import os.path as osp

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_dataset, register_default_dataset
from .layouts import MVDSequentialDefaultLayout, AllImagesLayout


class ScanNetImage:
    def __init__(self, path, height, width):
        self.path = path
        self.height = height
        self.width = width

    def load(self, root):
        image_path = osp.join(root, self.path)
        image = Image.open(image_path).resize((self.width, self.height), Image.LANCZOS)
        image = np.array(image, dtype=np.float32).transpose(2, 0, 1)  # 3, H, W
        return image


class ScanNetDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        import cv2
        path = osp.join(root, self.path)

        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth = (depth / 1000.0)  # H, W

        depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
        depth = np.expand_dims(depth, 0)  # 1HW

        return depth


class ScanNetSample(Sample):

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
class ScanNetRobustMVD(Dataset):

    base_dataset = 'scannet'
    split = 'robustmvd'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("scannet", "root")

        default_layouts = [
            MVDSequentialDefaultLayout("default", num_views=8, keyview_idx=3),
            AllImagesLayout("all_images", num_views=8),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(root=root, layouts=layouts, **kwargs)
