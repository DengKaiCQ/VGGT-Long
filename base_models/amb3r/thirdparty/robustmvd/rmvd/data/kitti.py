# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/kitti.py

import os.path as osp

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_dataset, register_default_dataset
from .layouts import MVDSequentialDefaultLayout, AllImagesLayout


class KITTIImage:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        path = osp.join(root, self.path)
        image = np.array(Image.open(path).convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
        return image


class KITTIDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        path = osp.join(root, self.path)

        depth_png = np.array(Image.open(path), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)

        depth = depth_png.astype(float) / 256.
        depth[depth_png == 0] = np.NaN

        depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
        depth = np.expand_dims(depth, 0)  # 1HW

        return depth


class KITTISample(Sample):

    def __init__(self, name):
        self.name = name
        self.data = {}

    def load(self, root):

        out_dict = {'_base': root, '_name': self.name}

        for key, val in self.data.items():
            if not isinstance(val, list):
                if getattr(val, "load", False):
                    out_dict[key] = val.load(root)
                else:
                    out_dict[key] = val
            else:
                out_dict[key] = [ele if isinstance(ele, np.ndarray) else ele.load(root) for ele in val]

        return out_dict


@register_default_dataset
class KITTIRobustMVD(Dataset):

    base_dataset = 'kitti'
    split = 'robustmvd'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("kitti", "root")

        default_layouts = [
            MVDSequentialDefaultLayout("default", num_views=21, keyview_idx=10),
            AllImagesLayout("all_images", num_views=21),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(root=root, layouts=layouts, **kwargs)
                

@register_dataset   
class KITTIEigenDenseDepthTest(Dataset):

    base_dataset = 'kitti'
    split = 'eigen_dense_depth_test'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("kitti", "root")

        default_layouts = [
            MVDSequentialDefaultLayout("default", num_views=1, keyview_idx=0),
            AllImagesLayout("all_images", num_views=1),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(root=root, layouts=layouts, **kwargs)
