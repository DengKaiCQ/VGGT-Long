# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/data/staticthings3d.py

import os.path as osp
import re
import itertools
from glob import glob
import pickle

from tqdm import tqdm
import numpy as np
from PIL import Image

from .dataset import Dataset, Sample, _get_sample_list_path
from .registry import register_default_dataset
from .layouts import MVDUnstructuredDefaultLayout, AllImagesLayout
from .flyingthings3d import SUBSET_FILTERED_SAMPLES, HARD_SAMPLES


def readFloat(name):
    f = open(name, 'rb')

    if (f.readline().decode("utf-8")) != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))
    data = np.fromfile(f, np.float32, count).reshape(dims)
    return data  # Hxw or CxHxW NxCxHxW


class DataConf:
    def __init__(self, id, perspective=None, offset=0):
        self.id = id
        self.perspective = perspective
        self.offset = offset

    @property
    def ext(self):
        if self.id == 'frames_cleanpass' or self.id == 'frames_finalpass':
            return 'png'
        else:
            return 'float3'
        
    @property
    def perspective_short(self):
        if self.perspective is None:
            return None
        else:
            return self.perspective[0]

    @property
    def path(self):
        if self.perspective is None:
            return self.id
        else:
            return osp.join(self.id, self.perspective)

    @property
    def glob(self):
        if self.perspective is None:
            return osp.join(self.id, "*.{}".format(self.ext))
        else:
            return osp.join(self.id, self.perspective, "*.{}".format(self.ext))


def load_image(root, cam, frame_num):
    filename = '{:04d}.png'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    img = np.array(Image.open(osp.join(root, "frames_cleanpass", cam, filename)))
    img = img.transpose([2, 0, 1]).astype(np.float32)
    return img  # 3, H, W, float32


def load_depth(root, cam, frame_num):
    filename = '{:04d}.float3'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    depth = readFloat(osp.join(root, "depths", cam, filename))
    depth[(depth < 0.) | np.isinf(depth) | np.isnan(depth)] = 0.
    depth = np.expand_dims(depth, 0).astype(np.float32)  # 1HW; float32
    return depth


def load_intrinsics(root, cam, frame_num):
    filename = '{:04d}.float3'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    intrinsics = readFloat(osp.join(root, "intrinsics", cam, filename))  # 3,3; float32
    return intrinsics


def load_pose(root, cam, frame_num):
    filename = '{:04d}.float3'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    pose = readFloat(osp.join(root, "poses", cam, filename))  # 4,4; float32
    return pose


def load(key, root, val):
    if isinstance(val, list):
        return [load(key, root, v) for v in val]
    else:
        if key == 'images':
            cam, frame_num = val
            return load_image(root, cam, frame_num)
        elif key == 'depth':
            cam, frame_num = val
            return load_depth(root, cam, frame_num)
        elif key == 'intrinsics':
            cam, frame_num = val
            return load_intrinsics(root, cam, frame_num)
        elif key == 'poses':
            cam, frame_num = val
            return load_pose(root, cam, frame_num)
        else:
            return val


class StaticThings3DSample(Sample):

    def __init__(self, base, name):
        self.base = base
        self.name = name
        self.data = {}

    def load(self, root):

        base = osp.join(root, self.base)
        out_dict = {'_base': base, '_name': self.name}

        for key, val in self.data.items():
            out_dict[key] = load(key, base, val)

        return out_dict


class StaticThings3D(Dataset):
    def _init_samples(self, sample_confs=None, filter_hard_samples=False, use_subset_only=False):
        sample_list_path = _get_sample_list_path(self.name)
        if sample_list_path is not None and osp.isfile(sample_list_path):
            super()._init_samples_from_list()
        else:
            self._init_samples_from_confs(sample_confs=sample_confs, filter_hard_samples=filter_hard_samples, use_subset_only=use_subset_only)
            self._write_samples_list()
    
    def _init_samples_from_confs(self, sample_confs, filter_hard_samples=False, use_subset_only=False):
        sequences = sorted(glob(osp.join(self.root, '*/*[0-9]')))

        for sequence in (tqdm(sequences) if self.verbose else sequences):
            sequence_files = glob(osp.join(sequence, '*/*/*'))
            sequence_files = [osp.relpath(f, sequence) for f in sequence_files]
            sequence_id = osp.join(osp.split(self.root)[1], osp.relpath(sequence, self.root))
            for sample_conf in sample_confs:
                for keyframe_num in range(6, 16):  # build samples for using frames 6 to 15 as keyframes
                    sample = StaticThings3DSample(base=osp.relpath(sequence, self.root),
                                                name=osp.relpath(sequence, self.root) + "/key{:02d}".format(keyframe_num))

                    sample_valid = True
                    for key, conf in sample_conf.items():

                        if not (isinstance(conf, DataConf) or isinstance(conf, list)):
                            sample.data[key] = conf
                            continue

                        elif isinstance(conf, DataConf):
                            offset_num = keyframe_num + conf.offset
                            filename = f'{offset_num:04d}.{conf.ext}'
                            if osp.join(conf.path, filename) in sequence_files:
                                if not filter_hard_samples or [sequence_id, f'{offset_num:04d}'] not in HARD_SAMPLES:
                                    if not use_subset_only or [sequence_id, f'{offset_num:04d}'] not in SUBSET_FILTERED_SAMPLES:
                                        sample.data[key] = (conf.perspective_short, offset_num)
                                    else:
                                        sample_valid = False
                                        break
                                else:
                                    sample_valid = False
                                    break
                            else:
                                sample_valid = False
                                break

                        elif isinstance(conf, list):
                            confs = conf
                            sample.data[key] = []
                            for conf in confs:
                                offset_num = keyframe_num + conf.offset
                                filename = f'{offset_num:04d}.{conf.ext}'
                                if osp.join(conf.path, filename) in sequence_files:
                                    if not filter_hard_samples or [sequence_id, f'{offset_num:04d}'] not in HARD_SAMPLES:
                                        if not use_subset_only or [sequence_id, f'{offset_num:04d}'] not in SUBSET_FILTERED_SAMPLES:
                                            sample.data[key].append((conf.perspective_short, offset_num))
                                        else:
                                            sample_valid = False
                                            break
                                    else:
                                        sample_valid = False
                                        break
                                else:
                                    sample_valid = False
                                    break

                    if sample_valid:
                        self.samples.append(sample)

    def _write_samples_list(self, path=None):
        path = _get_sample_list_path(self.name) if path is None else path
        if osp.isdir(osp.split(path)[0]):
            if self.verbose:
                print(f"Writing sample list to {path}")
            with open(path, 'wb') as file:
                pickle.dump(self.samples, file)
        elif self.verbose:
            print(f"Could not write sample list to {path}")


@register_default_dataset
class StaticThings3DSeq4Train(StaticThings3D):

    base_dataset = 'staticthings3d'
    split = 'robust_mvd'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("staticthings3d", "train", "root")
        
        sample_confs = self._get_sample_confs()
        filter_hard_samples = True
        use_subset_only = False

        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=5, max_views=5),
            AllImagesLayout("all_images", num_views=5),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(sample_confs=sample_confs, filter_hard_samples=filter_hard_samples, use_subset_only=use_subset_only, root=root, layouts=layouts, **kwargs)
        
    def _get_sample_confs(self):

        sample_confs = []

        images_key = [DataConf('frames_cleanpass', 'left', 0)]
        to_ref_transforms_base = [DataConf('poses', 'left', 0)]
        intrinsics_base = [DataConf('intrinsics', 'left', 0)]
        offset_list = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

        for offsets in itertools.combinations(offset_list, 4):
            images = images_key.copy()
            to_ref_transforms = to_ref_transforms_base.copy()
            intrinsics = intrinsics_base.copy()

            for offset in offsets:
                images = images + [DataConf('frames_cleanpass', 'left', offset)]
                to_ref_transforms = to_ref_transforms + [DataConf('poses', 'left', offset)]
                intrinsics = intrinsics + [DataConf('intrinsics', 'left', offset)]

            sample_conf = {
                'images': images,
                'poses': to_ref_transforms,
                'intrinsics': intrinsics,
                'depth': DataConf('depths', 'left', 0),
                'keyview_idx': 0,
            }
            sample_confs.append(sample_conf)

        return sample_confs
