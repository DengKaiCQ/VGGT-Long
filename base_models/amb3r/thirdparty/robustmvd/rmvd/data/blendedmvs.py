# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/blendedmvs.py

import os
import os.path as osp
import re

from tqdm import tqdm
import numpy as np
from PIL import Image

from .dataset import Dataset, Sample, _get_sample_list_path
from .registry import register_default_dataset, register_dataset
from .layouts import MVDUnstructuredDefaultLayout, AllImagesLayout


# from https://github.com/YoYo000/BlendedMVS/blob/master/project_lists/BlendedMVS_training.txt
BMVS_TRAIN_SCENES = ['5c1f33f1d33e1f2e4aa6dda4', '5bfe5ae0fe0ea555e6a969ca', '5bff3c5cfe0ea555e6bcbf3a',
                    '58eaf1513353456af3a1682a', '5bfc9d5aec61ca1dd69132a2', '5bf18642c50e6f7f8bdbd492',
                    '5bf26cbbd43923194854b270', '5bf17c0fd439231948355385', '5be3ae47f44e235bdbbc9771',
                    '5be3a5fb8cfdd56947f6b67c', '5bbb6eb2ea1cfa39f1af7e0c', '5ba75d79d76ffa2c86cf2f05',
                    '5bb7a08aea1cfa39f1a947ab', '5b864d850d072a699b32f4ae', '5b6eff8b67b396324c5b2672',
                    '5b6e716d67b396324c2d77cb', '5b69cc0cb44b61786eb959bf', '5b62647143840965efc0dbde',
                    '5b60fa0c764f146feef84df0', '5b558a928bbfb62204e77ba2', '5b271079e0878c3816dacca4',
                    '5b08286b2775267d5b0634ba', '5afacb69ab00705d0cefdd5b', '5af28cea59bc705737003253',
                    '5af02e904c8216544b4ab5a2', '5aa515e613d42d091d29d300', '5c34529873a8df509ae57b58',
                    '5c34300a73a8df509add216d', '5c1af2e2bee9a723c963d019', '5c1892f726173c3a09ea9aeb',
                    '5c0d13b795da9479e12e2ee9', '5c062d84a96e33018ff6f0a6', '5bfd0f32ec61ca1dd69dc77b',
                    '5bf21799d43923194842c001', '5bf3a82cd439231948877aed', '5bf03590d4392319481971dc',
                    '5beb6e66abd34c35e18e66b9', '5be883a4f98cee15019d5b83', '5be47bf9b18881428d8fbc1d',
                    '5bcf979a6d5f586b95c258cd', '5bce7ac9ca24970bce4934b6', '5bb8a49aea1cfa39f1aa7f75',
                    '5b78e57afc8fcf6781d0c3ba', '5b21e18c58e2823a67a10dd8', '5b22269758e2823a67a3bd03',
                    '5b192eb2170cf166458ff886', '5ae2e9c5fe405c5076abc6b2', '5adc6bd52430a05ecb2ffb85',
                    '5ab8b8e029f5351f7f2ccf59', '5abc2506b53b042ead637d86', '5ab85f1dac4291329b17cb50',
                    '5a969eea91dfc339a9a3ad2c', '5a8aa0fab18050187cbe060e', '5a7d3db14989e929563eb153',
                    '5a69c47d0d5d0a7f3b2e9752', '5a618c72784780334bc1972d', '5a6464143d809f1d8208c43c',
                    '5a588a8193ac3d233f77fbca', '5a57542f333d180827dfc132', '5a572fd9fc597b0478a81d14',
                    '5a563183425d0f5186314855', '5a4a38dad38c8a075495b5d2', '5a48d4b2c7dab83a7d7b9851',
                    '5a489fb1c7dab83a7d7b1070', '5a48ba95c7dab83a7d7b44ed', '5a3ca9cb270f0e3f14d0eddb',
                    '5a3cb4e4270f0e3f14d12f43', '5a3f4aba5889373fbbc5d3b5', '5a0271884e62597cdee0d0eb',
                    '59e864b2a9e91f2c5529325f', '599aa591d5b41f366fed0d58', '59350ca084b7f26bf5ce6eb8',
                    '59338e76772c3e6384afbb15', '5c20ca3a0843bc542d94e3e2', '5c1dbf200843bc542d8ef8c4',
                    '5c1b1500bee9a723c96c3e78', '5bea87f4abd34c35e1860ab5', '5c2b3ed5e611832e8aed46bf',
                    '57f8d9bbe73f6760f10e916a', '5bf7d63575c26f32dbf7413b', '5be4ab93870d330ff2dce134',
                    '5bd43b4ba6b28b1ee86b92dd', '5bccd6beca24970bce448134', '5bc5f0e896b66a2cd8f9bd36',
                    '5b908d3dc6ab78485f3d24a9', '5b2c67b5e0878c381608b8d8', '5b4933abf2b5f44e95de482a',
                    '5b3b353d8d46a939f93524b9', '5acf8ca0f3d8a750097e4b15', '5ab8713ba3799a1d138bd69a',
                    '5aa235f64a17b335eeaf9609', '5aa0f9d7a9efce63548c69a1', '5a8315f624b8e938486e0bd8',
                    '5a48c4e9c7dab83a7d7b5cc7', '59ecfd02e225f6492d20fcc9', '59f87d0bfa6280566fb38c9a',
                    '59f363a8b45be22330016cad', '59f70ab1e5c5d366af29bf3e', '59e75a2ca9e91f2c5526005d',
                    '5947719bf1b45630bd096665', '5947b62af1b45630bd0c2a02', '59056e6760bb961de55f3501',
                    '58f7f7299f5b5647873cb110', '58cf4771d0f5fb221defe6da', '58d36897f387231e6c929903',
                    '58c4bb4f4a69c55606122be4']


# from https://github.com/YoYo000/BlendedMVS/blob/master/project_lists/validation_list.txt
BMVS_VAL_SCENES = ['5b7a3890fc8fcf6781e2593a', '5c189f2326173c3a09ed7ef3', '5b950c71608de421b1e7318f', 
                   '5a6400933d809f1d8200af15', '59d2657f82ca7774b1ec081d', '5ba19a8a360c7c30c1c169df',
                   '59817e4a1bd4b175e7038d19']


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
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data


def load_image(root, path):
    path = f"blended_images/{path:08d}_masked.jpg"
    img_path = osp.join(root, path)
    img = np.array(Image.open(img_path))
    img = img.transpose(2, 0, 1).astype(np.float32)  # 3,H,W ; dtype np.uint8
    return img


def load_pose(root, path):
    path = f"cams/{path:08d}_cam.txt"
    pose_path = osp.join(root, path)
    with open(pose_path) as pose_file:
        pose_lines = [x[:-1] for x in pose_file.readlines()][1:5]
        pose_eles = [float(x) for line in pose_lines for x in line.split()]
        pose = np.array([pose_eles[0:4], pose_eles[4:8], pose_eles[8:12], pose_eles[12:16], ], dtype=np.float32)
    return pose  # 4, 4


def load_intrinsics(root, path):
    path = f"cams/{path:08d}_cam.txt"
    pose_path = osp.join(root, path)
    with open(pose_path) as pose_file:
        intrinsic_lines = [x[:-1] for x in pose_file.readlines()][7:10]
        intrinsic_eles = [float(x) for line in intrinsic_lines for x in line.split()]
        intrinsic = np.array([intrinsic_eles[0:3], intrinsic_eles[3:6], intrinsic_eles[6:9], ], dtype=np.float32)
    return intrinsic  # 3, 3


def load_depth(root, path):
    path = f"rendered_depth_maps/{path:08d}.pfm"
    depth = readPFM(osp.join(root, path))
    depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
    depth = np.expand_dims(depth, 0).astype(np.float32)  # 1HW
    return depth  # 1, H, W, np.float32


class BlendedMVSPair:
    def __init__(self, path):

        with open(path) as pair_file:
            pair_lines = pair_file.readlines()
            self.keyview_ids = [int(x.rstrip()) for x in pair_lines[1::2]]
            pair_lines = [x.rstrip() for x in pair_lines[2::2]]
            pair_lines = [x.split(" ") for x in pair_lines]
            pair_indices = [pair_line[1::2] for pair_line in pair_lines]
            self._other_view_ids = [list(map(int, indices)) for indices in pair_indices]
            pair_scores = [pair_line[2::2] for pair_line in pair_lines]
            self._other_view_scores = [list(map(float, scores)) for scores in pair_scores]

            for idx, other_view_ids in enumerate(self._other_view_ids):
                while 0 < len(other_view_ids) < 10:
                    other_view_scores = self._other_view_scores[idx]

                    to_be_added = min(len(other_view_ids), 10-len(other_view_ids))
                    other_view_ids += other_view_ids[0:to_be_added]
                    other_view_scores += other_view_scores[0:to_be_added]

                    self._other_view_ids[idx] = other_view_ids
                    self._other_view_scores[idx] = other_view_scores

    def get_source_ids(self, keyview_id):
        idx = self.keyview_ids.index(keyview_id)
        return self._other_view_ids[idx]

    def get_source_scores(self, keyview_id):
        idx = self.keyview_ids.index(keyview_id)
        return self._other_view_scores[idx]


class BlendedMVSMinDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        pose_path = osp.join(root, self.path)
        with open(pose_path) as pose_file:
            depth_line = pose_file.readlines()[11]
            depths = [float(x) for x in depth_line.split(" ")]
            min_depth, max_depth = depths[0], depths[-1]
        return min_depth  # float value


class BlendedMVSMaxDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        pose_path = osp.join(root, self.path)
        with open(pose_path) as pose_file:
            depth_line = pose_file.readlines()[11]
            depths = [float(x) for x in depth_line.split(" ")]
            min_depth, max_depth = depths[0], depths[-1]
        return max_depth  # float value


def load(key, root, val):
    if isinstance(val, list):
        return [load(key, root, v) for v in val]
    else:
        if key == 'images':
            return load_image(root, val)
        elif key == 'depth':
            return load_depth(root, val)
        elif key == 'intrinsics':
            return load_intrinsics(root, val)
        elif key == 'poses':
            return load_pose(root, val)
        else:
            return val


class BlendedMVSSample(Sample):

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


class BlendedMVSSequence:
    def __init__(self, root):

        self.root = root
        self.name = osp.split(root)[1]
        
        pair = BlendedMVSPair(osp.join(root, "cams", "pair.txt"))
        self.source_ids = {keyview_id: pair.get_source_ids(keyview_id) for keyview_id in pair.keyview_ids}
        self.source_scores = {keyview_id: pair.get_source_scores(keyview_id) for keyview_id in pair.keyview_ids}
        
        cam_files = [x for x in os.listdir(osp.join(root, "cams")) if x.endswith('cam.txt')]
        self.min_depths = {int(x[:8]): BlendedMVSMinDepth(osp.join("cams", x)).load(root) for x in cam_files}
        self.max_depths = {int(x[:8]): BlendedMVSMaxDepth(osp.join("cams", x)).load(root) for x in cam_files}

        # perform some checks that all data is there:
        images = [x for x in os.listdir(osp.join(root, "blended_images")) if x.endswith('masked.jpg')]
        self.images = [int(x[:8]) for x in images]
        depths = [x for x in os.listdir(osp.join(root, "rendered_depth_maps")) if x.endswith('.pfm')]
        self.depths = [int(x[:8]) for x in depths]
        self.intrinsics = [int(x[:8]) for x in cam_files]
        self.poses=[int(x[:8]) for x in cam_files]

        assert len(set(self.images).intersection(set(self.depths)).intersection(set(self.intrinsics)).intersection(set(self.min_depths.keys())).intersection(set(self.max_depths.keys())).intersection(set(self.poses))) == len(self.images)

        for key_id, cur_source_ids in self.source_ids.items():
            assert key_id in self.images
            assert key_id in self.depths
            assert key_id in self.poses
            assert key_id in self.intrinsics
            for source_id in cur_source_ids:
                assert source_id in self.images
                assert source_id in self.depths
                assert source_id in self.poses
                assert source_id in self.intrinsics
            assert len(cur_source_ids) == 10

    def __len__(self):
        return len(self.images)
    
    
class BlendedMVS(Dataset):
    
    def _init_samples(self, scene_names=None, num_source_views=None):
        sample_list_path = _get_sample_list_path(self.name)
        if sample_list_path is not None and osp.isfile(sample_list_path):
            super()._init_samples_from_list()
        else:
            self._init_samples_from_root_dir(scene_names=scene_names, num_source_views=num_source_views)
            self._write_samples_list()

    def _init_samples_from_root_dir(self, scene_names=None, num_source_views=None):

        from itertools import combinations

        seqs = [x for x in os.listdir(self.root) if osp.isdir(osp.join(self.root, x))]
        seqs = [x for x in seqs if x in scene_names] if scene_names is not None else seqs
        seqs = sorted(seqs)
        seqs = [BlendedMVSSequence(osp.join(self.root, x)) for x in seqs]

        for seq in (tqdm(seqs) if self.verbose else seqs):
            for key_id in seq.source_ids.keys():

                all_source_ids = seq.source_ids[key_id]
                all_source_scores = seq.source_scores[key_id]
                cur_num_source_views = num_source_views if num_source_views is not None else len(all_source_ids)
                source_id_combinations = [list(x) for x in list(combinations(all_source_ids, cur_num_source_views))]

                for source_ids in source_id_combinations:
                    sample = BlendedMVSSample(name=seq.name + "/key{:06d}".format(key_id), base=seq.name)

                    source_scores = [all_source_scores[all_source_ids.index(x)] for x in source_ids]
                    all_ids = [key_id] + source_ids

                    images = all_ids
                    poses = all_ids
                    intrinsics = all_ids
                    min_depth = seq.min_depths[key_id]
                    max_depth = seq.max_depths[key_id]
                    depth = key_id

                    sample.data['images'] = images
                    sample.data['poses'] = poses
                    sample.data['intrinsics'] = intrinsics
                    sample.data['depth'] = depth
                    sample.data['depth_range'] = (min_depth, max_depth)
                    sample.data['keyview_idx'] = 0
                    
                    # sample.data['_keyview_id'] = key_id
                    # sample.data['_source_view_ids'] = source_ids
                    # sample.data['_source_view_scores'] = source_scores

                    self.samples.append(sample)


class BlendedMVSSeq4Train(BlendedMVS):  # intentionally not registered as dataset, as this split is not used anywhere

    base_dataset = 'blendedmvs'
    split = 'seq4_train'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("blendedmvs", "root")
        
        scene_names = BMVS_TRAIN_SCENES

        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=5, max_views=5),
            AllImagesLayout("all_images", num_views=5),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(scene_names=scene_names, num_source_views=4, root=root, layouts=layouts, **kwargs)
        
        
@register_default_dataset
class BlendedMVSSeq4TrainSmall(BlendedMVSSeq4Train):
    split = 'robust_mvd'

    def _init_samples_from_root_dir(self, scene_names=None, num_source_views=None):
        super()._init_samples_from_root_dir(scene_names=scene_names, num_source_views=num_source_views)
        self.samples = self.samples[::2]
