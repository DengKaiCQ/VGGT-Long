# References:
#   https://github.com/HengyiWang/amb3r/blob/main/slam/datasets/demo.py

import os
import torch
import PIL.Image
import numpy as np
import os.path as osp

from torch.utils.data import Dataset

try:
    lanczos = PIL.Image.Resampling.LANCZOS
except AttributeError:
    lanczos = PIL.Image.LANCZOS


class DemoDataset(Dataset):
    def __init__(self, ROOT, resolution=(518, 392), **kwargs):
        """
        Args:
            ROOT (str): Directory containing image files.
            resolution (tuple): Target resolution (W, H).
        """
        self.ROOT = ROOT
        self.resolution = resolution

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        views = self._get_views()

        if not views:
            raise ValueError(f"No valid images found in {self.ROOT}")

        images = [torch.from_numpy(np.asarray(v['img'])).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0 for v in views]
        camera_poses = [torch.from_numpy(v['camera_pose']) for v in views]
        camera_intrinsics = [torch.from_numpy(v['camera_intrinsics']) for v in views]

        views_all = {
            'images': torch.stack(images),
            'camera_pose': torch.stack(camera_poses), # Placeholder
            'camera_intrinsics': torch.stack(camera_intrinsics), # Placeholder
        }

        return views, views_all

    def _get_views(self):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = sorted([f for f in os.listdir(self.ROOT)
                              if f.lower().endswith(valid_extensions)])

        if not image_files:
            print(f"Warning: No images found in {self.ROOT}")
            return []

        views = []
        num_images = len(image_files)

        for i, image_name in enumerate(image_files):
            print(f"Loading image {i+1}/{num_images} from {self.ROOT}...")
            image_path = osp.join(self.ROOT, image_name)

            image = PIL.Image.open(image_path).convert('RGB')
            image, intrinsics = self._center_crop_and_resize(image, self.resolution)

            views.append(dict(
                img=np.asarray(image),
                camera_pose=np.eye(4, dtype=np.float32),
                camera_intrinsics=intrinsics,
                dataset='demo',
                label=self.ROOT,
                instance=image_name,
            ))

        return views

    @staticmethod
    def _center_crop_and_resize(image, resolution):
        """
        Center-crop the image to match the target aspect ratio, then resize
        to the exact target resolution using Lanczos interpolation.

        This is equivalent to the previous _crop_resize_if_necessary when the
        principal point is at the image center and depthmap is None:
          1. First crop (centered on principal point) was a no-op.
          2. rescale_image_depthmap scaled so the image covers the target.
          3. camera_matrix_of_crop + bbox_from_intrinsics_in_out + crop_image_depthmap
             performed a centered crop to the exact target size.
        The net effect is center crop to target aspect ratio, then resize.

        Args:
            image: PIL Image.
            resolution: (target_W, target_H) tuple.
        Returns:
            (image, intrinsics) where image is a PIL Image of size resolution
            and intrinsics is a 3x3 float32 numpy array.
        """
        W, H = image.size
        target_W, target_H = resolution

        # Handle portrait images: swap target if image is portrait
        if target_W >= target_H:
            if H > 1.1 * W:
                target_W, target_H = target_H, target_W

        target_aspect = target_W / target_H
        image_aspect = W / H

        # Center crop to match the target aspect ratio
        if image_aspect > target_aspect:
            # Image is wider -> crop width
            new_W = int(round(H * target_aspect))
            left = (W - new_W) // 2
            image = image.crop((left, 0, left + new_W, H))
        elif image_aspect < target_aspect:
            # Image is taller -> crop height
            new_H = int(round(W / target_aspect))
            top = (H - new_H) // 2
            image = image.crop((0, top, W, top + new_H))

        # Resize to exact target resolution
        image = image.resize((target_W, target_H), resample=lanczos)
        if image_aspect > target_aspect:
            crop_W = int(round(H * target_aspect))
            scale = target_W / crop_W
        elif image_aspect < target_aspect:
            crop_H = int(round(W / target_aspect))
            scale = target_H / crop_H
        else:
            scale = target_W / W

        fx = fy = W * scale  # W_orig * scale (original fx = fy = W_orig)
        cx, cy = target_W / 2.0, target_H / 2.0

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return image, intrinsics