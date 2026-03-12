# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/data/batch_augmentations.py

import torchvision.transforms
from .registry import register_batch_augmentation
from .batch_transforms import Scale3DEqualizedBatch, MaskDepth


@register_batch_augmentation
def robust_mvd_batch_augmentations(**kwargs):
    transforms = [
        Scale3DEqualizedBatch(p=1, min_depth=1/2.75, max_depth=1/0.009),
        MaskDepth(min_depth=1/2.75, max_depth=1/0.009),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform
