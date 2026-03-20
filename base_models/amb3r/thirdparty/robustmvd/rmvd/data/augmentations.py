# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/augmentations.py

import torchvision.transforms
from .registry import register_augmentation
from .transforms import ResizeInputs, ResizeTargets, ColorJitter, Eraser, NormalizeImagesToMinMax, MaskDepth, SpatialAugmentation, NormalizeIntrinsics


@register_augmentation
def robust_mvd_augmentations_staticthings3d(**kwargs):
    transforms = [
        ColorJitter(saturation=(0, 2), contrast=(0.01, 8), brightness=(0.01, 2.0), hue=0.5),
        SpatialAugmentation(size=(384, 768), p=1.0),
        NormalizeImagesToMinMax(min_val=-0.4, max_val=0.6),
        NormalizeIntrinsics(),
        Eraser(bounds=[250, 500], p=0.6),
        MaskDepth(min_depth=1/2.75, max_depth=1/0.009),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform


@register_augmentation
def robust_mvd_augmentations_blendedmvs(**kwargs):
    transforms = [
        ColorJitter(saturation=(0, 2), contrast=(0.01, 8), brightness=(0.01, 2.0), hue=0.5),
        ResizeInputs(size=(384, 768)),
        ResizeTargets(size=(384, 768)),
        NormalizeImagesToMinMax(min_val=-0.4, max_val=0.6),
        NormalizeIntrinsics(),
        Eraser(bounds=[250, 500], p=0.6),
        # intentionally not masking depth
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform
