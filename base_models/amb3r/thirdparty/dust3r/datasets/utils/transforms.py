# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/dust3r/datasets/utils/transforms.py

import torchvision.transforms as tvf
from dust3r.utils.image import ImgNorm

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
to_numpy = tvf.Lambda(lambda t: t.permute(1, 2, 0).cpu().numpy())
ImgInvNorm = tvf.Compose([tvf.Normalize((-1, -1, -1), (2, 2, 2)), to_numpy])# to numpy array
