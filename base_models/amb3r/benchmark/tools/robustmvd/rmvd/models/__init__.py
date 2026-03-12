# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/__init__.py

from .robust_mvd import robust_mvd, robust_mvd_5M
from .wrappers.monodepth2 import monodepth2_mono_stereo_1024x320_wrapped, monodepth2_mono_stereo_640x192_wrapped
from .wrappers.mvsnet_pl import mvsnet_pl_wrapped
from .wrappers.midas import midas_big_v2_1_wrapped
from .wrappers.vis_mvsnet import vis_mvsnet_wrapped
from .wrappers.cvp_mvsnet import cvp_mvsnet_wrapped
from .wrappers.patchmatchnet import patchmatchnet_wrapped

from .factory import create_model, prepare_custom_model
from .registry import register_model, list_models, has_model
