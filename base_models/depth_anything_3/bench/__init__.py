# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# References:
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/__init__.py

"""
Depth Anything 3 Benchmark Evaluation Module.

This module provides tools for evaluating DepthAnything3 model on various benchmark datasets.
Currently supported datasets:
- DTU (3D Reconstruction)
- DTU-64 (Pose Evaluation Only)
- ETH3D (3D Reconstruction)
- 7Scenes (3D Reconstruction)
- ScanNet++ (3D Reconstruction)
- HiRoom (3D Reconstruction)

Supported evaluation modes:
- pose: Camera pose estimation evaluation
- recon_unposed: 3D reconstruction with predicted poses
- recon_posed: 3D reconstruction with ground truth poses
"""

from depth_anything_3.bench.registries import MV_REGISTRY, MONO_REGISTRY


def __getattr__(name):
    """Lazy import to avoid circular import when running as __main__."""
    if name == "Evaluator":
        from depth_anything_3.bench.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Evaluator", "MV_REGISTRY", "MONO_REGISTRY"]

