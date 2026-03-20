# References:
#   https://github.com/facebookresearch/map-anything/blob/main/mapanything/__init__.py

"""
将仓库内嵌的 MapAnything 暴露为顶级包 `mapanything`，以兼容其内部的绝对导入。
在 `base_models/base_model.py` 中先导入本包后，`from mapanything...` 就能正常解析。
"""

import sys
from pathlib import Path

# 把当前包对象注册为顶级模块别名
_pkg = sys.modules[__name__]
sys.modules.setdefault("mapanything", _pkg)

# 确保包搜索路径正确指向当前目录（防止某些环境下被意外覆盖）
_pkg.__path__ = [str(Path(__file__).resolve().parent)]

