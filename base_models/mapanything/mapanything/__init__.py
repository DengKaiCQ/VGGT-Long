
# References:
#   https://github.com/facebookresearch/map-anything/blob/main/mapanything/__init__.py

import sys
from pathlib import Path

_pkg = sys.modules[__name__]
sys.modules.setdefault("mapanything", _pkg)

_pkg.__path__ = [str(Path(__file__).resolve().parent)]

