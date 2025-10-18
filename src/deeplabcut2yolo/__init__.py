# deeplabcut2yolo is dual-licensed under GNU General Public License v3.0 and The 3-Clause BSD License, see LICENSE.
# Copyright 2024 Sira Pornsiriprasert <code@psira.me>

"""
Convert DeepLabCut dataset to YOLO format

Quick Start:
- d2y.convert("./deeplabcut-dataset/")
- d2y.convert("./deeplabcut-dataset/", train_paths, val_paths, skeleton_symmetric_pairs, data_yml_path)
"""

from .__about__ import *
from .deeplabcut2yolo import (
    convert,
    get_flip_idx,
)
