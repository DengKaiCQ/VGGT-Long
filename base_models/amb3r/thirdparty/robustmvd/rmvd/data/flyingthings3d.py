# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/flyingthings3d.py

import os.path as osp
import re
import itertools
from glob import glob
import pickle

from tqdm import tqdm
import numpy as np
from PIL import Image

from .dataset import Dataset, Sample, _get_sample_list_path
from .registry import register_default_dataset
from .layouts import MVDUnstructuredDefaultLayout, AllImagesLayout


# SUBSET_FILTERED_SAMPLES are the samples that are not included in the official " DispNet/FlowNet2.0 dataset subsets"
SUBSET_FILTERED_SAMPLES = [['TEST/A/0005', '0006'], ['TEST/A/0005', '0007'], ['TEST/A/0005', '0008'], ['TEST/A/0005', '0009'], ['TEST/A/0005', '0010'], ['TEST/A/0005', '0011'], ['TEST/A/0005', '0012'], ['TEST/A/0005', '0013'], ['TEST/A/0005', '0014'], ['TEST/A/0005', '0015'], ['TEST/A/0008', '0006'], ['TEST/A/0008', '0007'], ['TEST/A/0008', '0008'], ['TEST/A/0008', '0009'], ['TEST/A/0008', '0010'], ['TEST/A/0008', '0011'], ['TEST/A/0008', '0012'], ['TEST/A/0008', '0013'], ['TEST/A/0008', '0014'], ['TEST/A/0008', '0015'], ['TEST/A/0013', '0015'], ['TEST/A/0031', '0006'], ['TEST/A/0031', '0007'], ['TEST/A/0031', '0008'], ['TEST/A/0031', '0009'], ['TEST/A/0031', '0010'], ['TEST/A/0031', '0011'], ['TEST/A/0031', '0012'], ['TEST/A/0031', '0013'], ['TEST/A/0031', '0014'], ['TEST/A/0031', '0015'], ['TEST/A/0110', '0006'], ['TEST/A/0110', '0007'], ['TEST/A/0110', '0008'], ['TEST/A/0110', '0009'], ['TEST/A/0110', '0010'], ['TEST/A/0110', '0011'], ['TEST/A/0110', '0012'], ['TEST/A/0110', '0013'], ['TEST/A/0110', '0014'], ['TEST/A/0110', '0015'], ['TEST/A/0123', '0006'], ['TEST/A/0123', '0007'], ['TEST/A/0123', '0008'], ['TEST/A/0123', '0009'], ['TEST/A/0123', '0010'], ['TEST/A/0123', '0011'], ['TEST/A/0123', '0012'], ['TEST/A/0123', '0013'], ['TEST/A/0123', '0014'], ['TEST/A/0123', '0015'], ['TEST/A/0149', '0006'], ['TEST/A/0149', '0007'], ['TEST/A/0149', '0008'], ['TEST/A/0149', '0009'], ['TEST/A/0149', '0010'], ['TEST/A/0149', '0011'], ['TEST/A/0149', '0012'], ['TEST/A/0149', '0013'], ['TEST/A/0149', '0014'], ['TEST/A/0149', '0015'], ['TEST/B/0046', '0006'], ['TEST/B/0046', '0007'], ['TEST/B/0046', '0008'], ['TEST/B/0046', '0009'], ['TEST/B/0046', '0010'], ['TEST/B/0046', '0011'], ['TEST/B/0046', '0012'], ['TEST/B/0046', '0013'], ['TEST/B/0046', '0014'], ['TEST/B/0046', '0015'], ['TEST/B/0048', '0006'], ['TEST/B/0048', '0007'], ['TEST/B/0048', '0008'], ['TEST/B/0048', '0009'], ['TEST/B/0048', '0010'], ['TEST/B/0048', '0011'], ['TEST/B/0048', '0012'], ['TEST/B/0048', '0013'], ['TEST/B/0048', '0014'], ['TEST/B/0048', '0015'], ['TEST/B/0074', '0006'], ['TEST/B/0074', '0007'], ['TEST/B/0074', '0008'], ['TEST/B/0074', '0009'], ['TEST/B/0074', '0010'], ['TEST/B/0074', '0011'], ['TEST/B/0074', '0012'], ['TEST/B/0074', '0013'], ['TEST/B/0074', '0014'], ['TEST/B/0074', '0015'], ['TEST/B/0078', '0006'], ['TEST/B/0078', '0007'], ['TEST/B/0078', '0008'], ['TEST/B/0078', '0009'], ['TEST/B/0078', '0010'], ['TEST/B/0078', '0011'], ['TEST/B/0078', '0012'], ['TEST/B/0078', '0013'], ['TEST/B/0078', '0014'], ['TEST/B/0078', '0015'], ['TEST/B/0133', '0006'], ['TEST/B/0136', '0006'], ['TEST/B/0136', '0007'], ['TEST/B/0136', '0008'], ['TEST/B/0136', '0009'], ['TEST/B/0136', '0010'], ['TEST/B/0136', '0011'], ['TEST/B/0136', '0012'], ['TEST/B/0136', '0013'], ['TEST/B/0136', '0014'], ['TEST/B/0136', '0015'], ['TEST/B/0138', '0006'], ['TEST/B/0138', '0007'], ['TEST/B/0138', '0008'], ['TEST/B/0138', '0009'], ['TEST/B/0138', '0010'], ['TEST/B/0138', '0011'], ['TEST/B/0138', '0012'], ['TEST/B/0138', '0013'], ['TEST/B/0138', '0014'], ['TEST/B/0138', '0015'], ['TRAIN/A/0011', '0015'], ['TRAIN/A/0026', '0006'], ['TRAIN/A/0026', '0007'], ['TRAIN/A/0026', '0008'], ['TRAIN/A/0026', '0009'], ['TRAIN/A/0026', '0010'], ['TRAIN/A/0026', '0011'], ['TRAIN/A/0026', '0012'], ['TRAIN/A/0026', '0013'], ['TRAIN/A/0026', '0014'], ['TRAIN/A/0026', '0015'], ['TRAIN/A/0035', '0006'], ['TRAIN/A/0035', '0007'], ['TRAIN/A/0035', '0008'], ['TRAIN/A/0035', '0009'], ['TRAIN/A/0035', '0010'], ['TRAIN/A/0035', '0011'], ['TRAIN/A/0035', '0012'], ['TRAIN/A/0035', '0013'], ['TRAIN/A/0035', '0014'], ['TRAIN/A/0035', '0015'], ['TRAIN/A/0047', '0006'], ['TRAIN/A/0047', '0007'], ['TRAIN/A/0047', '0008'], ['TRAIN/A/0047', '0009'], ['TRAIN/A/0047', '0010'], ['TRAIN/A/0047', '0011'], ['TRAIN/A/0047', '0012'], ['TRAIN/A/0047', '0013'], ['TRAIN/A/0047', '0014'], ['TRAIN/A/0047', '0015'], ['TRAIN/A/0057', '0006'], ['TRAIN/A/0057', '0007'], ['TRAIN/A/0057', '0008'], ['TRAIN/A/0057', '0009'], ['TRAIN/A/0057', '0010'], ['TRAIN/A/0057', '0011'], ['TRAIN/A/0057', '0012'], ['TRAIN/A/0057', '0013'], ['TRAIN/A/0057', '0014'], ['TRAIN/A/0057', '0015'], ['TRAIN/A/0060', '0006'], ['TRAIN/A/0060', '0007'], ['TRAIN/A/0060', '0008'], ['TRAIN/A/0060', '0009'], ['TRAIN/A/0060', '0010'], ['TRAIN/A/0060', '0011'], ['TRAIN/A/0060', '0012'], ['TRAIN/A/0060', '0013'], ['TRAIN/A/0060', '0014'], ['TRAIN/A/0060', '0015'], ['TRAIN/A/0074', '0006'], ['TRAIN/A/0074', '0007'], ['TRAIN/A/0074', '0008'], ['TRAIN/A/0074', '0009'], ['TRAIN/A/0074', '0010'], ['TRAIN/A/0074', '0011'], ['TRAIN/A/0074', '0012'], ['TRAIN/A/0074', '0013'], ['TRAIN/A/0074', '0014'], ['TRAIN/A/0074', '0015'], ['TRAIN/A/0091', '0006'], ['TRAIN/A/0091', '0007'], ['TRAIN/A/0091', '0008'], ['TRAIN/A/0091', '0009'], ['TRAIN/A/0091', '0010'], ['TRAIN/A/0091', '0011'], ['TRAIN/A/0091', '0012'], ['TRAIN/A/0091', '0013'], ['TRAIN/A/0091', '0014'], ['TRAIN/A/0091', '0015'], ['TRAIN/A/0106', '0006'], ['TRAIN/A/0106', '0007'], ['TRAIN/A/0106', '0008'], ['TRAIN/A/0106', '0009'], ['TRAIN/A/0106', '0010'], ['TRAIN/A/0106', '0011'], ['TRAIN/A/0106', '0012'], ['TRAIN/A/0106', '0013'], ['TRAIN/A/0106', '0014'], ['TRAIN/A/0106', '0015'], ['TRAIN/A/0109', '0006'], ['TRAIN/A/0109', '0007'], ['TRAIN/A/0109', '0008'], ['TRAIN/A/0109', '0009'], ['TRAIN/A/0109', '0010'], ['TRAIN/A/0109', '0011'], ['TRAIN/A/0109', '0012'], ['TRAIN/A/0109', '0013'], ['TRAIN/A/0109', '0014'], ['TRAIN/A/0109', '0015'], ['TRAIN/A/0116', '0006'], ['TRAIN/A/0116', '0007'], ['TRAIN/A/0116', '0008'], ['TRAIN/A/0116', '0009'], ['TRAIN/A/0116', '0010'], ['TRAIN/A/0116', '0011'], ['TRAIN/A/0116', '0012'], ['TRAIN/A/0116', '0013'], ['TRAIN/A/0116', '0014'], ['TRAIN/A/0116', '0015'], ['TRAIN/A/0134', '0006'], ['TRAIN/A/0134', '0007'], ['TRAIN/A/0134', '0008'], ['TRAIN/A/0134', '0009'], ['TRAIN/A/0134', '0010'], ['TRAIN/A/0134', '0011'], ['TRAIN/A/0134', '0012'], ['TRAIN/A/0134', '0013'], ['TRAIN/A/0134', '0014'], ['TRAIN/A/0134', '0015'], ['TRAIN/A/0168', '0006'], ['TRAIN/A/0168', '0007'], ['TRAIN/A/0168', '0008'], ['TRAIN/A/0168', '0009'], ['TRAIN/A/0168', '0010'], ['TRAIN/A/0168', '0011'], ['TRAIN/A/0168', '0012'], ['TRAIN/A/0168', '0013'], ['TRAIN/A/0168', '0014'], ['TRAIN/A/0168', '0015'], ['TRAIN/A/0169', '0006'], ['TRAIN/A/0179', '0006'], ['TRAIN/A/0190', '0006'], ['TRAIN/A/0190', '0007'], ['TRAIN/A/0190', '0008'], ['TRAIN/A/0190', '0009'], ['TRAIN/A/0190', '0010'], ['TRAIN/A/0190', '0011'], ['TRAIN/A/0190', '0012'], ['TRAIN/A/0190', '0013'], ['TRAIN/A/0190', '0014'], ['TRAIN/A/0190', '0015'], ['TRAIN/A/0201', '0006'], ['TRAIN/A/0201', '0007'], ['TRAIN/A/0201', '0008'], ['TRAIN/A/0201', '0009'], ['TRAIN/A/0201', '0010'], ['TRAIN/A/0201', '0011'], ['TRAIN/A/0201', '0012'], ['TRAIN/A/0201', '0013'], ['TRAIN/A/0201', '0014'], ['TRAIN/A/0201', '0015'], ['TRAIN/A/0204', '0006'], ['TRAIN/A/0204', '0007'], ['TRAIN/A/0204', '0008'], ['TRAIN/A/0204', '0009'], ['TRAIN/A/0204', '0010'], ['TRAIN/A/0204', '0011'], ['TRAIN/A/0204', '0012'], ['TRAIN/A/0204', '0013'], ['TRAIN/A/0204', '0014'], ['TRAIN/A/0204', '0015'], ['TRAIN/A/0236', '0015'], ['TRAIN/A/0244', '0006'], ['TRAIN/A/0244', '0007'], ['TRAIN/A/0244', '0008'], ['TRAIN/A/0244', '0009'], ['TRAIN/A/0244', '0010'], ['TRAIN/A/0244', '0011'], ['TRAIN/A/0244', '0012'], ['TRAIN/A/0244', '0013'], ['TRAIN/A/0244', '0014'], ['TRAIN/A/0244', '0015'], ['TRAIN/A/0286', '0015'], ['TRAIN/A/0295', '0006'], ['TRAIN/A/0295', '0007'], ['TRAIN/A/0295', '0008'], ['TRAIN/A/0295', '0009'], ['TRAIN/A/0295', '0010'], ['TRAIN/A/0295', '0011'], ['TRAIN/A/0295', '0012'], ['TRAIN/A/0295', '0013'], ['TRAIN/A/0295', '0014'], ['TRAIN/A/0295', '0015'], ['TRAIN/A/0352', '0006'], ['TRAIN/A/0352', '0007'], ['TRAIN/A/0352', '0008'], ['TRAIN/A/0352', '0009'], ['TRAIN/A/0352', '0010'], ['TRAIN/A/0352', '0011'], ['TRAIN/A/0352', '0012'], ['TRAIN/A/0352', '0013'], ['TRAIN/A/0352', '0014'], ['TRAIN/A/0352', '0015'], ['TRAIN/A/0357', '0006'], ['TRAIN/A/0357', '0007'], ['TRAIN/A/0357', '0008'], ['TRAIN/A/0357', '0009'], ['TRAIN/A/0357', '0010'], ['TRAIN/A/0357', '0011'], ['TRAIN/A/0357', '0012'], ['TRAIN/A/0357', '0013'], ['TRAIN/A/0357', '0014'], ['TRAIN/A/0357', '0015'], ['TRAIN/A/0364', '0006'], ['TRAIN/A/0364', '0007'], ['TRAIN/A/0364', '0008'], ['TRAIN/A/0364', '0009'], ['TRAIN/A/0364', '0010'], ['TRAIN/A/0364', '0011'], ['TRAIN/A/0364', '0012'], ['TRAIN/A/0364', '0013'], ['TRAIN/A/0364', '0014'], ['TRAIN/A/0364', '0015'], ['TRAIN/A/0369', '0006'], ['TRAIN/A/0369', '0007'], ['TRAIN/A/0369', '0008'], ['TRAIN/A/0369', '0009'], ['TRAIN/A/0369', '0010'], ['TRAIN/A/0369', '0011'], ['TRAIN/A/0369', '0012'], ['TRAIN/A/0369', '0013'], ['TRAIN/A/0369', '0014'], ['TRAIN/A/0369', '0015'], ['TRAIN/A/0373', '0015'], ['TRAIN/A/0398', '0006'], ['TRAIN/A/0398', '0007'], ['TRAIN/A/0398', '0008'], ['TRAIN/A/0398', '0009'], ['TRAIN/A/0398', '0010'], ['TRAIN/A/0398', '0011'], ['TRAIN/A/0398', '0012'], ['TRAIN/A/0398', '0013'], ['TRAIN/A/0398', '0014'], ['TRAIN/A/0398', '0015'], ['TRAIN/A/0433', '0006'], ['TRAIN/A/0433', '0007'], ['TRAIN/A/0433', '0008'], ['TRAIN/A/0433', '0009'], ['TRAIN/A/0433', '0010'], ['TRAIN/A/0433', '0011'], ['TRAIN/A/0433', '0012'], ['TRAIN/A/0433', '0013'], ['TRAIN/A/0433', '0014'], ['TRAIN/A/0433', '0015'], ['TRAIN/A/0475', '0006'], ['TRAIN/A/0518', '0006'], ['TRAIN/A/0518', '0007'], ['TRAIN/A/0518', '0008'], ['TRAIN/A/0518', '0009'], ['TRAIN/A/0518', '0010'], ['TRAIN/A/0518', '0011'], ['TRAIN/A/0518', '0012'], ['TRAIN/A/0518', '0013'], ['TRAIN/A/0518', '0014'], ['TRAIN/A/0518', '0015'], ['TRAIN/A/0521', '0006'], ['TRAIN/A/0521', '0007'], ['TRAIN/A/0521', '0008'], ['TRAIN/A/0521', '0009'], ['TRAIN/A/0521', '0010'], ['TRAIN/A/0521', '0011'], ['TRAIN/A/0521', '0012'], ['TRAIN/A/0521', '0013'], ['TRAIN/A/0521', '0014'], ['TRAIN/A/0521', '0015'], ['TRAIN/A/0588', '0006'], ['TRAIN/A/0588', '0007'], ['TRAIN/A/0588', '0008'], ['TRAIN/A/0588', '0009'], ['TRAIN/A/0588', '0010'], ['TRAIN/A/0588', '0011'], ['TRAIN/A/0588', '0012'], ['TRAIN/A/0588', '0013'], ['TRAIN/A/0588', '0014'], ['TRAIN/A/0588', '0015'], ['TRAIN/A/0603', '0006'], ['TRAIN/A/0603', '0007'], ['TRAIN/A/0603', '0008'], ['TRAIN/A/0603', '0009'], ['TRAIN/A/0603', '0010'], ['TRAIN/A/0603', '0011'], ['TRAIN/A/0603', '0012'], ['TRAIN/A/0603', '0013'], ['TRAIN/A/0603', '0014'], ['TRAIN/A/0603', '0015'], ['TRAIN/A/0632', '0006'], ['TRAIN/A/0634', '0006'], ['TRAIN/A/0634', '0007'], ['TRAIN/A/0634', '0008'], ['TRAIN/A/0634', '0009'], ['TRAIN/A/0634', '0010'], ['TRAIN/A/0634', '0011'], ['TRAIN/A/0634', '0012'], ['TRAIN/A/0634', '0013'], ['TRAIN/A/0634', '0014'], ['TRAIN/A/0634', '0015'], ['TRAIN/A/0643', '0006'], ['TRAIN/A/0643', '0007'], ['TRAIN/A/0643', '0008'], ['TRAIN/A/0643', '0009'], ['TRAIN/A/0643', '0010'], ['TRAIN/A/0643', '0011'], ['TRAIN/A/0643', '0012'], ['TRAIN/A/0643', '0013'], ['TRAIN/A/0643', '0014'], ['TRAIN/A/0643', '0015'], ['TRAIN/A/0658', '0006'], ['TRAIN/A/0658', '0007'], ['TRAIN/A/0658', '0008'], ['TRAIN/A/0658', '0009'], ['TRAIN/A/0658', '0010'], ['TRAIN/A/0658', '0011'], ['TRAIN/A/0658', '0012'], ['TRAIN/A/0658', '0013'], ['TRAIN/A/0658', '0014'], ['TRAIN/A/0658', '0015'], ['TRAIN/A/0705', '0006'], ['TRAIN/A/0705', '0007'], ['TRAIN/A/0705', '0008'], ['TRAIN/A/0705', '0009'], ['TRAIN/A/0705', '0010'], ['TRAIN/A/0705', '0011'], ['TRAIN/A/0705', '0012'], ['TRAIN/A/0705', '0013'], ['TRAIN/A/0705', '0014'], ['TRAIN/A/0705', '0015'], ['TRAIN/A/0741', '0006'], ['TRAIN/A/0741', '0007'], ['TRAIN/A/0741', '0008'], ['TRAIN/A/0741', '0009'], ['TRAIN/A/0741', '0010'], ['TRAIN/A/0741', '0011'], ['TRAIN/A/0741', '0012'], ['TRAIN/A/0741', '0013'], ['TRAIN/A/0741', '0014'], ['TRAIN/A/0741', '0015'], ['TRAIN/B/0007', '0006'], ['TRAIN/B/0007', '0007'], ['TRAIN/B/0007', '0008'], ['TRAIN/B/0007', '0009'], ['TRAIN/B/0007', '0010'], ['TRAIN/B/0007', '0011'], ['TRAIN/B/0007', '0012'], ['TRAIN/B/0007', '0013'], ['TRAIN/B/0007', '0014'], ['TRAIN/B/0007', '0015'], ['TRAIN/B/0021', '0006'], ['TRAIN/B/0021', '0007'], ['TRAIN/B/0021', '0008'], ['TRAIN/B/0021', '0009'], ['TRAIN/B/0021', '0010'], ['TRAIN/B/0021', '0011'], ['TRAIN/B/0021', '0012'], ['TRAIN/B/0021', '0013'], ['TRAIN/B/0021', '0014'], ['TRAIN/B/0021', '0015'], ['TRAIN/B/0022', '0006'], ['TRAIN/B/0022', '0007'], ['TRAIN/B/0022', '0008'], ['TRAIN/B/0022', '0009'], ['TRAIN/B/0022', '0010'], ['TRAIN/B/0022', '0011'], ['TRAIN/B/0022', '0012'], ['TRAIN/B/0022', '0013'], ['TRAIN/B/0022', '0014'], ['TRAIN/B/0022', '0015'], ['TRAIN/B/0051', '0006'], ['TRAIN/B/0051', '0007'], ['TRAIN/B/0051', '0008'], ['TRAIN/B/0051', '0009'], ['TRAIN/B/0051', '0010'], ['TRAIN/B/0051', '0011'], ['TRAIN/B/0051', '0012'], ['TRAIN/B/0051', '0013'], ['TRAIN/B/0051', '0014'], ['TRAIN/B/0051', '0015'], ['TRAIN/B/0053', '0006'], ['TRAIN/B/0053', '0007'], ['TRAIN/B/0053', '0008'], ['TRAIN/B/0053', '0009'], ['TRAIN/B/0053', '0010'], ['TRAIN/B/0053', '0011'], ['TRAIN/B/0053', '0012'], ['TRAIN/B/0053', '0013'], ['TRAIN/B/0053', '0014'], ['TRAIN/B/0053', '0015'], ['TRAIN/B/0075', '0006'], ['TRAIN/B/0080', '0006'], ['TRAIN/B/0080', '0007'], ['TRAIN/B/0080', '0008'], ['TRAIN/B/0080', '0009'], ['TRAIN/B/0080', '0010'], ['TRAIN/B/0080', '0011'], ['TRAIN/B/0080', '0012'], ['TRAIN/B/0080', '0013'], ['TRAIN/B/0080', '0014'], ['TRAIN/B/0080', '0015'], ['TRAIN/B/0189', '0006'], ['TRAIN/B/0189', '0007'], ['TRAIN/B/0189', '0008'], ['TRAIN/B/0189', '0009'], ['TRAIN/B/0189', '0010'], ['TRAIN/B/0189', '0011'], ['TRAIN/B/0189', '0012'], ['TRAIN/B/0189', '0013'], ['TRAIN/B/0189', '0014'], ['TRAIN/B/0189', '0015'], ['TRAIN/B/0256', '0006'], ['TRAIN/B/0256', '0007'], ['TRAIN/B/0256', '0008'], ['TRAIN/B/0256', '0009'], ['TRAIN/B/0256', '0010'], ['TRAIN/B/0256', '0011'], ['TRAIN/B/0256', '0012'], ['TRAIN/B/0256', '0013'], ['TRAIN/B/0256', '0014'], ['TRAIN/B/0256', '0015'], ['TRAIN/B/0282', '0006'], ['TRAIN/B/0282', '0007'], ['TRAIN/B/0282', '0008'], ['TRAIN/B/0282', '0009'], ['TRAIN/B/0282', '0010'], ['TRAIN/B/0282', '0011'], ['TRAIN/B/0282', '0012'], ['TRAIN/B/0282', '0013'], ['TRAIN/B/0282', '0014'], ['TRAIN/B/0282', '0015'], ['TRAIN/B/0284', '0006'], ['TRAIN/B/0327', '0006'], ['TRAIN/B/0327', '0007'], ['TRAIN/B/0327', '0008'], ['TRAIN/B/0327', '0009'], ['TRAIN/B/0327', '0010'], ['TRAIN/B/0327', '0011'], ['TRAIN/B/0327', '0012'], ['TRAIN/B/0327', '0013'], ['TRAIN/B/0327', '0014'], ['TRAIN/B/0327', '0015'], ['TRAIN/B/0351', '0006'], ['TRAIN/B/0351', '0007'], ['TRAIN/B/0351', '0008'], ['TRAIN/B/0351', '0009'], ['TRAIN/B/0351', '0010'], ['TRAIN/B/0351', '0011'], ['TRAIN/B/0351', '0012'], ['TRAIN/B/0351', '0013'], ['TRAIN/B/0351', '0014'], ['TRAIN/B/0351', '0015'], ['TRAIN/B/0360', '0006'], ['TRAIN/B/0360', '0007'], ['TRAIN/B/0360', '0008'], ['TRAIN/B/0360', '0009'], ['TRAIN/B/0360', '0010'], ['TRAIN/B/0360', '0011'], ['TRAIN/B/0360', '0012'], ['TRAIN/B/0360', '0013'], ['TRAIN/B/0360', '0014'], ['TRAIN/B/0360', '0015'], ['TRAIN/B/0368', '0006'], ['TRAIN/B/0368', '0007'], ['TRAIN/B/0368', '0008'], ['TRAIN/B/0368', '0009'], ['TRAIN/B/0368', '0010'], ['TRAIN/B/0368', '0011'], ['TRAIN/B/0368', '0012'], ['TRAIN/B/0368', '0013'], ['TRAIN/B/0368', '0014'], ['TRAIN/B/0368', '0015'], ['TRAIN/B/0381', '0006'], ['TRAIN/B/0381', '0007'], ['TRAIN/B/0381', '0008'], ['TRAIN/B/0381', '0009'], ['TRAIN/B/0381', '0010'], ['TRAIN/B/0381', '0011'], ['TRAIN/B/0381', '0012'], ['TRAIN/B/0381', '0013'], ['TRAIN/B/0381', '0014'], ['TRAIN/B/0381', '0015'], ['TRAIN/B/0424', '0006'], ['TRAIN/B/0424', '0007'], ['TRAIN/B/0424', '0008'], ['TRAIN/B/0424', '0009'], ['TRAIN/B/0424', '0010'], ['TRAIN/B/0424', '0011'], ['TRAIN/B/0424', '0012'], ['TRAIN/B/0424', '0013'], ['TRAIN/B/0424', '0014'], ['TRAIN/B/0424', '0015'], ['TRAIN/B/0501', '0006'], ['TRAIN/B/0501', '0007'], ['TRAIN/B/0501', '0008'], ['TRAIN/B/0501', '0009'], ['TRAIN/B/0501', '0010'], ['TRAIN/B/0501', '0011'], ['TRAIN/B/0501', '0012'], ['TRAIN/B/0501', '0013'], ['TRAIN/B/0501', '0014'], ['TRAIN/B/0501', '0015'], ['TRAIN/B/0549', '0015'], ['TRAIN/B/0578', '0006'], ['TRAIN/B/0578', '0007'], ['TRAIN/B/0578', '0008'], ['TRAIN/B/0578', '0009'], ['TRAIN/B/0578', '0010'], ['TRAIN/B/0578', '0011'], ['TRAIN/B/0578', '0012'], ['TRAIN/B/0578', '0013'], ['TRAIN/B/0578', '0014'], ['TRAIN/B/0578', '0015'], ['TRAIN/B/0609', '0006'], ['TRAIN/B/0609', '0007'], ['TRAIN/B/0609', '0008'], ['TRAIN/B/0609', '0009'], ['TRAIN/B/0609', '0010'], ['TRAIN/B/0609', '0011'], ['TRAIN/B/0609', '0012'], ['TRAIN/B/0609', '0013'], ['TRAIN/B/0609', '0014'], ['TRAIN/B/0609', '0015'], ['TRAIN/B/0623', '0006'], ['TRAIN/B/0623', '0007'], ['TRAIN/B/0623', '0008'], ['TRAIN/B/0623', '0009'], ['TRAIN/B/0623', '0010'], ['TRAIN/B/0623', '0011'], ['TRAIN/B/0623', '0012'], ['TRAIN/B/0623', '0013'], ['TRAIN/B/0623', '0014'], ['TRAIN/B/0623', '0015'], ['TRAIN/B/0653', '0006'], ['TRAIN/B/0653', '0007'], ['TRAIN/B/0653', '0008'], ['TRAIN/B/0653', '0009'], ['TRAIN/B/0653', '0010'], ['TRAIN/B/0653', '0011'], ['TRAIN/B/0653', '0012'], ['TRAIN/B/0653', '0013'], ['TRAIN/B/0653', '0014'], ['TRAIN/B/0653', '0015'], ['TRAIN/B/0668', '0006'], ['TRAIN/B/0668', '0007'], ['TRAIN/B/0668', '0008'], ['TRAIN/B/0668', '0009'], ['TRAIN/B/0668', '0010'], ['TRAIN/B/0668', '0011'], ['TRAIN/B/0668', '0012'], ['TRAIN/B/0668', '0013'], ['TRAIN/B/0668', '0014'], ['TRAIN/B/0668', '0015'], ['TRAIN/B/0688', '0006'], ['TRAIN/B/0688', '0007'], ['TRAIN/B/0688', '0008'], ['TRAIN/B/0688', '0009'], ['TRAIN/B/0688', '0010'], ['TRAIN/B/0688', '0011'], ['TRAIN/B/0688', '0012'], ['TRAIN/B/0688', '0013'], ['TRAIN/B/0688', '0014'], ['TRAIN/B/0688', '0015'], ['TRAIN/B/0697', '0006'], ['TRAIN/B/0697', '0007'], ['TRAIN/B/0697', '0008'], ['TRAIN/B/0697', '0009'], ['TRAIN/B/0697', '0010'], ['TRAIN/B/0697', '0011'], ['TRAIN/B/0697', '0012'], ['TRAIN/B/0697', '0013'], ['TRAIN/B/0697', '0014'], ['TRAIN/B/0697', '0015'], ['TRAIN/B/0723', '0006'], ['TRAIN/B/0727', '0006'], ['TRAIN/B/0727', '0007'], ['TRAIN/B/0727', '0008'], ['TRAIN/B/0727', '0009'], ['TRAIN/B/0727', '0010'], ['TRAIN/B/0727', '0011'], ['TRAIN/B/0727', '0012'], ['TRAIN/B/0727', '0013'], ['TRAIN/B/0727', '0014'], ['TRAIN/B/0727', '0015']]

# some additional samples that are hard (determined using some heuristics):
HARD_SAMPLES = [['TRAIN/A/0057', '0006'], ['TRAIN/A/0391', '0012'], ['TRAIN/A/0542', '0009'], ['TRAIN/B/0318', '0006'], ['TRAIN/B/0318', '0007'], ['TRAIN/B/0318', '0008'], ['TRAIN/B/0318', '0009'], ['TRAIN/B/0318', '0010'], ['TRAIN/B/0318', '0011'], ['TRAIN/B/0318', '0012'], ['TRAIN/B/0318', '0013'], ['TRAIN/B/0318', '0014'], ['TRAIN/B/0318', '0015']]


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


class DataConf:
    def __init__(self, id, perspective=None, offset=0):
        self.id = id
        self.perspective = perspective
        self.offset = offset

    @property
    def ext(self):
        if self.id == 'frames_cleanpass' or self.id == 'frames_finalpass':
            return 'png'
        elif self.id == 'poses' or self.id == 'intrinsics':
            return 'npy'
        elif self.id == 'disparities':
            return 'pfm'
        
    @property
    def perspective_short(self):
        if self.perspective is None:
            return None
        else:
            return self.perspective[0]

    @property
    def path(self):
        if self.perspective is None:
            return self.id
        else:
            return osp.join(self.id, self.perspective)

    @property
    def glob(self):
        if self.perspective is None:
            return osp.join(self.id, "*.{}".format(self.ext))
        else:
            return osp.join(self.id, self.perspective, "*.{}".format(self.ext))


def load_image(root, cam, frame_num):
    filename = '{:04d}.png'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    img = np.array(Image.open(osp.join(root, "frames_cleanpass", cam, filename)))
    img = img.transpose([2, 0, 1]).astype(np.float32)
    return img  # 3, H, W, float32


def load_depth(root, cam, frame_num):
    filename = '{:04d}.pfm'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    disparity = readPFM(osp.join(root, "disparities", cam, filename))
    depth = 1050./(-1*disparity)
    depth[(depth < 0.) | np.isinf(depth) | np.isnan(depth)] = 0.
    depth = np.expand_dims(depth, 0).astype(np.float32)  # 1HW
    return depth


def load_intrinsics(root, cam, frame_num):
    filename = '{:04d}.npy'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    return np.load(osp.join(root, "intrinsics", cam, filename)).astype(np.float32)


def load_pose(root, cam, frame_num):
    filename = '{:04d}.npy'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    return np.load(osp.join(root, "poses", cam, filename)).astype(np.float32)


def load(key, root, val):
    if isinstance(val, list):
        return [load(key, root, v) for v in val]
    else:
        if key == 'images':
            cam, frame_num = val
            return load_image(root, cam, frame_num)
        elif key == 'depth':
            cam, frame_num = val
            return load_depth(root, cam, frame_num)
        elif key == 'intrinsics':
            cam, frame_num = val
            return load_intrinsics(root, cam, frame_num)
        elif key == 'poses':
            cam, frame_num = val
            return load_pose(root, cam, frame_num)
        else:
            return val


class FlyingThings3DSample(Sample):

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


class FlyingThings3D(Dataset):
    def _init_samples(self, sample_confs=None, filter_hard_samples=False, use_subset_only=False):
        sample_list_path = _get_sample_list_path(self.name)
        if sample_list_path is not None and osp.isfile(sample_list_path):
            super()._init_samples_from_list()
        else:
            self._init_samples_from_confs(sample_confs=sample_confs, filter_hard_samples=filter_hard_samples, use_subset_only=use_subset_only)
            self._write_samples_list()
    
    def _init_samples_from_confs(self, sample_confs, filter_hard_samples=False, use_subset_only=False):
        sequences = sorted(glob(osp.join(self.root, '*/*[0-9]')))

        for sequence in (tqdm(sequences) if self.verbose else sequences):
            sequence_files = glob(osp.join(sequence, '*/*/*'))
            sequence_files = [osp.relpath(f, sequence) for f in sequence_files]
            sequence_id = osp.join(osp.split(self.root)[1], osp.relpath(sequence, self.root))
            for sample_conf in sample_confs:
                for keyframe_num in range(6, 16):  # build samples for using frames 6 to 15 as keyframes
                    sample = FlyingThings3DSample(base=osp.relpath(sequence, self.root),
                                                name=osp.relpath(sequence, self.root) + "/key{:02d}".format(keyframe_num))

                    sample_valid = True
                    for key, conf in sample_conf.items():

                        if not (isinstance(conf, DataConf) or isinstance(conf, list)):
                            sample.data[key] = conf
                            continue

                        elif isinstance(conf, DataConf):
                            offset_num = keyframe_num + conf.offset
                            filename = f'{offset_num:04d}.{conf.ext}'
                            if osp.join(conf.path, filename) in sequence_files:
                                if not filter_hard_samples or [sequence_id, f'{offset_num:04d}'] not in HARD_SAMPLES:
                                    if not use_subset_only or [sequence_id, f'{offset_num:04d}'] not in SUBSET_FILTERED_SAMPLES:
                                        sample.data[key] = (conf.perspective_short, offset_num)
                                    else:
                                        sample_valid = False
                                        break
                                else:
                                    sample_valid = False
                                    break
                            else:
                                sample_valid = False
                                break

                        elif isinstance(conf, list):
                            confs = conf
                            sample.data[key] = []
                            for conf in confs:
                                offset_num = keyframe_num + conf.offset
                                filename = f'{offset_num:04d}.{conf.ext}'
                                if osp.join(conf.path, filename) in sequence_files:
                                    if not filter_hard_samples or [sequence_id, f'{offset_num:04d}'] not in HARD_SAMPLES:
                                        if not use_subset_only or [sequence_id, f'{offset_num:04d}'] not in SUBSET_FILTERED_SAMPLES:
                                            sample.data[key].append((conf.perspective_short, offset_num))
                                        else:
                                            sample_valid = False
                                            break
                                    else:
                                        sample_valid = False
                                        break
                                else:
                                    sample_valid = False
                                    break

                    if sample_valid:
                        self.samples.append(sample)


@register_default_dataset
class FlyingThings3DSeq4Train(FlyingThings3D):

    base_dataset = 'flyingthings3d'
    split = 'robust_mvd'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("flyingthings3d", "train", "root")
        
        sample_confs = self._get_sample_confs()
        filter_hard_samples = False
        use_subset_only = True

        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=5, max_views=5),
            AllImagesLayout("all_images", num_views=5),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(sample_confs=sample_confs, filter_hard_samples=filter_hard_samples, use_subset_only=use_subset_only, root=root, layouts=layouts, **kwargs)
        
    def _get_sample_confs(self):

        sample_confs = []

        images_key = [DataConf('frames_cleanpass', 'left', 0)]
        to_ref_transforms_base = [DataConf('poses', 'left', 0)]
        intrinsics_base = [DataConf('intrinsics', 'left', 0)]
        offset_list = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

        for offsets in itertools.combinations(offset_list, 4):
            images = images_key.copy()
            to_ref_transforms = to_ref_transforms_base.copy()
            intrinsics = intrinsics_base.copy()

            for offset in offsets:
                images = images + [DataConf('frames_cleanpass', 'left', offset)]
                to_ref_transforms = to_ref_transforms + [DataConf('poses', 'left', offset)]
                intrinsics = intrinsics + [DataConf('intrinsics', 'left', offset)]

            sample_conf = {
                'images': images,
                'poses': to_ref_transforms,
                'intrinsics': intrinsics,
                'depth': DataConf('disparities', 'left', 0),
                'keyview_idx': 0,
            }
            sample_confs.append(sample_conf)

        return sample_confs
