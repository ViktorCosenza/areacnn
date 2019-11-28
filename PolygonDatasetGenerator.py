# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from PIL import Image, ImageStat
from PIL.ImageDraw import ImageDraw

import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from random import randint
from math import pi

import os
from os import path

from fastai import vision
from fastai.vision import ImageList, FloatList, cnn_learner, pil2tensor, open_image
from fastai.callbacks import EarlyStoppingCallback, ReduceLROnPlateauCallback, CSVLogger
from fastai.train import ShowGraph, Learner
from fastai.metrics import mean_absolute_error, mean_squared_error
from torchvision import models
from torch import nn
import numpy as np

from functools import partial

## Local Imports ##
from models import helpers as model_helpers, models as custom_models
from datasets import helpers as dataset_helpers

# + [markdown] colab_type="text" id="QLVyf1-umCV2"
# ## Dataset

# +
DT_LEN = 25000
COUNT = True
ROOT_DIR = 'data'
DT_ROOT_DIR = 'polygon_data_counts' if COUNT else 'polygon_data'
DT_ROOT_DIR = path.join(ROOT_DIR, DT_ROOT_DIR)
DT_ROOT_DIR = path.abspath(DT_ROOT_DIR)
MODEL_ROOT_DIR = path.join(DT_ROOT_DIR, 'results')
W, H = 512, 512
BATCH_SIZE = 64

DATASETS = [
        {"root_dir": path.join(ROOT_DIR, "polygon_data_counts"), "count": True, "skip": True},
        {"root_dir": path.join(ROOT_DIR, "polygon_data_percentage"), "count": False, "skip": True},
        {"root_dir": path.join(ROOT_DIR, "ellipse_data_counts"), "count": True, "skip": False, "draw_polygon_fn":dataset_helpers.draw_ellipse},
        {"root_dir": path.join(ROOT_DIR, "ellipse_data_percentage"), "count": False, "skip": False, "draw_polygon_fn":dataset_helpers.draw_ellipse}
    ]

for dataset in DATASETS:
    df_train = dataset_helpers.gen_df(dt_len=DT_LEN, test=False, **dataset)
    df_test  = dataset_helpers.gen_df(dt_len=DT_LEN//2, test=True, **dataset)
# -


