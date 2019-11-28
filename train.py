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
from torch.utils import data
from torch.nn import functional as F

import torchvision
from torchvision import transforms

from random import randint
from math import pi

import os
from os import path

import numpy as np 
import math
from functools import partial, reduce
import itertools
from collections import namedtuple

from tqdm import tqdm

## Local Imports ##
from models import helpers as model_helpers, models as custom_models
from datasets import helpers as dataset_helpers, datasets as custom_datasets

# +
W, H = (256, 256)
BS = 128
MAX_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DT_ROOT = 'data'
POLYGON_COUNT_DIR = path.join(DT_ROOT, 'polygon_data_counts')
POLYGON_PERCENTAGE_DIR = path.join(DT_ROOT, 'polygon_data_percentage')
ELLIPSE_COUNT_DIR = path.join(DT_ROOT, 'ellipse_data_counts')
ELLIPSE_PERCENTAGE_DIR = path.join(DT_ROOT, 'ellipse_data_percentage')
TRANSFORM = transforms.Compose([transforms.ToTensor()])

dataset_generators = [
    model_helpers.Param('RECT_COUNT', custom_datasets.get_dataset(
        root_dir=POLYGON_COUNT_DIR, 
        df_path=path.join(POLYGON_COUNT_DIR, 'data.csv'),
        transform=TRANSFORM,
        bs=BS
    )),
    model_helpers.Param('RECT_PCT', custom_datasets.get_dataset(
        root_dir=POLYGON_PERCENTAGE_DIR, 
        df_path=path.join(POLYGON_PERCENTAGE_DIR, 'data.csv'),
        transform=TRANSFORM,
        bs=BS
    )),
    model_helpers.Param('ELLIPSE_COUNT', custom_datasets.get_dataset(
        root_dir=ELLIPSE_COUNT_DIR, 
        df_path=path.join(ELLIPSE_COUNT_DIR, 'data.csv'),
        transform=TRANSFORM,
        bs=BS
    )),
    model_helpers.Param('ELLIPSE_PCT', custom_datasets.get_dataset(
        root_dir=ELLIPSE_PERCENTAGE_DIR, 
        df_path=path.join(ELLIPSE_PERCENTAGE_DIR, 'data.csv'),
        transform=TRANSFORM,
        bs=BS
    ))
]

models_to_test = [
    *custom_models.get_models(input_size=(1, W, H)), 
    model_helpers.Param('PERCEPTRON', lambda: nn.Sequential(model_helpers.Flatten(), nn.Linear(W * H, 1)))
]

print(f"Device: {DEVICE}")

# +
OPTIMS = [
    model_helpers.Param('Adam', lambda: torch.optim.Adam),
    model_helpers.Param('SGD', lambda: partial(torch.optim.SGD, lr=0.00001))
]

LOSS_FNS = [
    model_helpers.Param('L1LOSS', lambda: model_helpers.squeeze_loss(nn.L1Loss())),
    model_helpers.Param('MSELOSS', lambda: model_helpers.squeeze_loss(nn.MSELoss()))
]


grid = model_helpers.new_grid_search(models_to_test, OPTIMS, LOSS_FNS)
grid = list(grid)

print(f"Will train {len(LOSS_FNS) * len(models_to_test) * len(dataset_generators)} models")
print(f"{len(models_to_test)} MODELS")
print(f"{len(dataset_generators)} DATASETS")
print(f"{len(LOSS_FNS)} LOSS_FNS")
print(f"{len(OPTIMS)} OPTIMS")


# +
def normalize_train_test_df(df, indexes):
    train_losses = (pd.DataFrame(df.train_losses.tolist(), index=df.set_index(indexes).index).stack()
        .reset_index(name='train_losses'))
    val_losses = (pd.DataFrame(df.val_losses.tolist(), index=df.set_index(indexes).index).stack()
        .reset_index(name='val_losses'))
    epoch = (pd.DataFrame(df.epoch.tolist(), index=df.set_index(indexes).index).stack()
        .reset_index(name='epoch'))
    val_losses["epoch"] = epoch.epoch
    val_losses["train_losses"] = train_losses.train_losses
    return val_losses.drop(columns="level_4").set_index(indexes)


indexes = [
        "model_name",
        "dataset",
        "optim",
        "loss_fn"
]

columns = [
        *indexes,
        "train_losses", 
        "val_losses"]

df = pd.DataFrame(
    columns=columns
)

for dt in dataset_generators:
    dl_train = dt.param.train()
    dl_test = dt.param.test()
    for row in grid:
        print(f"Pool: {row.model.name}\nOpt: {row.opt.name}\nLoss: {row.loss.name}\nDT: {dt.name}\n")
        metrics = model_helpers.train(
            dl_train, 
            dl_test, 
            row.opt.param(),
            row.loss.param(), 
            row.model.param(), 
            MAX_EPOCHS, 
            DEVICE)
        row = pd.Series({
            "model_name": row.model.name,
            "dataset": dt.name,
            "optim": row.opt.name,
            "loss_fn": row.loss.name
        }).append(pd.Series(metrics))
        df = df.append(row, ignore_index=True)
# -

df = normalize_train_test_df(df, indexes)
df.to_csv('FULL_RESULTS.csv')
