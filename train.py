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
import itertools
from collections import namedtuple

## Local Imports ##
from models import helpers as model_helpers, models as custom_models
from datasets import helpers as dataset_helpers, datasets as custom_datasets

# +
W, H = (512, 512)
BS = 256
DEVICE = 'cuda'
DT_ROOT = 'data'
POLYGON_COUNT_DIR = 'polygon_data_counts'
POLYGON_PERCENTAGE_DIR = 'polygon_data_percentage'
ELLIPSE_COUNT_DIR = 'ellipse_data_counts'
ELLIPSE_PERCENTAGE_DIR = 'ellipse_data_percentage'

options = {
    "bs": BS,
    "device": DEVICE,
    "transform_args": {"size": (W, H)}
}

rect_counts_options = custom_datasets.gen_dataset_options(
    root_dir=path.join(DT_ROOT, POLYGON_COUNT_DIR),
    **options
)

rect_percentage_options = custom_datasets.gen_dataset_options(
    root_dir=path.join(DT_ROOT, POLYGON_PERCENTAGE_DIR),
    **options
)

ellipse_counts_options = custom_datasets.gen_dataset_options(
    root_dir=path.join(DT_ROOT, ELLIPSE_COUNT_DIR),
    **options
)

ellipse_percentage_options = custom_datasets.gen_dataset_options(
    root_dir=path.join(DT_ROOT, ELLIPSE_PERCENTAGE_DIR),
    **options
)

dataset_generators = [
    model_helpers.Param('RECT_COUNT', lambda: custom_datasets.get_dataset(**rect_counts_options)),
    model_helpers.Param('RECT_PCT', lambda: custom_datasets.get_dataset(**rect_percentage_options)),
    model_helpers.Param('ELLIPSE_COUNT', lambda: custom_datasets.get_dataset(**ellipse_counts_options)),
    model_helpers.Param('ELLIPSE_PCT', lambda: custom_datasets.get_dataset(**ellipse_percentage_options))
]

# +
EARLY_STOP_PATIENCE = 100
REDUCE_ON_PLATEAU_PATIENCE = 30 
MAX_EPOCHS = 100

learner_args = {
    "metrics": [
        mean_squared_error,
        mean_absolute_error
    ],
    "callback_fns": [
            #partial(EarlyStoppingCallback, patience=EARLY_STOP_PATIENCE), 
            partial(ReduceLROnPlateauCallback, patience=REDUCE_ON_PLATEAU_PATIENCE),
            partial(CSVLogger, filename=f"history_mlp")
        ],
    "silent": True
}

OPTIMS = [
    #model_helpers.Param('Adam', torch.optim.Adam),
    model_helpers.Param('SGD', torch.optim.SGD)
]

LOSS_FNS = [
    model_helpers.Param('L1LOSS', nn.L1Loss),
    #model_helpers.Param('MSELOSS', nn.MSELoss)
]

models_to_test = custom_models.get_models()

cnn_grid = model_helpers.new_grid_search(models_to_test, OPTIMS, LOSS_FNS)

mlp_grid = model_helpers.new_grid_search([None], OPTIMS, LOSS_FNS)
cnn_grid, mlp_grid = list(cnn_grid), list(mlp_grid)
print(f"Will train {(len(cnn_grid) + len(mlp_grid)) * len(dataset_generators)} models")
## WHAT I DO WHITH DISSSS????
squeeze_loss = lambda loss_fn: lambda x,y: loss_fn(x.view(-1), y)
# -

df_results = pd.DataFrame(columns=["pth", "dt", "loss", "optim", "val_loss"])


# +
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

for dataset in dataset_generators:
    bunch = dataset.param()
    for row in mlp_grid:
        print(f"Training {dataset.name}, {row.opt.name}, {row.loss.name}")
        model_path = path.join(bunch.path,'..', '..', 'results', 'mlp', row.opt.name, row.loss.name)
        model_path = path.abspath(model_path)
        learn = Learner(
            model=nn.Sequential(Flatten(), nn.Linear(W * H, 1)),
            opt_func=row.opt.param,
            data=bunch,
            loss_func=squeeze_loss(row.loss.param()),
            path=model_path,
            **learner_args
        )
        learn.fit(MAX_EPOCHS)
        model_helpers.save_stats(learn)
        learn.save('model')
        df_results = df_results.append({
                "pth": learn.path, 
                "dt": dataset.name,
                "loss": row.loss.name, 
                "optim": row.opt.name, 
                "val_loss": learn.recorder.losses[-1].item()
            }, ignore_index=True)
# -

for dataset in dataset_generators:
    bunch = dataset.param()
    for row in cnn_grid:
        print(f"Training {dataset.name}, {row.model.name}, {row.opt.name}, {row.loss.name}")
        model_path = path.join(bunch.path, '..', '..', 'results', row.model.name, row.opt.name, row.loss.name)
        model_path = path.abspath(model_path)
        learn = cnn_learner(
            data=bunch,
            path=model_path,
            base_arch=lambda t: row.model.param,
            cut=lambda x: x,
            loss_func=squeeze_loss(row.loss.param()),
            opt_func=row.opt.param,
            **learner_args
        )
        learn.fit(MAX_EPOCHS)
        model_helpers.save_stats(learn)
        learn.save('model')
        df_results = df_results.append({
                "pth": learn.path, 
                "dt": dataset.name,
                "loss": row.loss.name, 
                "optim": row.opt.name, 
                "val_loss": learn.recorder.losses[-1].item()
            }, ignore_index=True)

df_results.to_csv('FULL_RESULTS.CSV')

# +
## Broken... ## 
#DT_ROOT_DIRS = map(lambda p: path.join(DT_ROOT, p, 'results'),[POLYGON_COUNT_DIR, POLYGON_PERCENTAGE_DIR])
#for root_dir in DT_ROOT_DIRS:
#    model_helpers.summarize_results([*models_to_test, {"name":"mlp"}], root_dir)
