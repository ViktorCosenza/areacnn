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
import torch

from torch import nn
from torch.utils import data
from torch.nn import functional as F

from torchvision import transforms
from random import randint

import os
from os import path

import pandas as pd

from functools import partial, reduce
from tqdm import tqdm

import datetime

## Local Imports ##
from models import helpers as model_helpers, models as custom_models
from datasets import helpers as dataset_helpers, datasets as custom_datasets

# +
W, H = (32, 32)
BS = 128
MAX_EPOCHS = 30

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
    model_helpers.Param('SGD', lambda: partial(torch.optim.SGD, lr=0.0005))
]

LOSS_FNS = [
    model_helpers.Param('L1LOSS', lambda: model_helpers.squeeze_loss(nn.L1Loss())),
    model_helpers.Param('MSELOSS', lambda: model_helpers.squeeze_loss(nn.MSELoss()))
]


grid = model_helpers.new_grid_search(models_to_test, OPTIMS, LOSS_FNS)
grid = list(grid)

print(f"Will train {len(LOSS_FNS) * len(models_to_test) * len(dataset_generators) * len(OPTIMS)} models")
print(f"{len(models_to_test)} MODELS")
print(f"{len(dataset_generators)} DATASETS")
print(f"{len(LOSS_FNS)} LOSS_FNS")
print(f"{len(OPTIMS)} OPTIMS")

# +
columns = [
    "model_name",
    "dataset",
    "optim",
    "loss_fn",
    "train_loss",
    "val_loss"
]

df = pd.DataFrame(columns=columns)

for dt in tqdm(dataset_generators):
    dl_train = dt.param.train()
    dl_test = dt.param.test()
    for row in tqdm(grid):
        #print(f"Pool: {row.model.name}\nOpt: {row.opt.name}\nLoss: {row.loss.name}\nDT: {dt.name}\n")
        metrics = model_helpers.train(
            dl_train, 
            dl_test, 
            row.opt.param(),
            row.loss.param(), 
            row.model.param(), 
            MAX_EPOCHS, 
            DEVICE)
        rows = list(map(lambda r: {
            "model_name": row.model.name,
            "optim": row.opt.name,
            "loss_fn": row.loss.name,
            "dataset": dt.name,
            "epoch": r["epoch"],
            "train_loss": r["train_loss"],
            "val_loss": r["val_loss"],
            "train_loss_avg": r["train_loss_avg"],
            "val_loss_avg": r["val_loss_avg"]}, metrics)) 
        df = df.append(rows, sort=True , ignore_index=True)
# -

df.to_csv(f'{str(datetime.datetime.now())}_FULL_RESULTS.csv')




