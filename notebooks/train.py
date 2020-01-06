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

import sys
import os
from os import path

import pandas as pd

from functools import partial, reduce
from tqdm import tqdm

import datetime

## Local Imports ##
if '../' not in sys.path:
    sys.path.insert(0, '../')
from models import helpers as model_helpers, model_definitions as custom_models
from datasets import helpers as dataset_helpers, datasets as custom_datasets

from train_single_script import create_arg_str

from VOC import DT_DEST_RGB_RANDOM, DT_DEST_RGB_SINGLE_CLASS 

# +
# Channel, Width, Height
C, W, H = (3, 128, 128)

TRAIN_SINGLE_PATH = './train_single_script.py'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HOME = path.expanduser('~')
DT_ROOT = path.abspath(
    path.join('..', 'data', f'{W}x{H}')
)

POLYGON_COUNT_DIR = path.join(DT_ROOT, 'polygon_data_counts')
POLYGON_PERCENTAGE_DIR = path.join(DT_ROOT, 'polygon_data_percentage')

POLYGON_RGB_COUNT_DIR = path.join(DT_ROOT, 'polygon_RGB_counts')
POLYGON_RGB_NOISED_COUNT_DIR = path.join(DT_ROOT, 'polygon_rgb_noised_counts')

ELLIPSE_COUNT_DIR = path.join(DT_ROOT, 'ellipse_data_counts')
ELLIPSE_PERCENTAGE_DIR = path.join(DT_ROOT, 'ellipse_data_percentage')

VOC_SEGS_COUNTS_DIR = path.join(HOME, 'datasets', 'VOC_FORMS')

# +
## Grid Search Params ##
RANDOM_SEARCH = False
SEARCH_LEN = 4

MODELS = custom_models.get_models((C, W, H))
MODELS = [
    "UNET",
    "UNET_HALF",
    "RESNET_18",
    "STRIDE_4",
    "STRIDE_8",
    "MAX_POOL_4",
    "MAX_POOL_8",
    "SUM_POOL_4",
    "SUM_POOL_8"
    
]

DATASETS = [
    #DT_DEST_RGB_RANDOM, 
    #DT_DEST_RGB_SINGLE_CLASS("AEROPLANE"),
    #VOC_SEGS_COUNTS_DIR,
    #POLYGON_COUNT_DIR
    POLYGON_RGB_NOISED_COUNT_DIR
]

OPTIMS = ["ADAM"]
LOSS_FNS = ["L1LOSS"]
LRS = [1e-2, 1e-3, 5e-4]

# +
grid = model_helpers.new_grid_search(MODELS, OPTIMS, LOSS_FNS, LRS)
grid = list(grid)

print(f"Will train {len(LOSS_FNS) * len(MODELS) * len(DATASETS) * len(OPTIMS) * len(LRS)} models")
print(f"{len(MODELS)} MODELS")
print(f"{len(DATASETS)} DATASETS")
print(f"{len(LOSS_FNS)} LOSS_FNS")
print(f"{len(OPTIMS)} OPTIMS")
print(f"Device: {DEVICE}")

models_str = '\t' + "\n\t".join(MODELS)
lrs_str = '\t' + "\n\t".join(map(str, LRS))

dts_str = '\t' + "\n\t".join([dt.split('/')[-1] for dt in DATASETS])
print(f"MODELS:")
print(models_str)
print(f"LRS:")
print(lrs_str)
print(f"DATASETS:")
print(dts_str)

# +
curr_time = datetime.datetime.now()
CURR_TIME_STR = (
    f"{curr_time.year}-{curr_time.month}-{curr_time.day}_"
    f"{curr_time.hour}-{curr_time.minute}-{curr_time.second}"
)
OUT_FILE = path.join("logs", f"out_{CURR_TIME_STR}.log")
MAX_EPOCHS = 25
BS = 32

BASE_ARGS = {
    "C": C,
    "H": H,
    "W": W,
    "bs": BS,
    "epochs": MAX_EPOCHS,
    "device": DEVICE,
    "id": CURR_TIME_STR,
    "epochs": MAX_EPOCHS
}

print(f"Epochs: {MAX_EPOCHS}")
print(f"BS: {BS}")
print(f"Timestamp: {CURR_TIME_STR}")


# -

def grid_search(dts, rows, sanity):
    if sanity: print("Performing sanity check...")
    else     : print("Training...")
    for dt in tqdm(dts):
        for row in tqdm(rows):
            command = (
                f'python3 {TRAIN_SINGLE_PATH} ' + 
                create_arg_str({
                    **BASE_ARGS,
                    "dataset": dt,
                    "model"  : row.model,
                    "optim"  : row.opt,
                    "loss_fn": row.loss,
                    "lr"     : row.lr,
                    "sanity" : sanity,
                }) + f' >> {OUT_FILE}')
            status = os.system(command)
            if status != 0: raise RuntimeError(f'FAILED: {command}')
    if sanity: print("Sanity Check: All Passed!")
    else     : print("Done Training!")


if __name__ == "__main__":
    grid_search(DATASETS, grid, sanity=True)
    grid_search(DATASETS, grid, sanity=False)


