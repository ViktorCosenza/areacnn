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
C, W, H = (3, 224, 224)

TRAIN_SINGLE_PATH = './train_single_script.py'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DT_ROOT = 'data'
POLYGON_COUNT_DIR = path.join(DT_ROOT, 'polygon_data_counts')
POLYGON_PERCENTAGE_DIR = path.join(DT_ROOT, 'polygon_data_percentage')

ELLIPSE_COUNT_DIR = path.join(DT_ROOT, 'ellipse_data_counts')
ELLIPSE_PERCENTAGE_DIR = path.join(DT_ROOT, 'ellipse_data_percentage')

VOC_SEGS_COUNTS_DIR = path.join('/home', 'victor', 'datasets', 'VOC_FORMS')

# +
## Grid Search Params ##
MODELS = custom_models.get_models(input_size=(C, W, H))
MODELS = [
    MODELS['MLP'], 
    MODELS['SMALLER_MLP_2'], 
    MODELS['SMALLER_MLP_3'], 
    MODELS['SMALLER_MLP_3_3'], 
    MODELS['PERCEPTRON'], 
    MODELS['RESNET_34']
]
MODELS = map(lambda p: p.name, MODELS)
MODELS = list(MODELS)

DATASETS = [DT_DEST_RGB_RANDOM, DT_DEST_RGB_SINGLE_CLASS('AEROPLANE')]
OPTIMS = ["ADAM", "SGD"]
LOSS_FNS = ["L1LOSS"]

# +
grid = model_helpers.new_grid_search(MODELS, OPTIMS, LOSS_FNS)
grid = list(grid)

print(f"Will train {len(LOSS_FNS) * len(MODELS) * len(DATASETS) * len(OPTIMS)} models")
print(f"{len(MODELS)} MODELS")
print(f"{len(DATASETS)} DATASETS")
print(f"{len(LOSS_FNS)} LOSS_FNS")
print(f"{len(OPTIMS)} OPTIMS")
print(f"Device: {DEVICE}")

# +
curr_time = datetime.datetime.now()
CURR_TIME_STR = (f"{curr_time.year}-{curr_time.month}-{curr_time.day}_"
                 f"{curr_time.hour}-{curr_time.minute}-{curr_time.second}")
MAX_EPOCHS = 50
BS = 32

BASE_ARGS = {
    "C"        : C,
    "H"        : H,
    "W"        : W,
    "bs"       : BS,
    "epochs"   : MAX_EPOCHS,
    "device"   : DEVICE,
    "id"       : CURR_TIME_STR
}

print(f"Epochs: {MAX_EPOCHS}")
print(f"BS: {BS}")
print(f"Timestamp: {CURR_TIME_STR}")


# +
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
                    "epochs" : MAX_EPOCHS,
                    "sanity" : sanity,
                }) + f' >> logs/out_{CURR_TIME_STR}.log')
            print(command)
            status = os.system(command)
            if status != 0: raise AssertionError(f'FAILED: {command}')
    if sanity: print("Sanity Check: All Passed!")
    else     : print("Done Training!")

grid_search(DATASETS, grid, sanity=True)
grid_search(DATASETS, grid, sanity=False)
# -


