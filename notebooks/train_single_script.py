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
from argparse import ArgumentParser
from os import path
import sys

import torch
from torch import optim
from torch import nn
from torchvision import transforms

import pandas as pd

if '../' not in sys.path: sys.path.insert(0, '../')
    

from models import helpers as model_helpers 
from models.helpers import Param
from models.models import models_to_test

from datasets import datasets as custom_datasets

# +
OPTIMS = {
    "Adam": optim.Adam,
    "SGD" : optim.SGD
}

LOSS_FNS = {
    "L1LOSS" : lambda: model_helpers.squeeze_loss(nn.L1Loss()),
    "MSELOSS": lambda: model_helpers.squeeze_loss(nn.MSELoss())
}


# +
def get_params(args):
    print(args.transform or transforms.ToTensor())
    return {
        "model"   : models_to_test((1, args.W, args.H))[args.model], 
        "dataset" : 
            Param(
                args.dataset, 
                custom_datasets.get_dataset(
                    root_dir=args.dataset,
                    df_path=path.join(args.dataset, 'data.csv'),
                    transform=args.transform or transforms.Compose([transforms.ToTensor()]),
                    bs=args.bs)),
        "optim"   : Param(args.optim, OPTIMS[args.optim]),
        "loss_fn" : Param(args.loss_fn, LOSS_FNS[args.loss_fn]),
        "epochs"  : args.epochs,
        "device"  : args.device,
    }


def train(model, dataset, optim, loss_fn, epochs, device):
    metrics = model_helpers.train(
        dl_train=dataset.param.train(),
        dl_val  =dataset.param.test(),
        model   =model.param(),
        opt_func=optim.param,
        loss_fn =loss_fn.param(),
        epochs  =epochs,
        device  =device
    )
    
    rows = list(map(lambda r: {
        "model_name"     : model.name,
        "optim"          : optim.name,
        "loss_fn"        : loss_fn.name,
        "dataset"        : dataset.name,
        "epoch"          : r["epoch"],
        "train_loss"     : r["train_loss"],
        "val_loss"       : r["val_loss"],
        "train_loss_avg" : r["train_loss_avg"],
        "val_loss_avg"   : r["val_loss_avg"]}, metrics))
    return pd.DataFrame(rows)
    
def main():
    p = ArgumentParser(description='Train on a dataset with a CNN')
    p.add_argument('-model'    , type=str, required=True , help='The model name')
    p.add_argument('-optim'    , type=str, required=True , help='The optim to use')
    p.add_argument('-loss_fn'  , type=str, required=True , help='The loss function')    
    p.add_argument('-dataset'  , type=str, required=True , help='The dataset path')
    p.add_argument('-bs'       , type=int, required=True , help='Batch size')
    p.add_argument('-epochs'   , type=int, required=True , help="Epochs")
    p.add_argument('-device'   , type=str, required=False, help='Torch device', default='cuda')
    p.add_argument('-transform', type=str, required=False, help='Transforms'  , default='')
    p.add_argument('-W'        , type=int, required=True , help='Input width')
    p.add_argument('-H'        , type=int, required=True , help='Input height')
    p.add_argument('--sanity'  , action='store_true'     , help='Run single image to check')
    args = p.parse_args()
    params = get_params(args)
    if args.sanity: model_helpers.sanity_check(params)
    else: 
        df = train(**params)
        csv_dest = (
            f'{params["model"].name}-'
            f'{params["dataset"].name}-'
            f'{params["loss_fn"].name}-'
            f'{params["optim"].name}.csv')
        print(csv_dest)
        print(df.head())
        #df.to_csv(csv_dest)
    
    
if __name__ == '__main__': main()
# -

f'\
abc \
def \
'


