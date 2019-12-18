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
from models.model_definitions import get_models

from functools import partial

from transform import TRANSFORM
from datasets import datasets as custom_datasets

# +
OPTIMS = {
    "ADAM": optim.Adam,
    "SGD" : partial(optim.SGD, lr=0.005)
}

LOSS_FNS = {
    "L1LOSS" : lambda: model_helpers.squeeze_loss(nn.L1Loss()),
    "MSELOSS": lambda: model_helpers.squeeze_loss(nn.MSELoss())
}

RESULTS_DIR = './results'


# +
def create_arg_str(args):
    return ' '.join([
        f'-model {args["model"]}',
        f'-optim {args["optim"]}',
        f'-loss_fn {args["loss_fn"]}',
        f'-dataset {args["dataset"]}',
        f'-bs {args["bs"]}',
        f'-epochs {args["epochs"]}',
        f'-device {args["device"]}',
        f'-W {args["W"]}',
        f'-H {args["H"]}',
        f'-id {args["id"]}',
        f'{"--sanity" if args["sanity"] else ""}'
    ])
    
def get_params(args):
    return {
        "model"   : get_models((1, args.W, args.H))[args.model], 
        "dataset" : 
            Param(
                args.dataset, 
                custom_datasets.get_dataset(
                    root_dir=args.dataset,
                    df_path=path.join(args.dataset, 'data.csv'),
                    transform=TRANSFORM,
                    bs=args.bs)),
        "optim"   : Param(args.optim, OPTIMS[args.optim]),
        "loss_fn" : Param(args.loss_fn, LOSS_FNS[args.loss_fn]),
        "epochs"  : args.epochs,
        "device"  : args.device,
    }


def train(model, dataset, optim, loss_fn, epochs, device):
    metrics = model_helpers.train(
        dl_train     =dataset.param.train(),
        dl_val       =dataset.param.test(),
        model        =model.param(),
        opt_func     =optim.param,
        loss_fn      =loss_fn.param(),
        epochs       =epochs,
        device       =device,
        show_progress=False
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

def sanity_check(model, dl):
    for e, l in dl.param.test():
        try: 
            model.param()(e)
            print(f'Sanity check on {model.name} {dl.name} OK!')
            
        except Exception as e: 
            print(f'Model: {model.name}\n'
                  f'Dataset: {dl.name}\n'
                  f'Exception: {e}\n')
            print(f'Sanity check on {model.name} {dl.name} FAILED!')
            raise e
        break

def main():
    p = ArgumentParser(description='Train on a dataset with a CNN')
    p.add_argument('-model'    , type=str, required=True , help='The model name')
    p.add_argument('-optim'    , type=str, required=True , help='The optim to use')
    p.add_argument('-loss_fn'  , type=str, required=True , help='The loss function')    
    p.add_argument('-dataset'  , type=str, required=True , help='The dataset path')
    p.add_argument('-bs'       , type=int, required=True , help='Batch size')
    p.add_argument('-epochs'   , type=int, required=True , help="Epochs")
    p.add_argument('-device'   , type=str, required=False, help='Torch device', default='cuda')
    p.add_argument('-W'        , type=int, required=True , help='Input width')
    p.add_argument('-H'        , type=int, required=True , help='Input height')
    p.add_argument('-id'       , type=str, required=True , help='Unique id for current execution')
    p.add_argument('--sanity'  , action='store_true'     , help='Run single image to check')
    args = p.parse_args()
    params = get_params(args)
    
    if not path.exists(RESULTS_DIR): 
        print(f'Creating results directory at: f{RESULTS_DIR}')
        os.makedirs(RESULTS_DIR)
    if args.sanity: 
        sanity_check(model=params["model"], dl=params["dataset"])
    else: 
        df = train(**params)
        csv_dest = path.join(RESULTS_DIR, f'{args.id}.csv')
        if path.exists(csv_dest): 
            print('Appending to existing file')
            old_df = pd.read_csv(csv_dest, index_col=0)
            df = pd.concat([old_df, df], sort=False)
            df.to_csv(csv_dest)
        else: 
            print(f'Saving to {csv_dest}')
            df.to_csv(csv_dest)
        print(f'Done with {args.model}, {args.dataset}, {args.loss_fn}, {args.optim}')
    
    
if __name__ == '__main__': main()
# -


