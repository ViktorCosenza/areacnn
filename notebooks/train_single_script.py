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

if '../' not in sys.path: sys.path.insert(0, '../')
    
from models.helpers import Param


# +
def get_params(args):
    print(Param('a', None))
    
def main():
    p = ArgumentParser(description='Train on a dataset with a CNN')
    p.add_argument('-model', type=str, help='The model name')
    p.add_argument('-optim', type=str, help='The optim to use')
    p.add_argument('-loss_fn', type=str, help='The loss function')    
    p.add_argument('-dataset', type=str, help='The dataset')    
    args = p.parse_args()
    get_params(args)
    
    
if __name__ == '__main__': main()
# -


