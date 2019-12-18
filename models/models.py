import torch
from torch import nn
from .helpers import SumPool2d, Flatten, Param
from functools import reduce
import math

def create_model(cnn_func, input_size, activation_fn=nn.ReLU):
    cnn = cnn_func()
    model = nn.Sequential(cnn, create_head_from_cnn(cnn, input_size, activation_fn))
    return model

def create_head(out_dims, activation_fn):
    out_size = reduce(lambda prev, e: prev * e, out_dims[-3:], 1)
    print(f"Min size: {out_size}")
    return nn.Sequential(
        Flatten(),
        nn.Linear(out_size, 1024),
        activation_fn(),
        nn.Linear(1024, 256),
        activation_fn(),
        nn.Linear(256, 1)
    )    
    
def create_head_from_cnn(cnn, input_size, activation_fn):
    return create_head(cnn(torch.zeros(1, *input_size)).shape, activation_fn)