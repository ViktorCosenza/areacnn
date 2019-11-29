import torch
from torch import nn
from .helpers import SumPool2d, Flatten, Param
from functools import reduce
import math

def create_head(out_dims, activation_fn):
    out_size = reduce(lambda prev, e: prev * e, out_dims[-2:], 1)
    nearest_pof2 = 2 ** int(math.log2(out_size))
    return nn.Sequential(
        Flatten(),
        nn.Linear(out_size, nearest_pof2),
        activation_fn(),
        nn.Linear(nearest_pof2, nearest_pof2//2),
        activation_fn(),
        nn.Linear(nearest_pof2//2, 1)
    )    
    
def create_head_from_cnn(cnn, input_size, activation_fn):
    return create_head(cnn(torch.zeros(1, *input_size)).shape, activation_fn)


def base_layer(conv_args, pool_layer, pool_size, activation_fn):
    return nn.Sequential(nn.Conv2d(**conv_args), pool_layer(pool_size), activation_fn())


def base_model(max_channels, activation_fn, pool_layer, pool_size, num_layers, input_size, conv_stride=1):
    cnn = nn.Sequential(
        *[base_layer({
            "in_channels": i if i < max_channels else max_channels, 
            "out_channels": abs(i+1) if i < max_channels else max_channels,
            "kernel_size": 3,
            "stride": conv_stride
            }, pool_layer, pool_size, activation_fn) for i in range(1, num_layers//2 + 1)],
        *[base_layer({
            "in_channels": i if i < max_channels else max_channels,
            "out_channels": i - 1 if i - 1 < max_channels else max_channels,
            "kernel_size": 3,
            "stride": conv_stride 
        }, pool_layer, pool_size, activation_fn) for i in range(num_layers//2 + 1, 1, -1)]
    )   
    return nn.Sequential(
        cnn,
        create_head_from_cnn(cnn, input_size, activation_fn)
    )     
        
default_params = {
    "max_channels": 3,
    "activation_fn": nn.ReLU,
    "pool_size": 2,
    "conv_stride": 1
}

def get_models(input_size, params=default_params):
    return [
        ## Depth = 3 ##
        Param('SUM_POOL_4', lambda: base_model(pool_layer=SumPool2d, num_layers=3, input_size=input_size,                       **params)),
        Param('MAX_POOL_4', lambda: base_model(pool_layer=nn.MaxPool2d, num_layers=3, input_size=input_size,                   **params)),
                    
        ## Depth = 4 ##
        Param('SUM_POOL_8', lambda: base_model(pool_layer=SumPool2d, num_layers=4, input_size=input_size,                       **params)),
        Param('SUM_POOL_8', lambda: base_model(pool_layer=SumPool2d, num_layers=4, input_size=input_size,                       **params)),
        
        ## Depth = 16 ##
        #Param('SUM_POOL_16', base_model(pool_layer=SumPool2d, num_layers=8, **params)),
        #Param('SUM_POOL_16', base_model(pool_layer=SumPool2d, num_layers=8, **params)),
    ]

