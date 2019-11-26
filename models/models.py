from torch import nn
from .helpers import SumPool2d, Param

def base_layer(conv_args, pool_layer, pool_size, activation_fn):
    return nn.Sequential(nn.Conv2d(**conv_args), pool_layer(pool_size), activation_fn())

def base_model(max_channels, activation_fn, pool_layer, pool_size, num_layers, conv_stride=1):
    return nn.Sequential(
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

default_params = {
    "max_channels": 3,
    "activation_fn": nn.ReLU,
    "pool_size": 2,
    "num_layers": 4,
    "conv_stride": 1
}

def get_models(params=default_params):
    return [
        Param('SUM_POOL', base_model(pool_layer=SumPool2d, **params)),
        Param('MAX_POOL', base_model(pool_layer=nn.MaxPool2d, **params)),
        Param('AVG_POOL', base_model(pool_layer=nn.AvgPool2d, **params))
    ]

