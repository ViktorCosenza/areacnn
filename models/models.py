import torch
from torch import nn
from .helpers import SumPool2d, Flatten, Param
from functools import reduce
import math

default_params = {
    "max_channels": 3,
    "activation_fn": nn.ReLU,
    "pool_size": 2,
    "conv_stride": 1
}

STRIDE_2 = lambda: nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=2),
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
                    nn.ReLU()
)

STRIDE_4 = lambda: nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=2),
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
                    nn.ReLU(),
)

STRIDE_6 = lambda: nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=4),
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
                    nn.ReLU(),
)

BASE_MODEL_2 = lambda Pooling: nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5),
    Pooling(2),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    Pooling(2),
    nn.ReLU()
)

BASE_MODEL_4 = lambda Pooling: nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5),
    Pooling(2),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    Pooling(2),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    Pooling(2),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    nn.ReLU()
)

BASE_MODEL_6 = lambda Pooling: nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5),
    Pooling(2),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    Pooling(2),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    Pooling(2),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    Pooling(2),
    nn.ReLU(), 
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    nn.ReLU()
)

MODELS_TO_TEST = {
        ## Depth = 2 ##
        'SUM_POOL_2_SQZ': Param('SUM_POOL_2_SQZ', 
              lambda: base_model(pool_layer=SumPool2d, num_layers=2, input_size=input_size, **params)),
        'MAX_POOL_2_SQZ': Param('MAX_POOL_2_SQZ', 
            lambda: base_model(pool_layer=nn.MaxPool2d, num_layers=2, input_size=input_size, **params)),
                    
        ## Depth = 3 ##
        'SUM_POOL_3_SQZ': Param('SUM_POOL_3_SQZ', 
            lambda: base_model(pool_layer=SumPool2d, num_layers=3, input_size=input_size, **params)),
        'MAX_POOL_3_SQZ': Param('MAX_POOL_3_SQZ', 
            lambda: base_model(pool_layer=nn.MaxPool2d, num_layers=3, input_size=input_size, **params)),
        
        ## Depth = 5 ##
        'SUM_POOL_5_SQZ': Param('SUM_POOL_5_SQZ', 
            lambda: base_model(pool_layer=SumPool2d, num_layers=5, input_size=input_size, **params)),
        'MAX_POOL_5_SQZ': Param('MAX_POOL_5_SQZ', 
            lambda: base_model(pool_layer=nn.MaxPool2d, num_layers=5, input_size=input_size, **params)),
        
        ## Depth = 2 ##
        'SUM_POOL_2': Param('SUM_POOL_2', 
            lambda: nn.Sequential(
                BASE_MODEL_2(SumPool2d), 
                create_head_from_cnn(BASE_MODEL_2(SumPool2d), input_size, nn.ReLU))),
        'MAX_POOL_2': Param('MAX_POOL_2', 
            lambda: nn.Sequential(
                BASE_MODEL_2(nn.MaxPool2d), 
                create_head_from_cnn(BASE_MODEL_2(nn.MaxPool2d), input_size, nn.ReLU))),
        
        ## Depth = 4 ##
        'SUM_POOL_4': Param('SUM_POOL_4', 
            lambda: nn.Sequential(
                BASE_MODEL_4(SumPool2d), 
                create_head_from_cnn(BASE_MODEL_4(SumPool2d), input_size, nn.ReLU))),
        'MAX_POOL_4': Param('MAX_POOL_4', 
            lambda: nn.Sequential(
                BASE_MODEL_4(nn.MaxPool2d), 
                create_head_from_cnn(BASE_MODEL_4(nn.MaxPool2d), input_size, nn.ReLU))),
        
        ## Depth = 6 ##
        'SUM_POOL_6': Param('SUM_POOL_6', 
            lambda: nn.Sequential(
                BASE_MODEL_6(SumPool2d), 
                create_head_from_cnn(BASE_MODEL_6(SumPool2d), input_size, nn.ReLU))),
        'MAX_POOL_6': Param('MAX_POOL_6', 
            lambda: nn.Sequential(
                BASE_MODEL_6(nn.MaxPool2d), 
                create_head_from_cnn(BASE_MODEL_6(nn.MaxPool2d), input_size, nn.ReLU))),
        
        ## Stride ##
        'STRIDE_2': Param('STRIDE_2', 
              lambda: nn.Sequential(
                  STRIDE_2(),
                  create_head_from_cnn(STRIDE_2(), input_size, nn.ReLU))),
        'STRIDE_4': Param('STRIDE_4',
             lambda: nn.Sequential(
                 STRIDE_4(),
                 create_head_from_cnn(STRIDE_4(), input_size, nn.ReLU))),
        'STRIDE_6': Param('STRIDE_6',
            lambda: nn.Sequential(
                STRIDE_4(),
                create_head_from_cnn(STRIDE_4(), input_size, nn.ReLU)))
}

def get_models(input_size, params=default_params):
    return list(MODELS_TO_TEST.values())

def create_head(out_dims, activation_fn):
    out_size = reduce(lambda prev, e: prev * e, out_dims[-3:], 1)
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
