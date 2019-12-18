from torch import nn
from .helpers import SumPool2d, DummyLayer, Flatten, Param
from .models import create_model
from functools import reduce


STRIDE_2 = lambda: nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=2),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
    nn.ReLU(),
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

STRIDE_8 = lambda: nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=4),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=4),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
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
    nn.ReLU(),
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
    nn.ReLU(),
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
    nn.ReLU(),
)

BASE_MODEL_8 = lambda Pooling: nn.Sequential(
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
    nn.ReLU(),
    Pooling(2),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
    Pooling(2),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
)


def get_models(input_size):
    return {
        ## Depth = 2 ##
        "SUM_POOL_2": Param(
            "SUM_POOL_2",
            lambda: create_model(lambda: BASE_MODEL_2(SumPool2d), input_size, nn.ReLU),
        ),
        "MAX_POOL_2": Param(
            "MAX_POOL_2",
            lambda: create_model(
                lambda: BASE_MODEL_2(nn.MaxPool2d), input_size, nn.ReLU
            ),
        ),
        ## Depth = 4 ##
        "SUM_POOL_4": Param(
            "SUM_POOL_4",
            lambda: create_model(lambda: BASE_MODEL_4(SumPool2d), input_size, nn.ReLU),
        ),
        "MAX_POOL_4": Param(
            "MAX_POOL_4",
            lambda: create_model(
                lambda: BASE_MODEL_4(nn.MaxPool2d), input_size, nn.ReLU
            ),
        ),
        ## Depth = 6 ##
        "SUM_POOL_6": Param(
            "SUM_POOL_6",
            lambda: create_model(lambda: BASE_MODEL_6(SumPool2d), input_size, nn.ReLU),
        ),
        "MAX_POOL_6": Param(
            "MAX_POOL_6",
            lambda: create_model(
                lambda: BASE_MODEL_6(nn.MaxPool2d), input_size, nn.ReLU
            ),
        ),
        ## Depth = 8 ##
        "SUM_POOL_8": Param(
            "SUM_POOL_8",
            lambda: create_model(lambda: BASE_MODEL_8(SumPool2d), input_size, nn.ReLU),
        ),
        "MAX_POOL_8": Param(
            "MAX_POOL_8",
            lambda: create_model(
                lambda: BASE_MODEL_8(nn.MaxPool2d), input_size, nn.ReLU
            ),
        ),
        ## Stride ##
        "STRIDE_2": Param(
            "STRIDE_2", lambda: create_model(lambda: STRIDE_2(), input_size, nn.ReLU),
        ),
        "STRIDE_4": Param(
            "STRIDE_4", lambda: create_model(lambda: STRIDE_4(), input_size, nn.ReLU)
        ),
        "STRIDE_6": Param(
            "STRIDE_6", lambda: create_model(lambda: STRIDE_6(), input_size, nn.ReLU)
        ),
        "STRIDE_8": Param(
            "STRIDE_8", lambda: create_model(lambda: STRIDE_8(), input_size, nn.ReLU)
        ),
        ## MLP ##
        "MLP": Param(
            "MLP", lambda: create_model(lambda: DummyLayer(), input_size, nn.ReLU)
        ),
        
        ## Smaller MLP ##
        'SMALLER_MLP_2': Param(
            'SMALLER_MLP_2', 
            lambda: nn.Sequential(
                Flatten(), 
                nn.Linear(reduce(lambda prev, e: prev * e, input_size, 1), 2),
                nn.Linear(2, 1)
            ),
        ),

        'SMALLER_MLP_3': Param(
            'SMALLER_MLP_3', 
            lambda: nn.Sequential(
                Flatten(), 
                nn.Linear(reduce(lambda prev, e: prev * e, input_size, 1), 3),
                nn.Linear(3, 1)
            ),
        ),

        'SMALLER_MLP_3_3': Param(
            'SMALLER_MLP_3_3', 
            lambda: nn.Sequential(
                Flatten(), 
                nn.Linear(reduce(lambda prev, e: prev * e, input_size, 1), 3),
                nn.Linear(3, 3),
                nn.Linear(3, 1)
            ),
        ),

        
        ## Perceptron ##
        "PERCEPTRON": Param(
            "PERCEPTRON",
            lambda: nn.Sequential(
                Flatten(), nn.Linear(reduce(lambda prev, e: prev * e, input_size, 1), 1)
            ),
        ),
    }
