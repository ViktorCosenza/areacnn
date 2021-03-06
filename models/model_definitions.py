from torch import nn
from torchvision import models
from .helpers import create_model, create_resnet, create_squeeze, Param
from .layers import SumPool2d, DummyLayer, Flatten
from functools import reduce, partial


def conv_block(in_ch, out_ch, ks, pooling, stride, regularizer, pool_size=2):
    return nn.Sequential(
        nn.Conv2d(kernel_size=ks, in_channels=in_ch, out_channels=out_ch),
        nn.ReLU(),
        regularizer(),
        nn.Conv2d(kernel_size=ks, in_channels=out_ch, out_channels=out_ch),
        nn.ReLU(),
        pooling(pool_size) if pool_size else pooling(),
        regularizer(),
    )


def unet_block(in_ch, out_ch, ks):
    return conv_block(
        in_ch,
        out_ch,
        ks,
        pooling=nn.MaxPool2d,
        stride=1,
        regularizer= lambda: nn.BatchNorm2d(out_ch),
    )


def conv_stride_block(in_ch, out_ch, ks, regularizer, stride=2):
    return conv_block(
        in_ch, out_ch, ks, pooling=DummyLayer, stride=stride, regularizer=regularizer, pool_size=False
    )


def conv_pool_block(in_ch, out_ch, ks, pooling, regularizer):
    return conv_block(
        in_ch, out_ch, ks, pooling=pooling, stride=1, regularizer=regularizer
    )


UNET = lambda in_ch: nn.Sequential(
    unet_block(in_ch, 32, 5), 
    unet_block(32, 64, 3), 
    unet_block(64, 128, 3),
    unet_block(128, 256, 3)
)

UNET_HALF = lambda in_ch: nn.Sequential(
    unet_block(in_ch, 32, 5),
    unet_block(32, 64, 3)
)

STRIDE_2 = lambda in_ch: conv_stride_block(
    in_ch, out_ch=3, ks=5, regularizer=DummyLayer, stride=2
)

STRIDE_4 = lambda in_ch: nn.Sequential(
    conv_stride_block(in_ch, out_ch=3, ks=5, regularizer=DummyLayer, stride=2),
    conv_stride_block(3, out_ch=3, ks=3, regularizer=DummyLayer, stride=2),
)

STRIDE_6 = lambda in_ch: nn.Sequential(
    conv_stride_block(in_ch, out_ch=3, ks=5, regularizer=DummyLayer, stride=2),
    conv_stride_block(3, out_ch=3, ks=3, regularizer=DummyLayer, stride=2),
    conv_stride_block(3, out_ch=3, ks=3, regularizer=DummyLayer, stride=2),
)

STRIDE_8 = lambda in_ch: nn.Sequential(
    conv_stride_block(in_ch, out_ch=3, ks=5, regularizer=DummyLayer, stride=2),
    conv_stride_block(3, out_ch=3, ks=3, regularizer=DummyLayer, stride=2),
    conv_stride_block(3, out_ch=3, ks=3, regularizer=DummyLayer, stride=2),
    conv_stride_block(3, out_ch=3, ks=3, regularizer=DummyLayer, stride=2),
)

BASE_MODEL_2 = lambda in_ch, pooling: nn.Sequential(
    conv_pool_block(in_ch, out_ch=3, ks=5, pooling=pooling, regularizer=DummyLayer)
)

BASE_MODEL_4 = lambda in_ch, pooling: nn.Sequential(
    conv_pool_block(in_ch=in_ch, out_ch=3, ks=5, pooling=pooling, regularizer=DummyLayer),
    conv_pool_block(in_ch=3, out_ch=3, ks=3, pooling=pooling, regularizer=DummyLayer),
)

BASE_MODEL_6 = lambda in_ch, pooling: nn.Sequential(
    conv_pool_block(in_ch=in_ch, out_ch=3, ks=5, pooling=pooling, regularizer=DummyLayer),
    conv_pool_block(in_ch=3, out_ch=3, ks=3, pooling=pooling, regularizer=DummyLayer),
    conv_pool_block(in_ch=3, out_ch=3, ks=3, pooling=pooling, regularizer=DummyLayer),
)

BASE_MODEL_8 = lambda in_ch, pooling: nn.Sequential(
    conv_pool_block(in_ch, out_ch=3, ks=5, pooling=pooling, regularizer=DummyLayer),
    conv_pool_block(in_ch=3, out_ch=3, ks=3, pooling=pooling, regularizer=DummyLayer),
    conv_pool_block(in_ch=3, out_ch=3, ks=3, pooling=pooling, regularizer=DummyLayer),
    conv_pool_block(in_ch=3, out_ch=3, ks=3, pooling=pooling, regularizer=DummyLayer),
)


def get_models(in_shape):
    in_ch = in_shape[0]
    return {
        ## Depth = 2 ##
        "SUM_POOL_2": lambda: create_model(
            lambda: BASE_MODEL_2(in_ch, SumPool2d), in_shape, nn.ReLU
        ),
        "MAX_POOL_2": lambda: create_model(
            lambda: BASE_MODEL_2(in_ch, nn.MaxPool2d), in_shape, nn.ReLU
        ),
        ## Depth = 4 ##
        "SUM_POOL_4": lambda: create_model(
            lambda: BASE_MODEL_4(in_ch, SumPool2d), in_shape, nn.ReLU
        ),
        "MAX_POOL_4": lambda: create_model(
            lambda: BASE_MODEL_4(in_ch, nn.MaxPool2d), in_shape, nn.ReLU
        ),
        ## Depth = 6 ##
        "SUM_POOL_6": lambda: create_model(
            lambda: BASE_MODEL_6(in_ch, SumPool2d), in_shape, nn.ReLU
        ),
        "MAX_POOL_6": lambda: create_model(
            lambda: BASE_MODEL_6(in_ch, nn.MaxPool2d), in_shape, nn.ReLU
        ),
        ## Depth = 8 ##
        "SUM_POOL_8": lambda: create_model(
            lambda: BASE_MODEL_8(in_ch, SumPool2d), in_shape, nn.ReLU
        ),
        "MAX_POOL_8": lambda: create_model(
            lambda: BASE_MODEL_8(in_ch, nn.MaxPool2d), in_shape, nn.ReLU
        ),
        ## Stride ##
        "STRIDE_2": lambda: create_model(lambda: STRIDE_2(in_ch), in_shape, nn.ReLU),
        "STRIDE_4": lambda: create_model(lambda: STRIDE_4(in_ch), in_shape, nn.ReLU),
        "STRIDE_6": lambda: create_model(lambda: STRIDE_6(in_ch), in_shape, nn.ReLU),
        "STRIDE_8": lambda: create_model(lambda: STRIDE_8(in_ch), in_shape, nn.ReLU),
        ## MLP ##
        "MLP": lambda: create_model(DummyLayer, in_shape, nn.ReLU),
        ## Smaller MLP ##
        "SMALLER_MLP_2": lambda: nn.Sequential(
            Flatten(),
            nn.Linear(reduce(lambda prev, e: prev * e, in_shape, 1), 2),
            nn.Linear(2, 1),
        ),
        "SMALLER_MLP_3": lambda: nn.Sequential(
            Flatten(),
            nn.Linear(reduce(lambda prev, e: prev * e, in_shape, 1), 3),
            nn.Linear(3, 1),
        ),
        "SMALLER_MLP_3_3": lambda: nn.Sequential(
            Flatten(),
            nn.Linear(reduce(lambda prev, e: prev * e, in_shape, 1), 3),
            nn.Linear(3, 3),
            nn.Linear(3, 1),
        ),
        ## Perceptron ##
        "PERCEPTRON": lambda: nn.Sequential(
            Flatten(), nn.Linear(reduce(lambda prev, e: prev * e, in_shape, 1), 1),
        ),

        #### Other Archs ####
        
        ## Resnet 34 ##
        "RESNET_34": lambda: create_resnet(models.resnet34, in_shape),
        ## Resnet ##
        "RESNET_18": lambda: create_resnet(models.resnet18, in_shape),
        
        
        ## Unet ##
        "UNET": lambda: create_model(lambda: UNET(in_ch), in_shape, nn.ReLU),
        "UNET_HALF": lambda: create_model(lambda: UNET_HALF(in_ch), in_shape, nn.ReLU),
        
        ## SqueezeNet ##
        "SQUEEZE": lambda: create_squeeze(models.squeezenet1_1, in_shape),
        
        ## MobileNet ##
        "MOBILENET": lambda: create_squeeze(models.mobilenet_v2, in_shape)
    }
