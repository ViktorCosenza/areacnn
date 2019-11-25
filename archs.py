from torch import nn

def base_model(pool_layer, pool_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1),
        pool_layer(pool_size),
        nn.ReLU()
        nn.Conv2d(3, 6, 3, 1),
        pool_layer(pool_size),
        nn.ReLU(),
        nn.Conv2d(6, 4, 2, 1),
        pool_layer(pool_size),
        nn.ReLU()
    )

def cnn_max_pool(pool_size=2):
    return base_model(nn.MaxPool2d, pool_size)

def cnn_sum_pool(pool_size=2):
    return base_model(nn.SumPool2d, pool_size)