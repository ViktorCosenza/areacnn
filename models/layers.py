from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, x):
        return x


class SumPool2d(nn.Module):
    def __init__(self, pool_size=2):
        super(SumPool2d, self).__init__()
        self.pool_size = pool_size
        self.area = pool_size * pool_size

    def forward(self, x):
        return F.avg_pool2d(x, self.pool_size) * self.area
