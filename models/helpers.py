from torch import nn

class SumPool2d(nn.Module):
    def __init__(self, pool_size):
        super(SumPool2d, self).__init__()
        self.pool_size = pool_size
        self.area = pool_size * pool_size
        
    def forward(self, x):
        return F.avg_pool2d(x, self.pool_size) * self.area

def base_layer(conv_args, pool_layer, pool_size, activation_fn):
    return nn.Sequential(nn.Conv2d(**conv_args), pool_layer(pool_size), activation_fn())
