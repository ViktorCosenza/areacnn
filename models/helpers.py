from torch import nn
from torch.nn import functional as F
from os import path
import pandas as pd
from tqdm import tqdm
from itertools import product as cartesian_product
from collections import namedtuple
import torch
from torch import nn
from functools import reduce

from .layers import Flatten

Param = namedtuple("Param", "name param")
Grid = namedtuple("Grid", "model opt loss")


def create_model(cnn_func, in_shape, activation_fn=nn.ReLU):
    cnn = cnn_func()
    model = nn.Sequential(cnn, create_head_from_cnn(cnn, in_shape, activation_fn))
    print(in_shape)
    return model


def create_resnet(resnet_fn, in_shape, pretrained=False):
    m = resnet_fn(pretrained)
    m.fc = Flatten()
    out_shape = m(torch.zeros((1, *in_shape))).shape
    m.fc = nn.Linear(reduce(lambda prev, e: prev * e, out_shape, 1), 1)
    return m


def create_head(out_shape, activation_fn=nn.ReLU):
    out_size = reduce(lambda prev, e: prev * e, out_shape, 1)
    print(f"Min size: {out_size}")
    return nn.Sequential(
        Flatten(),
        nn.Linear(out_size, 1024),
        activation_fn(),
        nn.Linear(1024, 256),
        activation_fn(),
        nn.Linear(256, 1),
    )


def create_head_from_cnn(cnn, input_size, activation_fn):
    return create_head(cnn(torch.zeros(1, *input_size)).shape, activation_fn)


def squeeze_loss(loss_fn):
    return lambda x, y: loss_fn(x.view(-1), (y.view(-1)))


def summarize_results(models, root_dir, order_by="mean_absolute_error"):
    df_results = pd.DataFrame()
    for model in models:
        name = model["name"]
        df_path = f"{root_dir}/{name}/history_{name}.csv"
        df = pd.read_csv(df_path, index_col=0)
        max_acc_thresh = df.iloc[df[order_by].idxmax()]
        max_idx = max_acc_thresh.name
        max_acc_thresh = max_acc_thresh.append(
            pd.Series({"name": name, "epoch": max_idx})
        )
        df_results = df_results.append(max_acc_thresh, ignore_index=True)

    df_results = df_results[["name", "epoch", order_by, "valid_loss", "train_loss"]]
    dest_file = f"{root_dir}/summary.csv"

    print(f"Saving to {dest_file}")
    df_results.to_csv(dest_file)


def save_stats(learn):
    p = learn.recorder.plot_losses(return_fig=True)
    p.savefig(path.join(learn.path, "losses"))
    plt.close(p)


def new_grid_search(models, opts, loss_fns):
    return map(lambda el: Grid(*el), cartesian_product(models, opts, loss_fns))
