from torch import nn
from torch.nn import functional as F
from os import path
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from itertools import product as cartesian_product
from collections import namedtuple

Param = namedtuple('Param', 'name param')

def squeeze_loss(loss_fn): return lambda x, y: loss_fn(x.view(-1), (y.view(-1)))

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

    df_results = df_results[['name', 'epoch', order_by, 'valid_loss', 'train_loss']]
    dest_file = f"{root_dir}/summary.csv"
    
    print(f"Saving to {dest_file}")
    df_results.to_csv(dest_file)

def save_stats(learn):
    p = learn.recorder.plot_losses(return_fig=True)
    p.savefig(path.join(learn.path, 'losses'))
    plt.close(p)


def new_grid_search(models, opts, loss_fns):
    Grid = namedtuple('Grid', 'model opt loss')
    return map(lambda el: Grid(*el), cartesian_product(models, opts, loss_fns))


def train_epoch(dl, model, opt, loss_fn, device, show_progress):
    total_loss = 0
    model.train()
    for example, label in (tqdm(dl) if show_progress else dl):
        example, label = example.to(device), label.to(device)
        opt.zero_grad()
        output = model(example)
        loss = loss_fn(output, label)
        loss.backward()
        opt.step()
        
        total_loss += loss.detach().item()
    return total_loss, total_loss / len(dl) 


def validate(dl, model, loss_fn, device):
    total_loss = 0
    model.eval()
    for example, label in dl:
        example, label = example.to(device), label.to(device)
        output = model(example)
        loss = loss_fn(output, label)
        total_loss += loss.item()
    return total_loss, total_loss / len(dl)


def train(dl_train, dl_val, opt_func, loss_fn, model, epochs, device, show_progress):
    model = model.to(device)
    opt = opt_func(model.parameters())
    metrics = []
    for epoch in (tqdm(range(epochs)) if show_progress else range(epochs)):
        train_loss, train_loss_avg = train_epoch(dl_train, model, opt, loss_fn, device, show_progress)
        val_loss, val_loss_avg = validate(dl_val, model, loss_fn, device)
        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_loss_avg": train_loss_avg,
            "val_loss_avg": val_loss_avg
        })
    return metrics
    