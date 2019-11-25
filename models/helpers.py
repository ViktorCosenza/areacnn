from torch import nn
from torch.nn import functional as F
from os import path
import pandas as pd

class SumPool2d(nn.Module):
    def __init__(self, pool_size):
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

def save_stats(learn, name):
    p = learn.recorder.plot_losses(return_fig=True)
    p.savefig(path.join(learn.path, 'losses'))
