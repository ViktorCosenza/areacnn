# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from tqdm import tqdm
from os import path

## Local Imports ##
from datasets import helpers as dataset_helpers

# + [markdown] colab_type="text" id="QLVyf1-umCV2"
# ## Dataset

# +
ROOT_DIR = "data"
DT_LEN = 20000
W, H = 32, 32

DATASETS = [
    {
        "root_dir": path.join(ROOT_DIR, "polygon_data_counts"),
        "count": True,
        "skip": False,
    },
    {
        "root_dir": path.join(ROOT_DIR, "polygon_data_percentage"),
        "count": False,
        "skip": False,
    },
    {
        "root_dir": path.join(ROOT_DIR, "ellipse_data_counts"),
        "count": True,
        "skip": False,
        "draw_polygon_fn": dataset_helpers.draw_ellipse,
    },
    {
        "root_dir": path.join(ROOT_DIR, "ellipse_data_percentage"),
        "count": False,
        "skip": False,
        "draw_polygon_fn": dataset_helpers.draw_ellipse,
    },
]

for dataset in tqdm(DATASETS):
    df_train = dataset_helpers.gen_df(
        dt_len=DT_LEN, test=False, img_size=(W, H), **dataset
    )
    df_test = dataset_helpers.gen_df(
        dt_len=int(DT_LEN * 0.8), test=True, img_size=(W, H), **dataset
    )
print("DONE!")
