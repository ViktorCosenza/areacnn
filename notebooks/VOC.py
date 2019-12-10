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
import torch
import torchvision
import numpy as np
from PIL import Image

import pandas as pd

from os import path
import os

# +
DT_ROOT = path.join('/home', 'victor', 'datasets', 'VOC')
DT_DEST = path.join('/home', 'victor', 'datasets', 'VOC_FORMS')
    
dt_train = torchvision.datasets.VOCSegmentation(
    root=path.join(DT_ROOT, 'train'),
    download=False,
    image_set='train'
)

dt_val = torchvision.datasets.VOCSegmentation(
    root=path.join(DT_ROOT, 'test'),
    download=False,
    image_set='val'
)


# +
def gen_example_from_voc(voc):
    for example, segmentation in voc:
        im = Image.fromarray(np.array(segmentation) > 0).resize((512, 512)).convert('1')
        area = np.sum(im)
        yield (im, area)
        
def gen_df_from_voc(root_dir, dt, skip=True):
    root_dir = path.abspath(root_dir)
    img_dir = path.join(root_dir, 'images')
    df_dest = path.join(root_dir, 'data.csv')
    if path.exists(df_dest) and skip:
        print(f"Found existing dataset, skipping for {root_dir}...")
        return pd.read_csv(df_dest, index_col=0)
    
    for directory in [root_dir, img_dir]:
        if not path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory {directory}")

    df = pd.DataFrame(columns=['filename', 'label'])
    for i, (img, area) in enumerate(gen_example_from_voc(dt)):
        filename = f"img_{i}.jpeg"
        dest_path = path.join(img_dir, filename)
        img.save(dest_path)
        row = pd.Series({"filename": filename, "label": area})
        df.loc[i] = row

    df.to_csv(df_dest)
    return df


# -

gen_df_from_voc(path.join(DT_DEST, 'train'), dt_train)
gen_df_from_voc(path.join(DT_DEST, 'val'), dt_val)


