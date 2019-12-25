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
from tqdm import tqdm
from functools import partial

import pandas as pd
import random
from os import path
import os

# +
ROOT_DIR = path.join("/home", "victor", "datasets")

DT_ROOT = path.join(ROOT_DIR, "VOC")
DT_DEST_BINARY = path.join(ROOT_DIR, "VOC_FORMS")
DT_DEST_RGB_RANDOM = path.join(ROOT_DIR, "VOC_FORMS_RGB")
DT_DEST_RGB_SINGLE_CLASS = lambda c: path.join(ROOT_DIR, f"VOC_FORMS_RGB_{c.upper()}")

object_categories = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# +
def gen_example_from_voc(voc):
    for example, segmentation in voc:
        im = Image.fromarray(np.array(segmentation) > 0).resize((512, 512)).convert("1")
        area = np.sum(im)
        yield (im, area)


def random_class_mask_generator(voc):
    for example, segmentation in voc:
        present_labels = np.setdiff1d(np.unique(segmentation), [0, 255])
        chosen = np.random.choice(present_labels)
        background = Image.fromarray(
            (np.asarray(segmentation) != chosen).astype(np.bool)
        )
        example.paste(0, mask=background)
        area = np.logical_not(background).sum()
        yield (example, area)


def class_mask_generator(voc, cl):
    for example, segmentation in voc:
        present_labels = np.setdiff1d(np.unique(segmentation), [0, 255])
        background = Image.fromarray((np.asarray(segmentation) != cl).astype(np.bool))
        area = np.logical_not(background).sum()
        if area == 0:
            continue

        example.paste(0, mask=background)
        yield (example, area)


def gen_df_from_voc(root_dir, dt, generator_fn, skip=True):
    root_dir = path.abspath(root_dir)
    img_dir = path.join(root_dir, "images")
    df_dest = path.join(root_dir, "data.csv")
    if skip and path.exists(df_dest):
        print(f"Found existing dataset, skipping for {root_dir}...")
        return pd.read_csv(df_dest, index_col=0)

    for directory in [root_dir, img_dir]:
        if not path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory {directory}")

    df = pd.DataFrame(columns=["filename", "label"])
    for i, (img, area) in enumerate(generator_fn(dt)):
        filename = f"img_{i}.jpeg"
        dest_path = path.join(img_dir, filename)
        img.save(dest_path)
        row = pd.Series({"filename": filename, "label": area})
        df.loc[i] = row

    df.to_csv(df_dest)
    return df


# +
def main():
    dt_train = torchvision.datasets.VOCSegmentation(
        root=path.join(DT_ROOT, "train"), download=False, image_set="train"
    )

    dt_val = torchvision.datasets.VOCSegmentation(
        root=path.join(DT_ROOT, "test"), download=False, image_set="val"
    )

    ## Binary ##
    gen_df_from_voc(
        path.join(DT_DEST_BINARY, "train"), dt_train, generator_fn=gen_example_from_voc
    )
    gen_df_from_voc(
        path.join(DT_DEST_BINARY, "val"), dt_val, generator_fn=gen_example_from_voc
    )

    ## RGB RANDOM CLASS ##
    gen_df_from_voc(
        path.join(DT_DEST_RGB_RANDOM, "train"),
        dt_train,
        generator_fn=random_class_mask_generator,
    )
    gen_df_from_voc(
        path.join(DT_DEST_RGB_RANDOM, "val"),
        dt_val,
        generator_fn=random_class_mask_generator,
    )

    ## RGB SINGLE CLASS ##
    for cl in tqdm(object_categories):
        generator_fn = partial(class_mask_generator, cl=1 + object_categories.index(cl))
        gen_df_from_voc(
            path.join(DT_DEST_RGB_SINGLE_CLASS(cl), "train"),
            dt_train,
            generator_fn=generator_fn,
        )
        gen_df_from_voc(
            path.join(DT_DEST_RGB_SINGLE_CLASS(cl), "val"),
            dt_val,
            generator_fn=generator_fn,
        )


if __name__ == "__main__":
    main()
