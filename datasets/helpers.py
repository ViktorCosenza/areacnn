from PIL import Image, ImageStat
from PIL.ImageDraw import ImageDraw
from os import path
import os
import pandas as pd
from random import randint, choice
import pandas as pd

from functools import partial


def random_point(w, h, minx=0, miny=0):
    x, y = randint(minx, w), randint(miny, h)
    return (x, y)


def random_point_pair(w, h):
    p1 = random_point(w, h)
    p2 = random_point(w, h, p1[0], p1[1])
    return [p1, p2]

def area_ellipse(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x_len = abs(x1 - x2)
    y_len = abs(y1 - y2)
    return pi * x_len * y_len

def draw_rectangle(p1, p2, draw, fill):
    draw.rectangle([p1, p2], fill=fill)
    
def draw_ellipse(p1, p2, draw, fill):
    draw.ellipse([p1, p2], fill)
    
def draw_random_rectangle(w, h, draw, fill=1):
    p1, p2 = random_point_pair(w, h)
    draw_rectangle(p1, p2, draw, fill)


def draw_random_ellipse(w, h, draw, fill=1):
    p1, p2 = random_point_pair(w, h)
    draw_ellipse(p1, p2, draw, fill)

def gen_random_color_example(w, h, count, draw_polygon_fns, max_polygons, min_polygons=1):
    im = Image.new('RGB', (w, h))
    grey_im = Image.new('1', (w, h))
    draw = ImageDraw(im, 'RGB')
    grey_draw = ImageDraw(grey_im, '1')
    for _ in range(randint(min_polygons, max_polygons)):
        p1, p2 = random_point_pair(w, h)
        draw_fn = partial(choice(draw_polygon_fns), p1, p2)
        draw_fn(draw, fill=tuple(randint(0, 255) for _ in range(3)))
        draw_fn(grey_draw, fill=1)
    (area, ) = ImageStat.Stat(grey_im).sum   
    total_area = w * h
    if not count:
        area /= total_area
    return (im, area)
    
def gen_binary_example(w, h, count, draw_polygon_fn, num_polygons=1):
    example = Image.new("1", (w, h))
    draw = ImageDraw(example, mode="1")

    for i in range(num_polygons):
        draw_polygon_fn(w, h, draw)
    (area,) = ImageStat.Stat(example).sum
    total_area = w * h
    if not count:
        area /= total_area
    return (example, area)


def gen_df(
    root_dir,
    dt_len,
    gen_example_fn,
    img_size=(512, 512),
    skip=True,
    test=False,
):
    root_dir = path.abspath(root_dir)
    root_dir = path.join(root_dir, "test" if test else "train")
    img_dir = path.join(root_dir, "images")
    df_dest = path.join(root_dir, "data.csv")
    if path.exists(df_dest) and skip:
        print(f"Found existing dataset, skipping for {root_dir}...")
        return pd.read_csv(df_dest, index_col=0)

    for directory in [root_dir, img_dir]:
        if not path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory {directory}")

    df = pd.DataFrame(columns=["filename", "label"])
    for i in range(dt_len):
        filename = f"img_{i}.jpeg"
        dest_path = path.join(img_dir, filename)
        img, area = gen_example_fn(*img_size)
        img.save(dest_path)
        row = pd.Series({"filename": filename, "label": area})
        df.loc[i] = row

    df.to_csv(df_dest)
    return df
