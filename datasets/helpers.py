from PIL import Image, ImageStat
from PIL.ImageDraw import ImageDraw
from os import path
import os
import pandas as pd
from random import randint
import pandas as pd

def random_point(w, h, minx=0, miny=0):
    x, y = randint(minx, w), randint(miny, h)
    return (x, y)

def area_ellipse(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x_len = abs(x1 - x2)
    y_len = abs(y1 - y2)
    return pi * x_len * y_len

def draw_rectangle(w, h, draw):
    p1 = random_point(w, h)
    p2 = random_point(w, h, p1[0], p1[1])
    draw.rectangle([p1, p2], fill=1)

def draw_ellipse(w, h, draw):
    p1 = random_point(w, h)
    p2 = random_point(w, h, p1[0], p1[1])
    draw.ellipse([p1, p2], fill=1)
    
def gen_example(w, h, num_polygons=1, count=False, draw_polygon_fn=draw_rectangle):
    example = Image.new('1', (w, h))
    draw = ImageDraw(example, mode='1')

    total_area = w * h
    area = 0
    for i in range(num_polygons):
        draw_polygon_fn(w, h, draw)
    (area, ) = ImageStat.Stat(example).sum
    if not count:
        area /= total_area
    return (example, area)

def gen_df(
        root_dir, 
        dt_len, 
        img_size=(512, 512), 
        skip=True, 
        test=False, 
        count=False,
        draw_polygon_fn=draw_rectangle):
    root_dir = path.abspath(root_dir)
    root_dir = path.join(root_dir, "test" if test else "train")
    img_dir = path.join(root_dir, 'images')
    df_dest = path.join(root_dir, 'data.csv')
    if path.exists(df_dest) and skip:
        print("Found existing dataset, skipping...")
        return pd.read_csv(df_dest, index_col=0)
    
    for directory in [root_dir, img_dir]:
        if not path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory {directory}")

    df = pd.DataFrame(columns=['filename', 'area'])
    for i in range(dt_len):
        filename = f"img_{i}.jpeg"
        dest_path = path.join(img_dir, filename)
        img, area = gen_example(*img_size, count=count, draw_polygon_fn=draw_polygon_fn)
        img.save(dest_path)
        row = pd.Series({"filename": filename, "area": area})
        df.loc[i] = row

    df.to_csv(df_dest)
    return df
