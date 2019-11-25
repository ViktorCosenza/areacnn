from fastai.vision import ImageList, FloatList
from os import path
import pandas as pd

def gen_dataset_options(
        root_dir, 
        bs, 
        device, 
        transform_args,
        train_dir='train', 
        test_dir='test', 
        csv='data.csv'):
    return {
        "train_dir": path.join(root_dir, 'train', 'images'),
        "test_dir": path.join(root_dir, 'test', 'images'),
        "df_train": pd.read_csv(
                path.join(root_dir, 'train', 'data.csv'),
                index_col=0
            ),
        "df_test": pd.read_csv(
                path.join(root_dir, 'train', 'data.csv'),
                index_col=0
            ),
        "bs": bs,
        "transform_args": transform_args,
        "device": "cuda"
    }
    

def get_dataset(train_dir, test_dir, df_train, df_test, bs, device, transform_args):
    test_item_list = (ImageList
        .from_df(path=test_dir, df=df_test, convert_mode='1')
    )
    return (ImageList
            .from_df(path=train_dir, df=df_train, convert_mode='1')
            .split_by_rand_pct()
            .label_from_df(cols=1, label_cls=FloatList)
            .transform(**transform_args)
            .add_test(test_item_list)
            .databunch(bs=bs, device=device)  
        )