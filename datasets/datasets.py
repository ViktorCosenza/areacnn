from fastai.vision import ImageList, FloatList

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os import path
import pandas as pd
from collections import namedtuple

class PolygonDataset(Dataset):
    def __init__(self, root_dir, df_path, transform=lambda x: x):
        self.root_dir = root_dir
        self.df = pd.read_csv(df_path, index_col=0)
        self.transform = transform
    
    def __getitem__(self, i):
        item = self.df.loc[i]
        filename, label = item["filename"], item["label"]
    
        return (
            self.transform(Image.open(path.join(self.root_dir, 'images', filename))),
            label
        )
    def __len__(self):
        return len(self.df)

    
def get_dataset(root_dir, df_path, bs, **kwargs):
    train_dir = path.join(root_dir, 'train')
    test_dir = path.join(root_dir, 'test')
    return namedtuple('Dl', 'train test')(
        lambda: DataLoader(
            PolygonDataset(
                train_dir, 
                path.join(train_dir, 'data.csv'), 
                **kwargs), 
            batch_size=bs),
        lambda: DataLoader(
            PolygonDataset(
                test_dir, 
                path.join(test_dir, 'data.csv'), 
                **kwargs), 
            batch_size=bs)
    )
