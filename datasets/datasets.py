from fastai.vision import ImageList, FloatList
from os import path

def get_dataset(train_dir, test_dir, df_train, df_test, bs, device,transform_args):
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