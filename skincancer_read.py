import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset

def get_data(base_dir, imageid_path_dict):
    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    # df_original.head()

    df_original[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']

    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)


    # This set will be df_original excluding all rows that are in the val set
    # This function identifies if an image is part of the train or val set.
    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']

    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [15,10,5,50,0,40,5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
    df_train['cell_type'].value_counts()

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    return df_train, df_val



def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y