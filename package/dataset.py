import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from itertools import cycle
from typing import List, Union, Tuple, Any
import cv2
from PIL import Image

import torch
import torchvision
from torchvision import transforms, io
from torchvision.transforms import functional as F

import pydicom

import random
np.random.seed(2022)
random.seed(2022)
torch.manual_seed(2022)


ROOT = Path(__file__).parent.resolve()



class DicomDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.list = pd.read_csv( str(ROOT/ "train.csv") )

        # edit file path
        self.list.FilePath = self.list.FilePath.apply(lambda _: ROOT / _[1:])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        # print(f"idx: {idx}")
        # print(f"selected slice index: {self.list.loc[idx, 'index']}")
        dcm = pydicom.read_file( str(self.list.FilePath[idx]) )
        
        # label
        label = int(self.list.Stage[idx])

        # Preprocessed Pixels: window, normalization, totensor, 3 channel
        pixel = dcm.pixel_array[ self.list.loc[idx, 'index'] ] # 用 index 當 column name 真的是天才
        low, high = self.get_low_high(dcm)
        pixeled = self.getWindow(pixel, low, high)
        img = (pixeled - np.min(pixeled)) / (np.max(pixeled) - np.min(pixeled))
        img = torch.tensor(img)
        img = torch.stack([img, img, img], dim=0)

        seed = np.random.randint(1e9)
        random.seed(seed)
        torch.manual_seed(seed)

        img = self.transform(img)

        return img, label

    @staticmethod
    def get_low_high(dcm):
        window_center = dcm.WindowCenter
        window_width = dcm.WindowWidth
        if isinstance(dcm.WindowCenter, pydicom.multival.MultiValue):
            window_center = dcm.WindowCenter[0]
            window_width = dcm.WindowWidth[0]
        low = window_center - window_width / 2
        high = window_center + window_width / 2

        return low, high


    @staticmethod
    def getWindow(img, low, high):
        if low > 33000 and low < 40000:
            img[img> 34050.0] = 34050.0
            img[img< 33600.0] = 33600.0
        elif low > 30000 and low < 33000:
            img[img> 33008.0] = 33008.0
            img[img< 32558.0] = 32558.0
        elif low > 500 and low < 1500:
            img[img > 1280] = 1300
            img[img < 830] = 830
        elif low < -100:
            img[img > 250] = 250
            img[img < -200] = -200
        else:
            img[img > high] = high
            img[img < low] = low

        return img