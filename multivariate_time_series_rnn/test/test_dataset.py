import unittest
from datasets import CustomSequence
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import torch.nn as nn

class TestDataset(unittest.TestCase):

    def test_myrdataset(self):
        dataset = CustomSequence(
            data_path='/home/roit/datasets/huawei2021/pb-data2/post/A_h_CW_detection_post.csv',
            seq_idxs=[6,8,11,43,88,76,44,12,34,235],
            mode='train',
            seq_length=15,
            X_names=['Tem','Hum','Ap','Wv'],#,'Wx','Wy'],
            Y_names=['SO2','CO','O3','NO2','PM10','PM2.5'],
            stat_dict={}
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True
        )
        print(len(data_loader))

        for idx,item in enumerate(data_loader):
            print(idx,item)


    def test_dataset_splits(self):
        pass

