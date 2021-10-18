import os
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
class CustomSequence(data.Dataset):
    def __init__(self,
                 data_path,
                 seq_idxs,
                 mode,
                 seq_length,
                 stat_dict,
                 X_names = None,
                 Y_names = None,
        ):
        super(CustomSequence, self).__init__()
        self.data_path = data_path
        self.seq_idxs = seq_idxs
        self.mode = mode
        self.seq_length=seq_length
        self.to_tensor = transforms.ToTensor()
        self.X_names = X_names
        self.Y_names = Y_names
        self.XdataFrame = pd.read_csv(data_path)[self.X_names]
        self.Xdata_np = np.array(self.XdataFrame)

        self.YdataFrame = pd.read_csv(data_path)[self.Y_names]
        self.Ydata_np = np.array(self.YdataFrame)

        # print('--> YDATA nan: {}'.format(np.isnan(self.Ydata_np).any()))
        # np.where(self.Ydata_np==np.nan)
        #
        # print('--> XDATA nan: {}'.format(np.isnan(self.Xdata_np).any()))
        # print(np.where(np.isnan(self.Xdata_np)))


        self.X_means = np.nanmean(self.Xdata_np,axis=0)
        self.X_stds = np.nanstd(self.Xdata_np,axis=0)

        self.Y_means = np.nanmean(self.Ydata_np,axis=0)
        self.Y_stds = np.nanstd(self.Ydata_np,axis=0)

        stat_dict['X_means'] = self.X_means
        stat_dict['X_stds'] = self.X_stds
        stat_dict['Y_means'] = self.Y_means
        stat_dict['Y_stds'] = self.Y_stds


        # self.Xdata_np =(self.Xdata_np - self.X_means)/self.X_stds
        # self.Ydata_np = (self.Ydata_np - self.Y_means)/self.Y_stds





        self.Xmin = np.nanmin(self.Xdata_np,axis=0)
        self.Xmax = np.nanmax(self.Xdata_np,axis=0)


        self.Ymin = np.nanmin(self.Ydata_np,axis=0)
        self.Ymax = np.nanmax(self.Ydata_np,axis=0)

        stat_dict['Xmax'] = self.Xmax
        stat_dict['Xmin'] = self.Xmin
        stat_dict['Ymax'] = self.Ymax
        stat_dict['Ymin'] = self.Ymin


        self.Xdata_np =(self.Xdata_np - self.Xmin)/(self.Xmax- self.Xmin)
        self.Ydata_np =(self.Ydata_np - self.Ymin)/(self.Ymax- self.Ymin)





    def __len__(self):
        return len(self.seq_idxs)

    def __getitem__(self, index):# index is generated in epoch process
        inputs = {}

        seq_idx = int(self.seq_idxs[index]) #get row_num



        inputs['seqs']= self.get_seqs(seq_idx)
        inputs['gts']= self.get_gts(seq_idx)




        return inputs

    def get_seqs(self,seq_idx):#X
        ret = self.Xdata_np[seq_idx:seq_idx+self.seq_length]
        return torch.tensor(ret,dtype=torch.float32)
    def get_gts(self,seq_idx):#Y
        ret = self.Ydata_np[seq_idx+self.seq_length:seq_idx + 2*self.seq_length]
        return torch.tensor(ret,dtype=torch.float32)

