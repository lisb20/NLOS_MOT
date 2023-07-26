import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from scipy.io import loadmat
import pdb

from .loader import npy_loader


class TrackingDataset(Dataset):
    def __init__(self,
                 dataset_root: str,
                 data_type: str = 'render',
                 route_len: int = 128,
                 use_fileclient: bool = False,
                 noisy: bool = True,
                 max_peo: int = 3,
                 ** kwags) -> None:
        self.dataset_root = dataset_root
        self.max_peo = max_peo
        self.num_frames = route_len
        if data_type == 'render':
            self.total_len = 256 
            self.npy_name = 'video_128_noisy.npy' if noisy else 'video_128.npy'
        elif data_type == 'real_shot':
            self.total_len = 256 
            self.npy_name = 'video_128.npy'


        self.dataset_dir = dataset_root
        self.dirs = []
        for peo in [1,2,3]:
            tmp_dir = [os.path.join(str(peo), d) for d in os.listdir(os.path.join(self.dataset_dir, str(peo)))]
            self.dirs += tmp_dir[:600]
        
        print('dirs: ', len(self.dirs))
        pdb.set_trace()

        if use_fileclient:
            self.npy_loader = npy_loader()
            self.load_npy = self.npy_loader.get_item
        else:
            self.load_npy = np.load

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        abs_png_dir = os.path.join(self.dataset_dir, self.dirs[idx])
        npy_file = os.path.join(abs_png_dir, self.npy_name)
        video = self.load_npy(npy_file)

        start_frame = random.randint(0, self.total_len - self.num_frames)
        video = video[:, start_frame:start_frame + self.num_frames]  # (3, T, H, W) or (3, T-1, H, W)

        mat_file = loadmat(os.path.join(abs_png_dir, 'route.mat'))

        route = mat_file['route'][start_frame:start_frame + self.num_frames]  # (T,)
        route = route.reshape((route.shape[0], -1))  # (T, 2n)        
        ## route: T*2n 对齐人数填充为 T*10。空缺填
        ## route 按照奇数行起始点排序 ----route[0:1]为起始最靠左的人
        npeo = route.shape[1] // 2
        avg = []
        # for p in range(npeo):
        #     avg.append({"st_x":route[0,2*p],"idx":p})
        
        # avg.sort(key=lambda x:x["st_x"])
        # tmp = np.zeros((route.shape[0], route.shape[1]))
        # for i in range(npeo):
        #     tmp[:,2*i:2*i+2] = route[:,2*avg[i]["idx"]:2*avg[i]["idx"]+2]
        # route = tmp
        # print('route', sum(route[:,0]), sum(route[:,2]))
        for p in range(npeo):
            avg.append({"sumx":sum(route[:,2*p]),"idx":p})
        avg.sort(key=lambda x:x["sumx"])
        tmp = np.zeros((route.shape[0], route.shape[1]))
        for i in range(npeo):
            tmp[:,2*i:2*i+2] = route[:,2*avg[i]["idx"]:2*avg[i]["idx"]+2]
        route = tmp

        route = np.concatenate((route, np.ones((route.shape[0], self.max_peo * 2 - route.shape[1])) * 0.5), axis=1) # (T，10)
        assert route.shape[1] == self.max_peo * 2 and route.shape[0] == self.num_frames
        map_size = mat_file['map_size']  # (1, 2)
        ## mapsize ->[mapsize * 5]
        map_size = np.tile(map_size, (1, self.max_peo))  # (1,10)

        return torch.from_numpy(video), torch.from_numpy(route).float(), torch.from_numpy(map_size).float()


def split_dataset(phase: str = 'train', train_ratio: float = 0.8, **kwargs):
    full_dataset = TrackingDataset(**kwargs)

    if phase == 'train':
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, val_size])
    elif phase == 'test':
        return full_dataset
