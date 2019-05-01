import os
import numpy as np
from utils import pil_loader
from torch.utils.data import Dataset


def build_dataset(filelist, prefix):
    img_lst = []
    lb_lst = []
    lb_max = -1
    with open(filelist) as f:
        for x in f.readlines():
            n, lb = x.strip().split(' ')
            lb = int(lb)
            lb_max = max(lb_max, lb)
            img_lst.append(os.path.join(prefix, n))
            lb_lst.append(lb)
    assert len(img_lst) == len(lb_lst)
    return img_lst, lb_lst, lb_max


class FileListDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None):
        self.img_lst, self.lb_lst, self.num_classes \
            = build_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform

    def __len__(self):
        return self.num

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = self.img_lst[idx]
        lb = self.lb_lst[idx]
        try:
            img = pil_loader(open(fn, 'rb').read())
            return img, lb
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        img, lb = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, lb
