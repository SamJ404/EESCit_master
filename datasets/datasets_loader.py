from PIL import Image
import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('hr_data_loader')
class HRImgLoader(Dataset):
    def __init__(self, split_file, cache='none'):
        self.cache = cache
        self.files = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                self.files.append(line.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]
        file_name = x

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB')), file_name

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x, file_name

        elif self.cache == 'in_memory':
            return x, file_name



@register('pair_data_loader')
class PairImgLoader(Dataset):
    def __init__(self, meta_file):

        self.files = []
        with open(meta_file, 'r') as f:
            for line in f.readlines():
                self.files.append(line.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pair = self.files[idx]
        lr_name, hr_name = pair.split(',')

        lr = transforms.ToTensor()(Image.open(lr_name).convert('RGB'))
        hr = transforms.ToTensor()(Image.open(hr_name).convert('RGB'))

        return lr,hr

