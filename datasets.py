import numpy as np
import random
from PIL import Image
import glob
import os

import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

class ImageTransform():
    """
    Args:
        resize = 224 : paramester from imagenet
        mean = (0.485, 0.456, 0.406) : paramester from imagenet
        std = (0.229, 0.224, 0.225): paramester from imagent
    """
    # size = 224
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose(
                [
                    # transforms.RandomPerspective(distortion_scale=0.6, p=0.4),
                    transforms.RandomHorizontalFlip(0.3),
                    transforms.RandomVerticalFlip(0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            ),
            'val': transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
        }

    def __call__(self, img, phase='train'):
        """
        Args:
            img: image iput
            phase: 'train' or 'val', default 'train'
        Return:
            image transformed with phase was adjusted
        """
        return self.data_transform[phase](img)

class TargetsTransform():
    def __call__(self, target):
        """
        TODO
        """
        pass

class MyDataset(Dataset):
    """
    Args:
        file_list: list of paths
        transform: using transformation with images
        target_transform: using transformation with labels
        phase: 'train' use for training data, 'val' use for valdiation data
    Return:
        Tuple of (datas, targets)
        Datas are transformed, targets are transformed and encoded to one-hot
    """
    def __init__(self, file_list, transform=None, target_transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.target_transform = target_transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data = Image.open(img_path)
        target = self.file_list[idx].split('/')[-2]
        if target == '0000':
            target = 0
        elif target == '1000':
            target = 1
        elif target == '5000':
            target = 2
        elif target == '20000':
            target = 3
        elif target == '50000':
            target = 4

        if self.transform:
            data = self.transform(data, phase=self.phase)
        if self.target_transform:
            target = self.target_transform(target)

        return data, target

def make_datapath_list():
    """
    Return:
         List of image path
    """
    rootpath = './data/'
    target_path = os.path.join(rootpath, '**/*.png')

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list
