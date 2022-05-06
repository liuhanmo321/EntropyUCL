# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
# from augmentations import get_aug
from PIL import Image


class SequentialMNIST(ContinualDataset):

    NAME = 'seq-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
   
    def get_data_loaders(self, args):

        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

        mean = (0.1,)
        std = (0.2752,)
        train_dataset = MNIST(base_path() + 'MNIST', train=True,
                                  download=True, transform=transform)
        
        memory_dataset = MNIST(base_path() + 'MNIST', train=True,
                                  download=True, transform=transform)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            test_dataset = MNIST(base_path() + 'MNIST',train=False,
                                   download=True, transform=transform)

        train, memory, test = store_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test
    
    def get_transform(self, ags):
        mean = (0.1,)
        std = (0.2752,)
        transform = transforms.Compose(
                [transforms.Pad(padding=2,fill=0),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                transforms.Normalize(mean, std)
                ])
        return transform

    def not_aug_dataloader(self, batch_size):
        mean = (0.1,)
        std = (0.2752,)
        transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize(mean, std)])

        train_dataset = MNIST(base_path() + 'MNIST', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

def get_aug(name='simsiam', image_size=28, train=True, train_classifier=None):
    if train==True:
        augmentation = SimSiamTransform(image_size)
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation


class SimSiamTransform():
    def __init__(self, image_size, mean_std=[[0.1], [0.2752]]):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

        # self.transform = transforms.Compose([
        #     # T.RandomCrop(64, padding=4),
        #     transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
        #     transforms.ToTensor(),
        #     transforms.Normalize(*mean_std)
        # ])

        self.transform = transforms.Compose([
                    transforms.Pad(padding=2,fill=0),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize(*mean_std)
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        not_aug_x = self.not_aug_transform(x)
        return x1, x2, not_aug_x

class Transform_single():
    def __init__(self, image_size, train, mean_std=[[0.1], [0.2752]]):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                # transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
            ])

    def __call__(self, x):
        return self.transform(x)