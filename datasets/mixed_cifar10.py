# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_mixed_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
from augmentations import get_aug
from PIL import Image


class MixedCIFAR10(ContinualDataset):

    NAME = 'mixed-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
   
    def get_data_loaders(self, args):
        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

        train_dataset = CIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        
        memory_dataset = CIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=test_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)

        train, memory, test = store_mixed_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test
    
    def get_transform(self, args):
        cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        if args.cl_default:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*cifar_norm)
                ])
        else:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*cifar_norm)
                ])

        return transform

    def not_aug_dataloader(self, batch_size):
        cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize(*cifar_norm)])

        train_dataset = CIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader


# class FuzzyCLDataLoader(object):
#     def __init__(self, datasets_per_task, batch_size, train=True):
#         bs = batch_size if train else 64
#         self.raw_datasets = datasets_per_task
#         self.datasets = [_ for _ in datasets_per_task]
#         for i in range(len(self.datasets) - 1):
#             self.datasets[i], self.datasets[i + 1] = self.mix_two_datasets(self.datasets[i], self.datasets[i + 1])
#         self.loaders = [
#                 torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=train, num_workers=0)
#                 for x in self.datasets ]

#     def shuffle(self, x, y):
#         perm = np.random.permutation(len(x))
#         x = x[perm]
#         y = y[perm]
#         return x, y

#     def mix_two_datasets(self, a, b, start=0.5):
#         a.x, a.y = self.shuffle(a.x, a.y)
#         b.x, b.y = self.shuffle(b.x, b.y)

#         def cmf_examples(i):
#             if i < start * len(a):
#                 return 0
#             else:
#                 return (1 - start) * len(a) * 0.25 * ((i / len(a) - start) / (1 - start)) ** 2

#         s, swaps = 0, []
#         for i in range(len(a)):
#             c = cmf_examples(i)
#             if s < c:
#                 swaps.append(i)
#                 s += 1

#         for idx in swaps:
#             a.x[idx], b.x[len(b) - idx], a.y[idx], b.y[len(b) - idx] = b.x[len(b) - idx], a.x[idx], b.y[len(b) - idx], a.y[idx]
#         return a, b

#     def __getitem__(self, idx):
#         return self.loaders[idx]

#     def __len__(self):
#         return len(self.loaders)