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
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
from augmentations import get_aug
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

import pickle

class SequentialCIFAR10Shuffled(ContinualDataset):

    NAME = 'seq-cifar10-shuffled'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]

    def get_data_loaders(self, args):
        transform = get_aug(train=True, mean_std=self.cifar_norm, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, mean_std=self.cifar_norm, **args.aug_kwargs)

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

        with open("data/CIFAR10/cifar-10-batches-py/batches.meta","rb") as file_handle:
            retrieved_data = pickle.load(file_handle)
        
        # selected_classes = [['bird', 'cat'], ['deer', 'dog'], ['airplane', 'automobile'], ['frog', 'horse'], ['ship', 'truck']]
        selected_classes = [['dog', 'cat'], ['deer', 'horse'], ['airplane', 'ship'], ['frog', 'bird'], ['automobile', 'truck']]
        class_indexes = [[retrieved_data['label_names'].index(name) for name in names] for names in selected_classes]

        print(selected_classes[self.i])
        train_mask = np.isin(np.array(train_dataset.targets), class_indexes[self.i])
        test_mask = np.isin(np.array(test_dataset.targets), class_indexes[self.i])
        # train_mask = np.logical_or(np.array(train_dataset.targets) == class_indexes[self.i][0],
        #     np.array(train_dataset.targets) == class_indexes[self.i][1])
        # test_mask = np.logical_or(np.array(test_dataset.targets) == class_indexes[self.i][0],
        #     np.array(test_dataset.targets) == class_indexes[self.i][1])
        
        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        target_set = set(train_dataset.targets)
        target_dict = {tgt: i + self.i * len(target_set) for i, tgt in enumerate(target_set)}
        print(target_dict)

        train_dataset.targets = np.array([target_dict[tgt] for tgt in train_dataset.targets])

        test_dataset.targets = np.array(test_dataset.targets)[test_mask]
        test_dataset.targets = np.array([target_dict[tgt] for tgt in test_dataset.targets])

        memory_dataset.data = memory_dataset.data[train_mask]
        memory_dataset.targets = np.array(memory_dataset.targets)[train_mask]
        memory_dataset.targets = np.array([target_dict[tgt] for tgt in memory_dataset.targets])

        train_loader = DataLoader(train_dataset,
                                batch_size=self.args.train.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset,
                                batch_size=self.args.train.batch_size, shuffle=False, num_workers=4)
        memory_loader = DataLoader(memory_dataset,
                                batch_size=self.args.train.batch_size, shuffle=False, num_workers=4)

        self.test_loaders.append(test_loader)
        self.train_loaders.append(train_loader)
        self.memory_loaders.append(memory_loader)
        self.train_loader = train_loader

        self.i += 1
        # train, memory, test = store_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train_loader, memory_loader, test_loader
    
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
