# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.seq_mnist import SequentialMNIST
from datasets.seq_fmnist import SequentialFMNIST
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar10_shuffled import SequentialCIFAR10Shuffled
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_cifar100_sub import SequentialCIFAR100Sub
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.seq_svhn import SequentialSVHN
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
from datasets.mixed_cifar10 import MixedCIFAR10
import torchvision

NAMES = {
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialFMNIST.NAME: SequentialFMNIST,
    SequentialSVHN.NAME: SequentialSVHN,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    MixedCIFAR10.NAME: MixedCIFAR10,
    SequentialCIFAR100Sub.NAME: SequentialCIFAR100Sub,
    SequentialCIFAR10Shuffled.NAME: SequentialCIFAR10Shuffled
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset_kwargs['dataset'] in NAMES.keys()
    return NAMES[args.dataset_kwargs['dataset']](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)

def get_dataset_all(args):
    from augmentations import get_aug
    from datasets.seq_tinyimagenet import base_path
    from torch.utils.data import DataLoader

    # if dataset == 'mnist':
    #     dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    # elif dataset == 'stl10':
    #     dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    if args.dataset.name == 'seq-cifar10':
        norm_std = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        transform = get_aug(train=True, mean_std=norm_std, **args.aug_kwargs)
        dataset = torchvision.datasets.CIFAR10(base_path() + 'CIFAR10', train=True, transform=transform, download=True) 
        loader = DataLoader(dataset,
                            batch_size=args.train.batch_size, shuffle=True, num_workers=4)       
    elif args.dataset.name == 'seq-cifar100':
        norm_std = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        transform = get_aug(train=True, mean_std=norm_std, **args.aug_kwargs)
        dataset = torchvision.datasets.CIFAR100(base_path() + 'CIFAR100', train=True, transform=transform, download=True)
        loader = DataLoader(dataset,
                            batch_size=args.train.batch_size, shuffle=True, num_workers=4)
    # elif dataset == 'imagenet':
    #     dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    # elif dataset == 'random':
    #     dataset = RandomDataset()
    else:
        raise NotImplementedError   

    return loader

def get_dataset_off(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    from augmentations import get_aug
    # if dataset == 'mnist':
    #     dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    # elif dataset == 'stl10':
    #     dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    if dataset == 'seq-cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
        norm_std = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
    elif dataset == 'seq-cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
        norm_std = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    # elif dataset == 'random':
    #     dataset = RandomDataset()
    else:
        raise NotImplementedError

    transform = get_aug(train=True, mean_std=norm_std, **args.aug_kwargs)

    return dataset