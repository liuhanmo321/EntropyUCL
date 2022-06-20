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

# data/CIFAR10

import torchvision
import numpy as np
import os
import codecs
from torch.distributions.categorical import Categorical
import torch.utils.data as data
from PIL import Image
import errno

class StreamCIFAR10(ContinualDataset):

    NAME = 'stream-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 1

    def get_data_loaders(self, args):
        selected_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        dataset = [DatasetsLoaders("CIFAR10", selected_classes, args, batch_size=args.train.batch_size,
                                num_workers=args.dataset.num_workers,
                                total_iters=args.train.num_epochs * args.dataset.iter_per_epoch,
                                contpermuted_beta=args.dataset.beta,
                                num_of_batches=args.dataset.iter_per_epoch)]
        test_loaders = [tloader for ds in dataset for tloader in ds.test_loader]
        train_loaders = [ds.train_loader for ds in dataset]
        memory_loaders = [mloader for ds in dataset for mloader in ds.memory_loader]

        self.test_loaders = test_loaders
        self.train_loaders = train_loaders
        self.memory_loaders = memory_loaders
        return train_loaders[0], memory_loaders, test_loaders

    
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
        # cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        # transform = transforms.Compose([transforms.ToTensor(), 
        #         transforms.Normalize(*cifar_norm)])

        # train_dataset = CIFAR10(base_path() + 'CIFAR10', train=True,
        #                           download=True, transform=transform)
        # train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return self.memory_loaders[0]


class SubDataSet(torch.utils.data.Dataset):
    """
    Obtain the sub data sets with respect to classes
    """
    def __init__(self, dataset, selected_classes):
        super(SubDataSet,self).__init__()
        self.selected_classes = selected_classes
        self.dataset = [dataset[i] for i in range(len(dataset)) if dataset[i][1] in selected_classes]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class DatasetsLoaders:
    def __init__(self, dataset, selected_classes, args, num_of_batches=400, batch_size=4, num_workers=4, pin_memory=True, **kwargs):
        self.dataset_name = dataset
        self.valid_loader = None
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 4

        pin_memory = pin_memory if torch.cuda.is_available() else False
        self.batch_size = batch_size
        cifar10_mean = (0.5, 0.5, 0.5)
        cifar10_std = (0.5, 0.5, 0.5)
        cifar100_mean = (0.5070, 0.4865, 0.4409)
        cifar100_std = (0.2673, 0.2564, 0.2761)
        mnist_mean = [33.318421449829934]
        mnist_std = [78.56749083061408]
        fashionmnist_mean = [73.14654541015625]
        fashionmnist_std = [89.8732681274414]

        if dataset == "CIFAR10":
            # CIFAR10:
            #   type               : uint8
            #   shape              : train_set.train_data.shape (50000, 32, 32, 3)
            #   test data shape    : (10000, 32, 32, 3)
            #   number of channels : 3
            #   Mean per channel   : train_set.train_data[:,:,:,0].mean() 125.306918046875
            #                        train_set.train_data[:,:,:,1].mean() 122.95039414062499
            #                        train_set.train_data[:,:,:,2].mean() 113.86538318359375
            #   Std per channel   :  train_set.train_data[:, :, :, 0].std() 62.993219278136884
            #                        train_set.train_data[:, :, :, 1].std() 62.088707640014213
            #                        train_set.train_data[:, :, :, 2].std() 66.704899640630913
            self.mean = cifar10_mean
            self.std = cifar10_std

            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])

            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])


            transform_train = get_aug(train=True, **args.aug_kwargs) # return two aug and one non-aug
            transform_test = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

            self.train_set = CIFAR10(root='./data/CIFAR10', train=True,
                                                          download=True, transform=transform_train)
            
            self.memory_set = CIFAR10(root='./data/CIFAR10', train=True,
                                                download=True, transform=transform_test)

            self.test_set = CIFAR10(root='./data/CIFAR10', train=False,
                                                         download=True, transform=transform_test)
            
            selected_classes = selected_classes
            self.num_of_tasks = len(selected_classes)
            print(self.num_of_tasks)
            train_task_datasets = [SubDataSet(self.train_set, selected_classes[0])]

            memory_loaders = [torch.utils.data.DataLoader(SubDataSet(self.memory_set, selected_classes[0]), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=pin_memory)]

            tasks_samples_indices = [torch.tensor(range(len(train_task_datasets[0])), dtype=torch.int32)]

            total_len = len(train_task_datasets[0])
            test_loaders = [torch.utils.data.DataLoader(SubDataSet(self.test_set, selected_classes[0]),
                                                        batch_size=self.batch_size, shuffle=False,
                                                        num_workers=self.num_workers, pin_memory=pin_memory)]
            # self.num_of_permutations = len(kwargs.get("all_permutation"))

            # all_permutation = kwargs.get("all_permutation", None)
            for i in range(1, self.num_of_tasks):
                # Add train set:
                train_task_datasets.append(SubDataSet(self.train_set, selected_classes[i]))
                tasks_samples_indices.append(torch.tensor(range(total_len,
                                                                total_len + len(train_task_datasets[-1])
                                                                ), dtype=torch.int32))
                total_len += len(train_task_datasets[-1])
                # Add test set:
                test_set = SubDataSet(self.test_set, selected_classes[i])
                memory_set = SubDataSet(self.memory_set, selected_classes[i])
                memory_loaders.append(torch.utils.data.DataLoader(memory_set, batch_size=self.batch_size,
                                                shuffle=False, num_workers=self.num_workers,
                                                pin_memory=pin_memory))
                test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                                shuffle=False, num_workers=self.num_workers,
                                                                pin_memory=pin_memory))
            self.test_loader = test_loaders
            self.memory_loader = memory_loaders
            # Concat datasets
            total_iters = kwargs.get("total_iters", None)

            assert total_iters is not None
            beta = 3
            train_all_datasets = torch.utils.data.ConcatDataset(train_task_datasets)

            # Create probabilities of tasks over iterations
            self.tasks_probs_over_iterations = [_create_task_probs(total_iters, self.num_of_tasks, task_id,
                                                                    beta=beta) for task_id in
                                                 range(self.num_of_tasks)]
            normalize_probs = torch.zeros_like(self.tasks_probs_over_iterations[0])
            for probs in self.tasks_probs_over_iterations:
                normalize_probs.add_(probs)
            for probs in self.tasks_probs_over_iterations:
                probs.div_(normalize_probs)
            self.tasks_probs_over_iterations = torch.cat(self.tasks_probs_over_iterations).view(-1, self.tasks_probs_over_iterations[0].shape[0])
            tasks_probs_over_iterations_lst = []
            for col in range(self.tasks_probs_over_iterations.shape[1]):
                tasks_probs_over_iterations_lst.append(self.tasks_probs_over_iterations[:, col])
            self.tasks_probs_over_iterations = tasks_probs_over_iterations_lst

            train_sampler = ContinuousMultinomialSampler(data_source=train_all_datasets, samples_in_batch=self.batch_size,
                                                         tasks_samples_indices=tasks_samples_indices,
                                                         tasks_probs_over_iterations=
                                                             self.tasks_probs_over_iterations,
                                                         num_of_batches=num_of_batches)
            
            self.train_loader = torch.utils.data.DataLoader(train_all_datasets, batch_size=self.batch_size,
                                                            num_workers=self.num_workers, sampler=train_sampler, pin_memory=pin_memory)

class ContinuousMultinomialSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    self.tasks_probs_over_iterations is the probabilities of tasks over iterations.
    self.samples_distribution_over_time is the actual distribution of samples over iterations
                                            (the result of sampling from self.tasks_probs_over_iterations).
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, samples_in_batch=128, num_of_batches=69, tasks_samples_indices=None,
                 tasks_probs_over_iterations=None):
        self.data_source = data_source
        assert tasks_samples_indices is not None, "Must provide tasks_samples_indices - a list of tensors," \
                                                  "each item in the list corrosponds to a task, each item of the " \
                                                  "tensor corrosponds to index of sample of this task"
        self.tasks_samples_indices = tasks_samples_indices
        self.num_of_tasks = len(self.tasks_samples_indices)
        assert tasks_probs_over_iterations is not None, "Must provide tasks_probs_over_iterations - a list of " \
                                                         "probs per iteration"
        assert all([isinstance(probs, torch.Tensor) and len(probs) == self.num_of_tasks for
                    probs in tasks_probs_over_iterations]), "All probs must be tensors of len" \
                                                              + str(self.num_of_tasks) + ", first tensor type is " \
                                                              + str(type(tasks_probs_over_iterations[0])) + ", and " \
                                                              " len is " + str(len(tasks_probs_over_iterations[0]))
        self.tasks_probs_over_iterations = tasks_probs_over_iterations
        self.current_iteration = 0

        self.samples_in_batch = samples_in_batch
        self.num_of_batches = num_of_batches

        # Create the samples_distribution_over_time
        self.samples_distribution_over_time = [[] for _ in range(self.num_of_tasks)]
        self.iter_indices_per_iteration = []

        if not isinstance(self.samples_in_batch, int) or self.samples_in_batch <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.samples_in_batch))

    def generate_iters_indices(self, num_of_iters):
        from_iter = len(self.iter_indices_per_iteration)
        for iter_num in range(from_iter, from_iter+num_of_iters):

            # Get random number of samples per task (according to iteration distribution)
            tsks = Categorical(probs=self.tasks_probs_over_iterations[iter_num]).sample(torch.Size([self.samples_in_batch]))
            # Generate samples indices for iter_num
            iter_indices = torch.zeros(0, dtype=torch.int32)
            for task_idx in range(self.num_of_tasks):
                if self.tasks_probs_over_iterations[iter_num][task_idx] > 0:
                    num_samples_from_task = (tsks == task_idx).sum().item()
                    self.samples_distribution_over_time[task_idx].append(num_samples_from_task)
                    # Randomize indices for each task (to allow creation of random task batch)
                    tasks_inner_permute = np.random.permutation(len(self.tasks_samples_indices[task_idx]))
                    rand_indices_of_task = tasks_inner_permute[:num_samples_from_task]
                    iter_indices = torch.cat([iter_indices, self.tasks_samples_indices[task_idx][rand_indices_of_task]])
                else:
                    self.samples_distribution_over_time[task_idx].append(0)
            self.iter_indices_per_iteration.append(iter_indices.tolist())

    def __iter__(self):
        self.generate_iters_indices(self.num_of_batches)
        self.current_iteration += self.num_of_batches
        return iter([item for sublist in self.iter_indices_per_iteration[self.current_iteration - self.num_of_batches:self.current_iteration] for item in sublist])

    def __len__(self):
        return self.samples_in_batch


def _get_linear_line(start, end, direction="up"):
    if direction == "up":
        return torch.FloatTensor([(i - start)/(end-start) for i in range(start, end)])
    return torch.FloatTensor([1 - ((i - start) / (end - start)) for i in range(start, end)])


def _create_task_probs(iters, tasks, task_id, beta=3):
    if beta <= 1:
        peak_start = int((task_id/tasks)*iters)
        peak_end = int(((task_id + 1) / tasks)*iters)
        start = peak_start
        end = peak_end
    else:
        start = max(int(((beta*task_id - 1)*iters)/(beta*tasks)), 0)
        peak_start = int(((beta*task_id + 1)*iters)/(beta*tasks))
        peak_end = int(((beta * task_id + (beta - 1)) * iters) / (beta * tasks))
        end = min(int(((beta * task_id + (beta + 1)) * iters) / (beta * tasks)), iters)

    probs = torch.zeros(iters, dtype=torch.float)
    if task_id == 0:
        probs[start:peak_start].add_(1)
    else:
        probs[start:peak_start] = _get_linear_line(start, peak_start, direction="up")
    probs[peak_start:peak_end].add_(1)
    if task_id == tasks - 1:
        probs[peak_end:end].add_(1)
    else:
        probs[peak_end:end] = _get_linear_line(peak_end, end, direction="down")
    return probs