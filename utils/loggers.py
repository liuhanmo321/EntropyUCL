# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

import torch.nn.functional as F 
import torch
from utils.metrics import mask_classes
import pandas as pd
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def plot_feat(net, memory_data_loaders, test_data_loaders, task_id, device, cl_default, axs):
    net.eval()
    feature_bank = []
    label_bank = []
    with torch.no_grad():
        # generate feature bank
        for i in range(len(memory_data_loaders)):
            temp_label_bank = []
            for idx, (images, labels) in enumerate(memory_data_loaders[i]):
                if cl_default:
                    feature = net(images.to(device), return_features=True)
                else:
                    feature = net(images.to(device))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                temp_label_bank += labels
            label_bank += temp_label_bank
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        label_bank = np.array(label_bank)
        feature_bank = feature_bank.cpu()

    feature_embedded = TSNE(n_components=2, learning_rate='auto', perplexity=40,
                init='pca').fit_transform(feature_bank)

    label_color = np.array([colors[i] for i in label_bank])
    
    tot_index = []
    for i in range(len(test_data_loaders)):
        prefix = int(feature_embedded.shape[0] / len(test_data_loaders) * i)
        plot_range = int(min(feature_embedded.shape[0] / len(test_data_loaders), 500))
        index = list(range(prefix, prefix + plot_range))
        tot_index += index
        axs[i + 1, len(test_data_loaders) - 1].scatter(feature_embedded[index, 0], feature_embedded[index, 1], c=label_color[index])

    axs[0, len(test_data_loaders) - 1].scatter(feature_embedded[tot_index, 0], feature_embedded[tot_index, 1], c=label_color[tot_index])
    # ax.scatter(feature_embedded[index, 0], feature_embedded[index, 1], c=label_bank[index])
    return axs

def plot_feat_sep(net, memory_data_loaders, test_data_loaders, task_id, device, cl_default, ax):
    net.eval()
    feature_bank = []
    label_bank = []
    with torch.no_grad():
        # generate feature bank
        for idx, (images, labels) in enumerate(memory_data_loaders[task_id]):
            if cl_default:
                feature = net(images.to(device), return_features=True)
            else:
                feature = net(images.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            label_bank += labels
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        label_bank = np.array(label_bank)
        feature_bank = feature_bank.cpu()

    feature_embedded = TSNE(n_components=2, learning_rate='auto', perplexity=40,
                init='pca').fit_transform(feature_bank)
    
    plot_range = int(min(feature_embedded.shape[0], 500))
    index = list(range(plot_range))
    ax.scatter(feature_embedded[index, 0], feature_embedded[index, 1], c=label_bank[index])
    # ax.scatter(feature_embedded[index, 0], feature_embedded[index, 1], c=label_bank[index])
    return ax

def plot_feat_tot(net, memory_data_loader, test_data_loaders, device, cl_default, ax):
    net.eval()
    feature_bank = []
    label_bank = []
    with torch.no_grad():
        # generate feature bank
        for i in range(len(test_data_loaders)):
            # temp_feature_bank = []
            # temp_label_bank = []
            for idx, (images, labels) in enumerate(test_data_loaders[i]):
            # for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
                if cl_default:
                    # feature = self.net.backbone(notaug_images.cuda(non_blocking=True), return_features=True)
                    feature = net(images.to(device), return_features=True)
                else:
                    feature = net(images.to(device))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                label_bank += labels

            # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        label_bank = np.array(label_bank)
        feature_bank = feature_bank.cpu()

            # feature_bank.append(temp_feature_bank)
            # label_bank.append(temp_label_bank)

    feature_embedded = TSNE(n_components=2, learning_rate='auto', perplexity=40,
                init='pca').fit_transform(feature_bank)
    
    index = []
    for i in range(len(test_data_loaders)):
        prefix = int(feature_embedded.shape[0] / len(test_data_loaders) * i)
        plot_range = int(min(feature_embedded.shape[0] / len(test_data_loaders), 200))
        index += list(range(prefix, prefix + plot_range))
        # index.append(temp_index)
    # index = np.concatenate(index, axis=0)
    ax.scatter(feature_embedded[index, 0], feature_embedded[index, 1], c=label_bank[index])
    return ax



def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in useless_args:
            if cc in args:
                del args[cc]

        columns = list(args.keys())

        new_cols = []
        for i, acc in enumerate(self.accs):
            args['task' + str(i + 1)] = acc
            new_cols.append('task' + str(i + 1))

        args['forward_transfer'] = self.fwt
        new_cols.append('forward_transfer')

        args['backward_transfer'] = self.bwt
        new_cols.append('backward_transfer')

        args['forgetting'] = self.forgetting
        new_cols.append('forgetting')

        columns = new_cols + columns

        create_if_not_exists(base_path() + "results/" + self.setting)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset + "/" + self.model)

        write_headers = False
        path = base_path() + "results/" + self.setting + "/" + self.dataset\
               + "/" + self.model + "/mean_accs.csv"
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)

        if self.setting == 'class-il':
            create_if_not_exists(base_path() + "results/task-il/"
                                 + self.dataset)
            create_if_not_exists(base_path() + "results/task-il/"
                                 + self.dataset + "/" + self.model)

            for i, acc in enumerate(self.accs_mask_classes):
                args['task' + str(i + 1)] = acc

            args['forward_transfer'] = self.fwt_mask_classes
            args['backward_transfer'] = self.bwt_mask_classes
            args['forgetting'] = self.forgetting_mask_classes

            write_headers = False
            path = base_path() + "results/task-il" + "/" + self.dataset + "/"\
                   + self.model + "/mean_accs.csv"
            if not os.path.exists(path):
                write_headers = True
            with open(path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)
