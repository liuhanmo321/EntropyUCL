# from utils.buffer import Buffer
from typing import Tuple

import numpy as np
import torch
from augmentations import get_aug

import pandas as pd
import hdbscan
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch.nn import functional as F
from torchvision import transforms

from models.utils.continual_model import ContinualModel


class Ours_sep_replay(ContinualModel):
    NAME = 'ours_sep_replay'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    pca_decomps = []

    def __init__(self, backbone, loss, args, len_train_lodaer, transform):
        super(Ours_sep_replay, self).__init__(backbone, loss, args, len_train_lodaer, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        if self.buffer.is_empty():
            if self.args.cl_default:
                labels = labels.to(self.device)
                outputs = self.net.module.backbone(inputs1.to(self.device))
                loss = self.loss(outputs, labels).mean()
                data_dict = {'loss': loss}

            else:
                data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
                loss = data_dict['loss'].mean()
                data_dict['loss'] = data_dict['loss'].mean()

        else:
            if self.args.cl_default:
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.train.batch_size, transform=self.transform)
                buf_labels = buf_labels.to(self.device).long()
                labels = labels.to(self.device).long()
                lam = np.random.beta(self.args.train.alpha, self.args.train.alpha)
                mixed_x = lam * inputs1.to(self.device) + (1 - lam) * buf_inputs[:inputs1.shape[0]].to(self.device)
                net_output = self.net.module.backbone(mixed_x.to(self.device, non_blocking=True))
                buf_labels = buf_labels[:inputs1.shape[0]].to(self.device)
                loss = self.loss(net_output, labels) + (1 - lam) * self.loss(net_output, buf_labels)
                data_dict = {'loss': loss}
                data_dict['penalty'] = 0.0
            else:
                buf_inputs, buf_inputs1 = self.buffer.get_data(
                    self.args.train.replay_size, transform=self.transform)
                # lam = np.random.beta(self.args.train.alpha, self.args.train.alpha)
                # mixed_x = lam * inputs1.to(self.device) + (1 - lam) * buf_inputs[:inputs1.shape[0]].to(self.device)
                # mixed_x_aug = lam * inputs2.to(self.device) + (1 - lam) * buf_inputs1[:inputs1.shape[0]].to(self.device)
                data_dict = self.net.forward(inputs1.to(self.device), inputs2.to(self.device))
                old_data_dict = self.net.forward(buf_inputs.to(self.device), buf_inputs1.to(self.device))
                # data_dict = self.net.forward(mixed_x.to(self.device, non_blocking=True), mixed_x_aug.to(self.device, non_blocking=True))
                loss = data_dict['loss'].mean() + self.args.train.beta * old_data_dict['loss'].mean()
                data_dict['loss'] = loss
                data_dict['penalty'] = 0.0
            
        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        # if self.args.cl_default:
        #     self.buffer.add_data(examples=notaug_inputs, logits=labels)
        # else:
        #     self.buffer.add_data(examples=notaug_inputs, logits=inputs2)

        return data_dict

    def cluster(self, train_loader, device):
        self.net.backbone.eval()
        classes = len(train_loader.dataset.classes)
        # classes = 100
        total_top1 = total_top1_mask = total_top5 = total_num = 0.0
        feature_bank = []
        image_bank = []
        aug_image_bank = []
        with torch.no_grad():
            # generate feature bank
            for idx, ((_, images2, notaug_images), labels) in enumerate(train_loader):
            # for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
                if self.args.cl_default:
                    # feature = self.net.backbone(notaug_images.cuda(non_blocking=True), return_features=True)
                    feature = self.net.backbone(notaug_images.to(device), return_features=True)
                else:
                    feature = self.net.backbone(notaug_images.to(device))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                image_bank.append(notaug_images)
                aug_image_bank.append(images2)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).contiguous()
            image_bank = torch.cat(image_bank, dim=0).contiguous()
            aug_image_bank = torch.cat(aug_image_bank, dim=0).contiguous()
            print(feature_bank.shape)

        # feature_bank = feature_bank.cpu()
        # pca = PCA(n_components=32)
        # pca.fit(feature_bank)
        # data_reduced = pca.transform(feature_bank)
        # print(data_reduced.shape)

        feature_embedded = TSNE(n_components=2, learning_rate='auto', perplexity=40, init='pca').fit_transform(feature_bank.cpu().numpy())

        # data_sim = cosine_similarity(data_reduced, data_reduced)
        # dbsc = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2).fit(data_sim)
        dbsc = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=160).fit(feature_embedded)
        cluster_labels = dbsc.labels_
        cluster_probs = dbsc.probabilities_
        # print(set(cluster_labels))
        cluster_df = pd.DataFrame({'label': cluster_labels, 'probs': cluster_probs})
        print(cluster_df['label'].value_counts())
        # plt.show()
        # print(cluster_df['label'].unique())
        # print(train_loader.dataset[0])
        # print(train_loader.dataset.size)
        prototype_list = []
        aug_prototype_list = []
        for label in cluster_df['label'].unique():
            if label >= 0:
                relatedness = torch.Tensor(cluster_df[cluster_df['label'] == label]['probs'].to_numpy())
                relatedness_val, relatedness_ind = relatedness.topk(k=5, dim=0)
                # print(relatedness_val)
                # print(relatedness_ind)
                prototype_idx = list(cluster_df[cluster_df['label'] == label].iloc[relatedness_ind].index)
                prototype_list.append(image_bank[prototype_idx, :])
                aug_prototype_list.append(aug_image_bank[prototype_idx, :])
        # prototype_list
        
        # pca = PCA(n_components=32)
        # pca.fit(feature_bank.numpy())
        # self.pca_decomps.append(pca)
        # reduced_feature = pca.transform(feature_bank.numpy())
        # reduced_feature = torch.Tensor(reduced_feature)
        
        print(prototype_list[0].size)
        # print()
        for i in range(len(prototype_list)):
            self.buffer.add_data(examples=prototype_list[i], logits=aug_prototype_list[i])



# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.




def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)
        
        for i in range(examples.shape[0]):
            self.num_seen_examples += 1
            self.examples[self.num_seen_examples] = examples[i].to(self.device)
            self.logits[self.num_seen_examples] = logits[i].to(self.device)

        # for i in range(examples.shape[0]):
        #     index = reservoir(self.num_seen_examples, self.buffer_size)
        #     self.num_seen_examples += 1
        #     if index >= 0:
        #         self.examples[index] = examples[i].to(self.device)
        #         if labels is not None:
        #             self.labels[index] = labels[i].to(self.device)
        #         if logits is not None:
        #             self.logits[index] = logits[i].to(self.device)
        #         if task_labels is not None:
        #             self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        # import pdb
        # pdb.set_trace()
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
