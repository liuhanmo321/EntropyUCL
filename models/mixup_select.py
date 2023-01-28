# from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch

from torchvision import transforms
from sklearn.cluster import KMeans, AgglomerativeClustering, kmeans_plusplus

from models.utils.continual_model import ContinualModel
from typing import Tuple
import numpy as np
import torch.nn as nn
from .optimizers import get_optimizer

import pandas as pd
from sklearn.metrics import pairwise_distances_argmin

from sklearn.decomposition import PCA

class Mixup_select(ContinualModel):
    NAME = 'mixup_select'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_lodaer, transform):
        super(Mixup_select, self).__init__(backbone, loss, args, len_train_lodaer, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.seen_tasks = 0

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
                    self.args.train.batch_size, transform=self.transform)
                lam = np.random.beta(self.args.train.alpha, self.args.train.alpha)
                mixed_x = lam * inputs1.to(self.device) + (1 - lam) * buf_inputs[:inputs1.shape[0]].to(self.device)
                mixed_x_aug = lam * inputs2.to(self.device) + (1 - lam) * buf_inputs1[:inputs1.shape[0]].to(self.device)
                data_dict = self.net.forward(mixed_x.to(self.device, non_blocking=True), mixed_x_aug.to(self.device, non_blocking=True))
                loss = data_dict['loss'].mean()
                data_dict['loss'] = data_dict['loss'].mean()
                data_dict['penalty'] = 0.0
            
        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        # if self.args.cl_default:
        #     self.buffer.add_data(examples=notaug_inputs, logits=labels)
        # else:
        #     self.buffer.add_data(examples=notaug_inputs, logits=inputs2)

        return data_dict


    def end_task(self, dataset):        
        self.seen_tasks += 1
        self.net.backbone.eval()
        feature_bank = []
        image_bank = []
        # aug_image_bank = []
        label_bank = []
        with torch.no_grad():
            # generate feature bank
            for idx, ((_, images2, notaug_images), labels) in enumerate(dataset.train_loaders[-1]):
            # for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
                if self.args.cl_default:
                    # feature = self.net.backbone(notaug_images.cuda(non_blocking=True), return_features=True)
                    feature = self.net.backbone(notaug_images.to(self.device), return_features=True)
                else:
                    feature = self.net.encoder(notaug_images.to(self.device))
                # feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                image_bank.append(notaug_images)
                # aug_image_bank.append(images2)
                label_bank += labels
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).contiguous()
            image_bank = torch.cat(image_bank, dim=0).contiguous()
            # aug_image_bank = torch.cat(aug_image_bank, dim=0).contiguous()
            label_bank = np.array(label_bank)

            feature_bank = F.normalize(feature_bank, dim=1)
            print(feature_bank.shape)
            # print(feature_bank[0, :].view(1, -1).shape)

        if self.args.train.cluster_type == 'k-means':
            k_means = KMeans(n_clusters=self.args.train.cluster_number).fit(feature_bank.cpu())
            centers = k_means.cluster_centers_
            indices = pairwise_distances_argmin(centers, feature_bank.cpu())

            cluster_data = []
            for i in range(self.args.train.cluster_number):
                similarities = F.cosine_similarity(torch.unsqueeze(feature_bank[indices[i]], dim=0), feature_bank, dim=-1)
                # real_indices = torch.max(torch.unsqueeze(similarities, 0), 0)[1][:20] # knn neighbors
                _, real_indices = torch.sort(similarities, descending=True, stable=True)
                # print(real_indices)
                cluster_data.append(image_bank[real_indices[:20]])
        
        elif self.args.train.cluster_type == 'uniform':
            centers, indices = kmeans_plusplus(feature_bank.cpu().numpy(), n_clusters=self.args.train.cluster_number)
            cluster_data = []
            for i in range(self.args.train.cluster_number):
                similarities = F.cosine_similarity(torch.unsqueeze(feature_bank[indices[i]], dim=0), feature_bank, dim=-1)
                # real_indices = torch.max(torch.unsqueeze(similarities, 0), 0)[1][:20] # knn neighbors
                _, real_indices = torch.sort(similarities, descending=True, stable=True)
                # print(real_indices)
                cluster_data.append(image_bank[real_indices[:20]])

        elif self.args.train.cluster_type == 'pca':
            # feature_bank = feature_bank.cpu().numpy()

            pca = PCA()
            pca.fit(feature_bank.cpu().numpy().T) # k * n
            ort_vec = pca.components_[:, :self.args.model.buffer_size].T
            print(ort_vec.shape)

            indices = pairwise_distances_argmin(ort_vec, feature_bank.cpu().numpy())
            std_list = None
            
            cluster_data = []
            for i in range(self.args.train.cluster_number):
                similarities = F.cosine_similarity(torch.unsqueeze(feature_bank[indices[i]], dim=0), feature_bank, dim=-1)
                # real_indices = torch.max(torch.unsqueeze(similarities, 0), 0)[1][:20] # knn neighbors
                _, real_indices = torch.sort(similarities, descending=True, stable=True)
                # print(real_indices)
                cluster_data.append(image_bank[real_indices[:20]])

        self.buffer.add_data(examples=cluster_data, seen=self.seen_tasks)
        saved_info = pd.Series(label_bank[indices]).value_counts()
        print(saved_info)

        print('number of examples: ', len(self.buffer.examples))
        print('buffer number: ', len(self.buffer.ds_buffer))
        print('selected num of data from current data: ', len(self.buffer.ds_buffer[-1]))


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
        self.cluster_number = 13
        self.seen_tasks = 0
        self.examples = None
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.ds_buffer = []

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

    def add_data(self, examples, labels=None, logits=None, task_labels=None, seen=0):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        # if not hasattr(self, 'examples'):
        #     self.init_tensors(examples, labels, logits, task_labels)
        
        self.ds_buffer.append(examples)
        final_examples = []
        sampling_range = int(np.ceil(256 / seen / self.cluster_number))
        for ds_id in range(seen):
            for cluster_id in range(self.cluster_number):
                final_examples.append(self.ds_buffer[ds_id][cluster_id][:sampling_range])
        
        self.examples = torch.cat(final_examples, 0).contiguous()

        self.num_seen_examples = self.examples.shape[0]
            # if ds_id + 1 == seen:
            #     for i in range((self.buffer_size - ds_id * sampling_range)):
            #         self.examples[i + ds_id * sampling_range] = self.ds_buffer[ds_id][i]
            #         # self.logits[i + ds_id * sampling_range] = self.ds_buffer[ds_id][1][i]
            # else:
            #     for i in range(sampling_range):
            #         self.examples[i + ds_id * sampling_range] = self.ds_buffer[ds_id][i]
            #         # self.logits[i + ds_id * sampling_range] = self.ds_buffer[ds_id][1][i]


    # def get_data(self, size: int, transform: transforms=None) -> Tuple:
    #     """
    #     Random samples a batch of size items.
    #     :param size: the number of requested items
    #     :param transform: the transformation to be applied (data augmentation)
    #     :return:
    #     """
    #     if size > self.examples.shape[0]:
    #         size = self.examples.shape[0]
    #     # print("size is: ", size)

    #     choice = np.random.choice(self.examples.shape[0], size=size, replace=False)
    #     if transform is None: transform = lambda x: x
    #     # import pdb
    #     # pdb.set_trace()
    #     ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
    #     # ret_tuple = (torch.stack([transform(ee.cpu())
    #     #                 for ee in self.examples]).to(self.device),
    #     #             torch.stack([transform(ee.cpu())
    #     #                 for ee in self.examples]).to(self.device))

    #     for attr_str in self.attributes[1:]:
    #         if hasattr(self, attr_str):
    #             attr = getattr(self, attr_str)
    #             ret_tuple += (attr[choice],)

    #     return ret_tuple

    # def add_data(self, examples, labels=None, logits=None, task_labels=None, seen=0):
    #     """
    #     Adds the data to the memory buffer according to the reservoir strategy.
    #     :param examples: tensor containing the images
    #     :param labels: tensor containing the labels
    #     :param logits: tensor containing the outputs of the network
    #     :param task_labels: tensor containing the task labels
    #     :return:
    #     """
    #     if not hasattr(self, 'examples'):
    #         self.init_tensors(examples, labels, logits, task_labels)
        
    #     self.ds_buffer.append(examples)
    #     self.examples = torch.cat(self.ds_buffer, dim=0)

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > self.examples.shape[0]:
            size = self.examples.shape[0]

        choice = np.random.choice(self.examples.shape[0], size=size, replace=False)

        if transform is None: transform = lambda x: x
        # import pdb
        # pdb.set_trace()
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),
                        torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device), )
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                 for ee in self.examples[choice]]).to(self.device),
        #             torch.stack([transform(ee.cpu())
        #                 for ee in self.examples[choice]]).to(self.device),)
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