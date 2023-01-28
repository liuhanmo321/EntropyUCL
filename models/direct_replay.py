# from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch

from torchvision import transforms
from sklearn.cluster import KMeans

from models.utils.continual_model import ContinualModel
from typing import Tuple
import numpy as np
import torch.nn as nn
from .optimizers import get_optimizer

import pandas as pd
from sklearn.metrics import pairwise_distances_argmin

def D(p, z, T=None, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        if T == None:
            return -(p*z).sum(dim=1).mean()
        else:
            return -torch.pow((p*z), T).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        if T == None:
            return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            # print('distillation called')
            return -torch.pow((1 + F.cosine_similarity(p, z.detach(), dim=-1)), T).mean()
        # return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class Direct_replay(ContinualModel):
    NAME = 'direct_replay'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Direct_replay, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.seen_tasks = 0

        distill_proj_hidden_dim = 2048
        output_dim = 2048
        self.distill_predictor = nn.Sequential(
            nn.Linear(output_dim, distill_proj_hidden_dim),
            nn.BatchNorm1d(distill_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(distill_proj_hidden_dim, output_dim),
        )

        self.dp_opt = get_optimizer(
                args.train.optimizer.name, self.distill_predictor, 
                lr=args.train.base_lr*args.train.batch_size/256, 
                momentum=args.train.optimizer.momentum,
                weight_decay=args.train.optimizer.weight_decay)

    def forward_old(self, x, model_old, x2=None):
        f, h, f_old, h_old = self.net.encoder, self.net.predictor, model_old.net.encoder, model_old.net.predictor
        if x2 == None:
            z = f(x)
            if self.args.train.distill_old:
                dp = self.distill_predictor(z)
                p = h(dp)
            else:
                p = h(z)
            with torch.no_grad():
                z_old = f_old(x)
                # p_old = h_old(z_old)
            # L_dis = D(p1, z1_old, T=self.args.train.T) / 2 + D(p2, z2_old, T=self.args.train.T) / 2
            L_dis = self.args.train.beta * (D(p, z_old)) / 2
            # L_dis = D(p1, z2_old, T=self.args.train.T) / 2 + D(p2, z1_old, T=self.args.train.T) / 2
        else:
            z1, z2 = f(x), f(x2)
            dp1, dp2 = self.distill_predictor(z1), self.distill_predictor(z2)
            p1, p2 = h(dp1), h(dp2)
            with torch.no_grad():
                z1_old, z2_old = f_old(x), f_old(x2)
            L_dis = self.args.train.beta * (D(p1, z2_old) / 2 + D(p2, z1_old) / 2)
        return L_dis.mean()

    def forward(self, x1, x2, model_old):
        f, h, f_old = self.net.encoder, self.net.predictor, model_old.net.encoder
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        dp1, dp2 = self.distill_predictor(z1), self.distill_predictor(z2)
        L_con = D(p1, z2) / 2 + D(p2, z1) / 2

        with torch.no_grad():
            z1_old, z2_old = f_old(x1), f_old(x2)
        # L_dis = D(p1, z1_old, T=self.args.train.T) / 2 + D(p2, z2_old, T=self.args.train.T) / 2
        L_dis = (D(h(dp1), z1_old) / 2 + D(h(dp2), z2_old) / 2)
        # L_dis = D(p1, z2_old, T=self.args.train.T) / 2 + D(p2, z1_old, T=self.args.train.T) / 2
        return {'loss': L_con, 'penalty': L_dis}

    def observe(self, inputs1, labels, inputs2, notaug_inputs, model_old=None):

        self.opt.zero_grad()
        if model_old != None:
            self.dp_opt.zero_grad()

        if self.args.cl_default:
            labels = labels.to(self.device)
            # outputs = self.net.module.backbone(inputs1.to(self.device))
            outputs = self.net.backbone(inputs1.to(self.device))
            loss = self.loss(outputs, labels).mean()
            data_dict = {'loss': loss, 'penalty': 0}
        else:
            data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
            data_dict['loss'] = data_dict['loss'].mean()
            data_dict['penalty'] = 0

            if self.seen_tasks > 0:
                buf_inputs = self.buffer.get_data(self.args.train.batch_size, transform=self.transform)                
                data_dict['penalty'] += self.net.forward(buf_inputs[0], buf_inputs[1])['loss'].mean()
            
            loss = data_dict['loss'] + data_dict['penalty']

        loss.backward()
        self.opt.step()
        if model_old != None:
            self.dp_opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        # self.buffer.add_data(examples=notaug_inputs)

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
            k_means = KMeans(n_clusters=self.args.model.buffer_size).fit(feature_bank.cpu())
            centers = k_means.cluster_centers_
            indices = pairwise_distances_argmin(centers, feature_bank.cpu())        
        elif self.args.train.cluster_type == 'random':
            indices = np.random.choice(feature_bank.shape[0], self.args.model.buffer_size, replace=False)
        self.buffer.add_data(examples=image_bank[indices, :], seen=self.seen_tasks)
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
    #     sampling_range = int(self.buffer_size / seen)
    #     for ds_id in range(seen):
    #         if ds_id + 1 == seen:
    #             for i in range((self.buffer_size - ds_id * sampling_range)):
    #                 self.examples[i + ds_id * sampling_range] = self.ds_buffer[ds_id][i]
    #                 # self.logits[i + ds_id * sampling_range] = self.ds_buffer[ds_id][1][i]
    #         else:
    #             for i in range(sampling_range):
    #                 self.examples[i + ds_id * sampling_range] = self.ds_buffer[ds_id][i]
    #                 # self.logits[i + ds_id * sampling_range] = self.ds_buffer[ds_id][1][i]


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

    def add_data(self, examples, labels=None, logits=None, task_labels=None, seen=0):
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
        
        self.ds_buffer.append(examples)
        self.examples = torch.cat(self.ds_buffer, dim=0)

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
        # ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        ret_tuple = (torch.stack([transform(ee.cpu())
                        for ee in self.examples[choice]]).to(self.device),
                    torch.stack([transform(ee.cpu())
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