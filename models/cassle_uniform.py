# from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch

from torchvision import transforms

from models.utils.continual_model import ContinualModel
from typing import Tuple
import numpy as np
import torch.nn as nn
from .optimizers import get_optimizer

import pandas as pd
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA


from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum
import scipy.sparse as sp
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader


def D(p, z, noise=None, T=None, version='simplified', type='simsiam'): # negative cosine similarity
    if type == 'simsiam':
        if version == 'original':
            z = z.detach() # stop gradient
            p = F.normalize(p, dim=1) # l2-normalize 
            z = F.normalize(z, dim=1) # l2-normalize 
            if T == None:
                return -(p*z).sum(dim=1).mean()
            else:
                return -torch.pow((p*z), T).sum(dim=1).mean()

        elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
            if noise == None:
                return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
            else:
                # print("noise")
                z = z.detach()
                # print(noise.reshape(-1, 1))
                noise = noise.reshape(-1, 1).expand(-1, 2048).to(p.device) * torch.randn(p.shape).to(p.device)
                # p_noise = F.normalize(p, dim=1)
                z_noise = F.normalize(z, dim=1) + noise
                return - F.cosine_similarity(p, z_noise, dim=-1).mean()
            # else:
            #     # print('distillation called')
            #     return -torch.pow((1 + F.cosine_similarity(p, z.detach(), dim=-1)), T).mean()
            # return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception
    else:
        p_norm = (p - p.mean(0)) / p.std(0) # NxD
        if noise == None:
            z_norm = (z - z.mean(0)) / z.std(0) # NxD
        else:
            # z = z.detach()
            z_mag = torch.norm(z, dim=1)
            noise = (noise.to(p.device) * z_mag).reshape(-1, 1).expand(-1, 2048) * torch.randn(p.shape).to(p.device)
            z_noise = z + noise
            # noise = noise.reshape(-1, 1).expand(-1, 2048).to(p.device) * torch.randn(p.shape).to(p.device)
            # z_noise = F.normalize(z, dim=1) + noise
            z_norm = (z_noise - z_noise.mean(0)) / z_noise.std(0)

        N = p_norm.size(0)
        D = p_norm.size(1)

        # cross-correlation matrix
        c = torch.mm(p_norm.T, z_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=p_norm.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= 5e-3
        loss = c_diff.sum()

        return loss

class Cassle_uniform(ContinualModel):
    NAME = 'cassle_uniform'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Cassle_uniform, self).__init__(backbone, loss, args, len_train_loader, transform)
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

        if args.train.parallel:
            self.distill_predictor = nn.DataParallel(self.distill_predictor)
        # self.distill_predictor.to(args.device)

        self.dp_opt = get_optimizer(
                args.train.optimizer.name, self.distill_predictor, 
                lr=args.train.base_lr*args.train.batch_size/256, 
                momentum=args.train.optimizer.momentum,
                weight_decay=args.train.optimizer.weight_decay)

    def forward_old(self, x, model_old, x2=None, noise=None):
        if not self.args.train.parallel:
            f, h, f_old = self.net.encoder, self.net.predictor, model_old.encoder
        else:
            f, h, f_old = self.net.module.encoder, self.net.module.predictor, model_old.encoder
        
        if x2 == None:
            z = self.net.forward(x, old_data=True)
            if self.args.train.distill_old:
                # print("distill_old")
                dp = self.distill_predictor(z)
                if self.args.model.name == 'simsiam':
                    p = h(dp)
                else:
                    p = dp
            else:
                p = h(z)
            with torch.no_grad():
                z_old = f_old(x)
                # p_old = h_old(z_old)
            # L_dis = D(p1, z1_old, T=self.args.train.T) / 2 + D(p2, z2_old, T=self.args.train.T) / 2
            L_dis = self.args.train.beta * (D(p, z_old, noise=noise, type=self.args.model.name)) / 2
            # L_dis = D(p1, z2_old, T=self.args.train.T) / 2 + D(p2, z1_old, T=self.args.train.T) / 2
        else:
            z1, z2 = f(x), f(x2)
            dp1, dp2 = self.distill_predictor(z1), self.distill_predictor(z2)
            p1, p2 = h(dp1), h(dp2)
            with torch.no_grad():
                z1_old, z2_old = f_old(x), f_old(x2)
            L_dis = self.args.train.beta * (D(p1, z2_old) / 2 + D(p2, z1_old) / 2)
        
        # L_relation = 0
        # if self.args.train.distill_relation:
        #     # print("distill relationship")
        #     dp_norm = F.normalize(dp, dim=1) # l2-normalize 
        #     z_old_norm = F.normalize(z_old, dim=1) # l2-normalize

        #     new_sim = torch.matmul(dp_norm, dp_norm.T) 
        #     old_sim = torch.matmul(z_old_norm, z_old_norm.T)

        #     logits_mask = torch.scatter(
        #         torch.ones_like(new_sim),
        #         1,
        #         torch.arange(new_sim.size(0)).view(-1, 1).to(self.args.device),
        #         0
        #     )

        #     new_logits_max, _ = torch.max(new_sim * logits_mask, dim=1, keepdim=True)
        #     new_sim = new_sim - new_logits_max.detach()
        #     row_size = new_sim.size(0)
        #     new_logits = torch.exp(new_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(new_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
 
        #     old_logits_max, _ = torch.max(old_sim * logits_mask, dim=1, keepdim=True)
        #     old_sim = old_sim - old_logits_max.detach()
        #     old_logits = torch.exp(old_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(old_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        #     L_relation = (-old_logits * torch.log(new_logits)).sum(1).mean()

        return L_dis.mean()

    def forward(self, x1, x2, model_old):
        print("torch.cuda.memory_allocated 1: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        if self.args.train.parallel:
            f, h, f_old = self.net.module.encoder, self.net.module.predictor, model_old.module.encoder
        else:
            f, h, f_old = self.net.encoder, self.net.predictor, model_old.encoder
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        dp1, dp2 = self.distill_predictor(z1), self.distill_predictor(z2)
        # print("torch.cuda.memory_allocated 2: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        L_con = D(p1, z2) / 2 + D(p2, z1) / 2
        # print("torch.cuda.memory_allocated 3: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

        with torch.no_grad():
            z1_old, z2_old = f_old(x1), f_old(x2)
        # print("torch.cuda.memory_allocated 4: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        # L_dis = D(p1, z1_old, T=self.args.train.T) / 2 + D(p2, z2_old, T=self.args.train.T) / 2
        L_dis = (D(h(dp1), z1_old) / 2 + D(h(dp2), z2_old) / 2)
        # print("torch.cuda.memory_allocated 5: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
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
            return_feat = model_old != None
            if return_feat:
                data_dict, z1, z2 = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True), return_feat=return_feat)
                dp1, dp2 = self.distill_predictor(z1), self.distill_predictor(z2)
                with torch.no_grad():
                    z1_old, z2_old = model_old.encoder(inputs1.to(self.device, non_blocking=True)), model_old.encoder(inputs2.to(self.device, non_blocking=True))
                if self.args.model.name == 'simsiam':
                    if self.args.train.parallel:    
                        h = self.net.module.predictor
                    else:
                        h = self.net.predictor
                    L_dis = (D(h(dp1), z1_old, type=self.args.model.name) / 2 + D(h(dp2), z2_old, type=self.args.model.name) / 2)
                else:
                    L_dis = (D((dp1), z1_old, type=self.args.model.name) / 2 + D((dp2), z2_old, type=self.args.model.name) / 2)
                data_dict['penalty'] = L_dis.mean()
            else:            
                data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
                data_dict['penalty'] = 0
            data_dict['loss'] = data_dict['loss'].mean()
            
            if model_old != None:
                buf_inputs = self.buffer.get_data(self.args.train.batch_size, transform=self.transform)
                if len(buf_inputs) > 1:
                    data_dict['penalty'] += self.forward_old(buf_inputs[0].to(self.device, non_blocking=True), model_old, noise=buf_inputs[1])
                else:
                    data_dict['penalty'] += self.forward_old(buf_inputs[0].to(self.device, non_blocking=True), model_old)

            loss = data_dict['loss'] + data_dict['penalty']

        loss.backward()
        self.opt.step()
        if model_old != None:
            self.dp_opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        # self.buffer.add_data(examples=notaug_inputs)
        # torch.cuda.empty_cache()

        return data_dict
    
    def end_task(self, dataset, model_old=None):
        self.seen_tasks += 1

        if not self.args.train.parallel:
            if self.args.train.encoder_feat:
                extractor = self.net.encoder
            else:
                extractor = self.net.backbone
        else:
            if self.args.train.encoder_feat:
                extractor = self.net.module.encoder
            else:
                extractor = self.net.module.backbone
        extractor.eval()
        
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
                    feature = extractor(notaug_images.to(self.device))
                
                feature_bank.append(feature.cpu())
                image_bank.append(notaug_images.cpu())
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
            clustering = KMeans(n_clusters=self.args.model.buffer_size).fit(feature_bank.cpu())
            centers = clustering.cluster_centers_
            cluster_labels = clustering.labels_
            indices = pairwise_distances_argmin(centers, feature_bank.cpu())
            
            if self.args.train.add_noise:
                print('add noise!!!!')
                std_list = []
                for i in range(self.args.model.buffer_size):
                    mask = cluster_labels == i
                    cluster_features = feature_bank[mask, :]
                    std = torch.std(cluster_features, dim=0)
                    std_list.append(torch.mean(std))
            else:
                std_list = None
        
        if self.args.train.cluster_type == 'k-means-var':
            temp_dataset = dataset.train_loaders[-1].dataset
            temp_loader = DataLoader(temp_dataset, batch_size=self.args.train.batch_size, shuffle=False, num_workers=5)
            
            feature_bank = []
            image_bank = []
            aug_feature_bank = [[],[],[],[],[]]
            label_bank = []
            with torch.no_grad():
                # generate feature bank
                for idx, ((_, _, notaug_images), labels) in enumerate(temp_loader):
                    feature = extractor(notaug_images.to(self.device))                    
                    feature_bank.append(feature.cpu())
                    image_bank.append(notaug_images.cpu())
                    # aug_image_bank.append(images2)
                    label_bank += labels
                # [D, N]
                feature_bank = torch.cat(feature_bank, dim=0).contiguous()
                image_bank = torch.cat(image_bank, dim=0).contiguous()
                label_bank = np.array(label_bank)

                feature_bank = F.normalize(feature_bank, dim=1)

                for i in range(5):
                    for _, ((images1, _, _), _) in enumerate(temp_loader):
                        feature = extractor(images1.to(self.device))                    
                        aug_feature_bank[i].append(feature.cpu())
                
                    aug_feature_bank[i] = torch.cat(aug_feature_bank[i], dim=0).contiguous()
                    aug_feature_bank[i] = F.normalize(aug_feature_bank[i], dim=1)
                
                aug_feature_bank = torch.stack(aug_feature_bank, dim=0).contiguous()
                print(aug_feature_bank.shape)
                aug_feature_bank = torch.permute(aug_feature_bank, (1, 0, 2))                        
            
            print(aug_feature_bank.shape)
            mean_vec = aug_feature_bank.mean(dim=1, keepdim=True)
            std = torch.pow(aug_feature_bank - mean_vec, 2).sum(dim=-1, keepdim=False).sum(-1, keepdim=False)
            
            clustering = KMeans(n_clusters=self.args.train.cluster_number).fit(feature_bank.cpu())            
            cluster_labels = clustering.labels_
            
            indices = []
            for i in range(self.args.train.cluster_number):
                mask = cluster_labels == i
                indices += list(std[mask].argsort(dim=-1, descending=False)[:6])
                # indices = pairwise_distances_argmin(centers, feature_bank.cpu())
            
            std_list = None

        elif self.args.train.cluster_type == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=self.args.model.buffer_size, linkage=self.args.train.linkage).fit(feature_bank.cpu())
            cluster_labels = clustering.labels_
            centers = []
            std_list = []
            for i in range(self.args.model.buffer_size):
                mask = cluster_labels == i
                cluster_features = feature_bank[mask, :]
                center = torch.mean(cluster_features.cpu(), dim=0)
                # center = np.mean(cluster_features, axis=0)
                centers.append(center)
                std = torch.std(cluster_features, dim=0)
                std_list.append(torch.mean(std))
            centers = torch.stack(centers).contiguous()
            indices = pairwise_distances_argmin(centers, feature_bank.cpu())

        elif self.args.train.cluster_type == 'uniform':
            if self.args.train.remove_outliers:
                print("removing outliers!")
                feature_bank = feature_bank.cpu().numpy()
                neigh = NearestNeighbors(n_neighbors=10)
                neigh.fit(feature_bank)
                neigh_dist = neigh.kneighbors(feature_bank)[0]

                # mean_neigh_dist = np.mean(neigh_dist, axis=1)
                # # print(mean_neigh_dist.shape)
                # core_indices = np.argsort(mean_neigh_dist)[:-100]
                # # print(core_indices.shape)

                max_neigh_dist = np.max(neigh_dist, axis=1)
                # print(max_neigh_dist)
                T2_dist = np.quantile(max_neigh_dist, 0.9)

                canopies = canopy(feature_bank, T2_dist + 0.01, T2_dist)

                core_indices = []
                for ca in canopies.keys():
                    if len(canopies[ca][1]) > 5:
                        core_indices += canopies[ca][1]
                
                core_indices = list(set(core_indices))
                centers, indices = kmeans_plusplus(feature_bank[core_indices], n_clusters=self.args.model.buffer_size)
            else:
                print('most distant!')
                centers, indices = kmeans_plusplus(feature_bank.cpu().numpy(), n_clusters=self.args.model.buffer_size, random_state=None)
            std_list = None 

        elif self.args.train.cluster_type == 'random':         
            indices = np.random.choice(feature_bank.shape[0], size=self.args.model.buffer_size, replace=False)
            std_list = None

        elif self.args.train.cluster_type == 'greedy': 
            feature_bank = feature_bank.cpu().numpy()

            pca = PCA()
            pca.fit(feature_bank)
            ort_vec = pca.components_

            sim_mat = np.matmul(feature_bank, ort_vec.T)

            threshold = self.args.train.threshold
            feat_num = sim_mat.shape[1]
            
            sim_mat = np.abs(sim_mat)
            sorted_mat = np.sort(sim_mat)
            # print(sorted_mat.mean(axis=0))
            mean_vec = sorted_mat.mean(axis=0)
            threshold = mean_vec[mean_vec > self.args.train.threshold][0]
            print(threshold)
            # threshold = sorted_mat.mean(axis=0)[-5]

            non_zero_mat = sim_mat >= threshold
            sim_mat[non_zero_mat] = 1
            sim_mat[np.logical_not(non_zero_mat)] = 0
            # sim_mat is now about 1 and 0

            s_selected = np.zeros((1, feature_bank.shape[1]))
            
            selected_indices = []
            budget = self.args.model.buffer_size

            coverage = 0
            for i in range(budget):
                coverage = s_selected.mean()
                print(coverage)
                marginal_coverage = 0
                indice = 0
                for j in range(feature_bank.shape[0]):
                    if j in selected_indices:
                        continue
                    temp_s = s_selected + sim_mat[j]
                    temp_s[temp_s > 0] = 1
                    temp_marginal_coverage = temp_s.mean() - coverage
                    if temp_marginal_coverage >= marginal_coverage:
                        indice = j
                        marginal_coverage = temp_marginal_coverage

                if marginal_coverage == 0:
                    print("features exhausted")
                    _, selected_indices = kmeans_plusplus(feature_bank, budget, random_state = None, given_set=selected_indices)
                    break
                
                selected_indices.append(indice)
                s_selected = s_selected + sim_mat[indice]
                s_selected[s_selected > 0] = 1
                # print(s_selected)
            
            std_list = None
            indices = selected_indices
        
        elif self.args.train.cluster_type == 'pca':
            temp_feature_bank = feature_bank.cpu().numpy()

            pca = PCA(n_components=self.args.model.buffer_size)
            # pca.fit(feature_bank.T) # k * n
            # ort_vec = pca.components_[:, :self.args.model.buffer_size].T
            ort_vec = pca.fit_transform(temp_feature_bank.T).T
            print(ort_vec.shape)

            indices = pairwise_distances_argmin(ort_vec, temp_feature_bank)            
            
            if self.args.train.add_noise:
                std_list = []              
                neigh = NearestNeighbors(n_neighbors=self.args.train.knn_n)
                neigh.fit(temp_feature_bank)
                _, knn_indices = neigh.kneighbors(temp_feature_bank[indices, :], n_neighbors=self.args.train.knn_n)
                print(knn_indices.shape)
                for i in range(knn_indices.shape[0]):
                    related_features = feature_bank[knn_indices[i], :]
                    std = torch.std(related_features, dim=0)
                    std_list.append(torch.mean(std))
                # print(std_list)
            else:
                std_list = None
            
        
        elif self.args.train.cluster_type == 'max_etp':
            # feature_bank = feature_bank.cpu().numpy()
            # feature_bank.to(self.args.device)
            indices = []            
            for i in range(self.args.model.buffer_size):
                print("current id: ", i)
                if i == 0:
                    indices.append(np.random.randint(feature_bank.shape[0]))
                else:
                    entropy = -np.inf
                    selected_indice = 0
                    for j in range(feature_bank.shape[0]):
                        if j in indices:
                            continue
                        else:
                            temp_list = indices + [j]
                            temp_entropy = feature_entropy(feature_bank[temp_list])
                            if temp_entropy >= entropy:
                                entropy = temp_entropy
                                selected_indice = j
                    indices.append(selected_indice)
                    print("entropy is: ", entropy)
            std_list = None                     
        # print(std_list)
             
        # if self.args.train.add_noise and model_old != None:
        #     # print(len(self.buffer.noise_buffer))
        #     fracs = []
        #     for i, stored_data in enumerate(self.buffer.ds_buffer):                
        #         with torch.no_grad():
                    
        #             if self.args.train.encoder_feat:
        #                 features = self.net.encoder(stored_data.to(self.device))
        #                 old_features = model_old.encoder(stored_data.to(self.device))
        #             else:
        #                 features = self.net.backbone(stored_data.to(self.device))
        #                 old_features = model_old.backbone(stored_data.to(self.device))
                    
        #             std = torch.mean(torch.std(features, dim=0))
        #             old_std = torch.mean(torch.std(old_features, dim=0))
        #             frac = std / old_std
        #             fracs.append(frac)                     
        #             self.buffer.noise_buffer[i] = self.buffer.noise_buffer[i] * frac.cpu()
        #     print(fracs) 
        
        print('noise stored: ', std_list != None)
        self.buffer.add_data(examples=image_bank[indices, :], logits=std_list, seen=self.seen_tasks)                

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
        self.noise_buffer = []

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
            self.init_tensors(examples, labels, logits=None, task_labels=None)
        
        self.ds_buffer.append(examples)
        self.examples = torch.cat(self.ds_buffer, dim=0)

        if logits != None:
            self.noise_buffer.append(torch.Tensor(logits))
            self.noises = torch.cat(self.noise_buffer, dim=0)        
            print(self.noises.shape)


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
        
        # if len(self.noise_buffer) == 0:
        #     ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        # else:
        #     ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device), self.noises[choice], )
        
        if len(self.noise_buffer) == 0:
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]),)
        else:
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]), self.noises[choice], )

        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                 for ee in self.examples[choice]]).to(self.device),
        #             torch.stack([transform(ee.cpu())
        #                 for ee in self.examples[choice]]).to(self.device),)
        # for attr_str in self.attributes[1:]:
        #     if hasattr(self, attr_str):
        #         attr = getattr(self, attr_str)
        #         ret_tuple += (attr[choice],)

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


def canopy(data, T1, T2):
    unselected_indices = list(range(len(data)))

    init_indice = unselected_indices.pop()

    canopy_bins = {init_indice: [[init_indice], [init_indice]]} # cluster center indice, T1 cluster member, T2 cluster member
    centers = [init_indice]
    T2_list = [init_indice]

    while len(unselected_indices) != 0:
        # print(len(unselected_indices))
        indice = np.random.choice(unselected_indices)
        delete = False
        create_new_center = False
        
        center_indice = centers[int(pairwise_distances_argmin([data[indice]], data[centers]))]
        cluster_indices, T2_indices = canopy_bins[center_indice]
        # for center_indice, cluster_indices, T2_indices in canopy_bins:
        dist = pairwise_distances([data[indice]], [data[center_indice]])[0][0]
        # print(dist)
        if dist <= T1:
            cluster_indices.append(indice)
            # T1_data_list.append(indice)
            if dist <= T2:
                delete = True
                T2_indices.append(indice)
                T2_list.append(indice)
            else:
                create_new_center = True
                delete = True
        else:
            create_new_center = True
            delete = True
        
        if delete:
            unselected_indices.remove(indice)
        if create_new_center:
            centers.append(indice)
            # canopy_bins += {indice: [[indice], [indice]]}
            canopy_bins[indice] = [[indice], [indice]]
            # print(canopy_bins)
    return canopy_bins


def kmeans_plusplus(X, n_clusters, random_state, n_local_trials=None, given_set=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.
    n_clusters : int
        The number of seeds to choose.
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    
    x_squared_norms = row_norms(X, squared=True)
    random_state = check_random_state(random_state)
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    if given_set != None:
        # # Pick first center randomly and track index of point
        # # center_id = random_state.randint(n_samples)
        # center_id = given_set[0]
        # indices = np.full(n_clusters, -1, dtype=int)
        # if sp.issparse(X):
        #     centers[0] = X[center_id].toarray()
        # else:
        #     centers[0] = X[center_id]
        # indices[0] = center_id

        # # Initialize list of closest distances and calculate current potential
        # closest_dist_sq = _euclidean_distances(
        #     centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
        # )
        # current_pot = closest_dist_sq.sum()

        # # Pick the remaining n_clusters-1 points
        # for c in range(1, n_clusters):
        #     # Choose center candidates by sampling with probability proportional
        #     # to the squared distance to the closest existing center
        #     rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        #     candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        #     # XXX: numerical imprecision can result in a candidate_id out of range
        #     np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        #     # Compute distances to center candidates
        #     distance_to_candidates = _euclidean_distances(
        #         X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        #     )

        #     # update closest distances squared and potential for each candidate
        #     np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        #     candidates_pot = distance_to_candidates.sum(axis=1)

        #     # Decide which candidate is the best
        #     if c >= len(given_set):
        #         best_candidate = np.argmin(candidates_pot)
        #     else:
        #         best_candidate = given_set[c]
        #     current_pot = candidates_pot[best_candidate]
        #     closest_dist_sq = distance_to_candidates[best_candidate]
        #     best_candidate = candidate_ids[best_candidate]

        #     # Permanently add best center candidate found in local tries
        #     if sp.issparse(X):
        #         centers[c] = X[best_candidate].toarray()
        #     else:
        #         centers[c] = X[best_candidate]
        #     indices[c] = best_candidate

        center_id = random_state.randint(n_samples)
        indices = np.full(n_clusters, -1, dtype=int)

        seen_len = len(given_set)

        indices[:seen_len] = given_set

        distance_to_candidates = _euclidean_distances(
            X[given_set], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        # np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]

        # Pick the remaining n_clusters-1 points
        for c in range(seen_len, n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            # Compute distances to center candidates
            distance_to_candidates = _euclidean_distances(
                X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
            )

            # update closest distances squared and potential for each candidate
            np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            if sp.issparse(X):
                centers[c] = X[best_candidate].toarray()
            else:
                centers[c] = X[best_candidate]
            indices[c] = best_candidate

    else:

        # Pick first center randomly and track index of point
        center_id = random_state.randint(n_samples)
        indices = np.full(n_clusters, -1, dtype=int)
        if sp.issparse(X):
            centers[0] = X[center_id].toarray()
        else:
            centers[0] = X[center_id]
        indices[0] = center_id

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = _euclidean_distances(
            centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
        )
        current_pot = closest_dist_sq.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            # Compute distances to center candidates
            distance_to_candidates = _euclidean_distances(
                X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
            )

            # update closest distances squared and potential for each candidate
            np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            if sp.issparse(X):
                centers[c] = X[best_candidate].toarray()
            else:
                centers[c] = X[best_candidate]
            indices[c] = best_candidate

    return centers, indices


def feature_entropy(features, order=4, eps=64):
    
    m, d = features.shape
    c = d / m / eps * torch.mm(features, features.T)
    # print(c.shape)
    power_matrix = c
    sum_matrix = torch.zeros_like(power_matrix)

    for k in range(1, order+1):
        if k > 1:
            power_matrix = torch.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else: 
            sum_matrix -= power_matrix / k

    trace = torch.trace(sum_matrix)

    return trace