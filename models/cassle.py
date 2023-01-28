from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch
from .optimizers import get_optimizer
import torch.nn as nn

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
            z = z.detach()
            noise = noise.reshape(-1, 1).expand(-1, 2048).to(p.device) * torch.randn(p.shape).to(p.device)
            z_noise = F.normalize(z, dim=1) + noise
            z_norm = (z_noise - z_noise.mean(0)) / z_noise.std(0)

        N = p_norm.size(0)
        D = p_norm.size(1)

        # cross-correlation matrix
        c = torch.mm(p_norm.T, z_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=p_norm.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= 5e-3
        loss = 0.1 * c_diff.sum()

        return loss

class Cassle(ContinualModel):
    NAME = 'cassle'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Cassle, self).__init__(backbone, loss, args, len_train_loader, transform)
        # self.buffer = Buffer(self.args.model.buffer_size, self.device)
        # self.net = backbone
        # # self.net = nn.DataParallel(self.net)
        # self.loss = loss
        # self.args = args
        # self.transform = transform
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
        # self.opt = get_optimizer(
        #     args.train.optimizer.name, self.net, 
        #     lr=args.train.base_lr*args.train.batch_size/256, 
        #     momentum=args.train.optimizer.momentum,
        #     weight_decay=args.train.optimizer.weight_decay)

    def forward(self, x1, x2, model_old):
        f, h, f_old = self.net.encoder, self.net.predictor, model_old.encoder
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
                    L_dis = (D(dp1, z1_old, type=self.args.model.name) / 2 + D(dp2, z2_old, type=self.args.model.name) / 2)
                data_dict['penalty'] = L_dis.mean()
            else:            
                data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
                data_dict['penalty'] = 0
            data_dict['loss'] = data_dict['loss'].mean()

        loss = data_dict['loss'] + data_dict['penalty']
        loss.backward()
        self.opt.step()
        if model_old != None:
            self.dp_opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        # self.buffer.add_data(examples=notaug_inputs, logits=outputs.data)

        return data_dict
