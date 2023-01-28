from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch

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

class Ours_dis(ContinualModel):
    NAME = 'ours_dis'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Ours_dis, self).__init__(backbone, loss, args, len_train_loader, transform)
        # self.buffer = Buffer(self.args.model.buffer_size, self.device)
    
    def forward(self, x1, x2, model_old):
        f, h, f_old = self.net.encoder, self.net.predictor, model_old.net.encoder
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L_con = D(p1, z2) / 2 + D(p2, z1) / 2

        with torch.no_grad():
            z1_old, z2_old = f_old(x1), f_old(x2)
        # L_dis = D(p1, z1_old, T=self.args.train.T) / 2 + D(p2, z2_old, T=self.args.train.T) / 2
        L_dis = self.args.train.beta * (D(p1, z1_old) / 2 + D(p2, z2_old) / 2)
        # L_dis = D(p1, z2_old, T=self.args.train.T) / 2 + D(p2, z1_old, T=self.args.train.T) / 2
        return {'loss': L_con, 'penalty': L_dis}

    def observe(self, inputs1, labels, inputs2, notaug_inputs, model_old=None):

        self.opt.zero_grad()
        if self.args.cl_default:
            labels = labels.to(self.device)
            # outputs = self.net.module.backbone(inputs1.to(self.device))
            outputs = self.net.backbone(inputs1.to(self.device))
            loss = self.loss(outputs, labels).mean()
            data_dict = {'loss': loss, 'penalty': 0}
        else:
            if model_old != None:
                data_dict = self.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True), model_old)
                data_dict['loss'] = data_dict['loss'].mean()
                data_dict['penalty'] = data_dict['penalty'].mean()
            else:
                data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
                data_dict['loss'] = data_dict['loss'].mean()
                data_dict['penalty'] = 0
            loss = data_dict['loss'] + data_dict['penalty']



        # if not self.buffer.is_empty():
        #     buf_inputs, buf_logits = self.buffer.get_data(
        #         self.args.train.batch_size, transform=self.transform)
        #     # buf_outputs = self.net.module.backbone(buf_inputs)
        #     buf_outputs = self.net.backbone(buf_inputs)
        #     data_dict['penalty'] = self.args.train.alpha * F.mse_loss(buf_outputs, buf_logits)
        #     loss += data_dict['penalty']

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        # self.buffer.add_data(examples=notaug_inputs, logits=outputs.data)

        return data_dict
