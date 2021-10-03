import torch
from torch import nn
from vgg import Vgg19, BatchNormVgg19

class VggBNLoss(nn.Module):
    def __init__(self, model_path=None):
        super(VggBNLoss, self).__init__()
        self.vgg = BatchNormVgg19(model_path)
        self.vgg.eval()

    def forward(self, inputs, targets):
        src = self.normalize_batch(inputs['rgb'])
        tgt = self.normalize_batch(targets)
        src_output = self.vgg(src)
        tgt_output = self.vgg(tgt)
        p0 = self.compute_error(src, tgt)
        p1 = self.compute_error(src_output[0], tgt_output[0]) / 2.6
        p2 = self.compute_error(src_output[1], tgt_output[1]) / 4.8
        p3 = self.compute_error(src_output[2], tgt_output[2]) / 3.7
        p4 = self.compute_error(src_output[3], tgt_output[3]) / 5.6
        p5 = self.compute_error(src_output[4], tgt_output[4]) * 10 / 1.5
        total_loss = p0 + p1 + p2 + p3 + p4 + p5
        return total_loss

    def compute_error(self, fake, real):
        return torch.mean(torch.abs((fake - real)))

    @staticmethod
    def normalize_batch(batch):
        """Normalize batch with VGG mean and std
        """
        mean = torch.zeros_like(batch)
        std = torch.zeros_like(batch)

        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225

        ret = batch - mean
        ret = ret / std
        return ret

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb'], targets)

        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb'], targets)

        return loss


loss_dict = {'mse': MSELoss, 'vgg': VggBNLoss, 'l1': L1Loss}