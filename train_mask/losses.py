import torch
from torch import nn
import torch.nn.functional as F
from vgg import Vgg19, BatchNormVgg19

class VggBNLoss(nn.Module):
    def __init__(self, model_path=None):
        super(VggBNLoss, self).__init__()
        self.vgg = BatchNormVgg19(model_path)
        self.vgg.eval()

    def forward(self, inputs, targets):
        src = self.normalize_batch(inputs)
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
        loss = self.loss(inputs['rgb'], targets['tgt_rgb'])

        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb'], targets['tgt_rgb'])

        return loss

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='sum')
    
    def forward(self, inputs, targets):
        masked_rgb = inputs['rgb'] * targets['tgt_mask']
        masked_tgt = targets['tgt_rgb'] * targets['tgt_mask']
        loss = self.loss(masked_rgb, masked_tgt) / torch.sum(targets['tgt_mask'])
        return loss

class VGGVideoLoss(nn.Module):
    def __init__(self, model_path=None):
        super(VGGVideoLoss, self).__init__()
        self.loss = VggBNLoss(model_path)

    def forward(self, inputs, targets):
        loss = 0.0
        for rgb_seq, tgt_seq in zip(inputs['rgb'], targets['tgt_rgb']):
            loss += self.loss(rgb_seq, tgt_seq)
        return loss

class VGGTensorLoss(nn.Module):
    def __init__(self, model_path=None):
        super(VGGTensorLoss, self).__init__()
        self.loss = VggBNLoss(model_path)

    def forward(self, inputs, targets):
        loss = 0.0
        for rgb_seq, tgt_seq in zip(inputs, targets):
            loss += self.loss(rgb_seq, tgt_seq)
        return loss

class FgbgL1Loss(nn.Module):
    def __init__(self):
        super(FgbgL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        render_loss = self.loss(inputs['rgb'], targets['tgt_rgb'])
        fg_loss = self.loss(inputs['fg_rgb'], targets['tgt_rgb'])
        bg_loss = self.loss(inputs['bg_rgb'], targets['tgt_bg'])
        loss = render_loss + fg_loss + bg_loss
        return loss

class FgbgVGGLoss(nn.Module):
    def __init__(self, model_path=None):
        super(FgbgVGGLoss, self).__init__()
        self.video_loss = VGGTensorLoss(model_path)
        self.vgg_loss = VggBNLoss(model_path)

    def forward(self, inputs, targets):
        render_loss = self.video_loss(inputs['rgb'], targets['tgt_rgb'])
        # fg_loss = self.video_loss(inputs['fg_rgb'], targets['tgt_rgb'])
        # bg_loss = self.vgg_loss(inputs['bg_rgb'], targets['tgt_bg'])
        loss = render_loss# + bg_loss
        return loss

class MaskLoss(nn.Module):
    def __init__(self, ks=5, eps=1e-8):
        super(MaskLoss, self).__init__()

        self.ks = ks
        self.eps = eps

    def forward(self, inputs, targets):
        alpha = inputs # (batch, height, width)
        fg_mask = targets # (batch, height, width)

        kernel = torch.ones(1, 1, self.ks, self.ks, device=alpha.device)
        p = int((self.ks - 1) / 2)
        dilated = torch.clamp(F.conv2d(fg_mask[:, None], kernel, padding=(p, p)), 0, 1)

        bg_mask = (1 - dilated[:, 0])

        fg_region = fg_mask * (1-alpha)
        bg_region = bg_mask * alpha

        fg_loss = torch.sum(torch.abs(fg_region)) / 2 / (torch.sum(torch.abs(fg_mask))+self.eps)
        bg_loss = torch.sum(torch.abs(bg_region)) / 2 / (torch.sum(torch.abs(bg_mask))+self.eps)

        return fg_loss + bg_loss

class MaskVideoLoss(nn.Module):
    def __init__(self, ks=5):
        super(MaskVideoLoss, self).__init__()
        self.loss = MaskLoss(ks=ks)

    def forward(self, inputs, targets):
        loss = 0.0
        for alpha_seq, tgt_seq in zip(inputs['alpha'], targets['tgt_mask']):
            loss += self.loss(alpha_seq, tgt_seq)
        return loss

class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()

    def forward(self, inputs):
        alpha = inputs # (batch, height, width)

        return torch.sum(torch.abs(alpha)) / torch.numel(alpha)

class CombinedLoss(nn.Module):
    def __init__(self, model_path=None):
        super(CombinedLoss, self).__init__()

        self.vgg_loss = VggBNLoss(model_path)
        self.mask_loss = MaskLoss()
        self.sparse_loss = SparseLoss()

    def forward(self, inputs, targets):

        loss = self.vgg_loss(inputs['rgb'], targets['tgt_rgb']) + \
            0.25*self.mask_loss(inputs['alpha'], targets['tgt_mask']) + \
            0.10*self.sparse_loss(inputs['volume'])
        return loss

class NoMaskLoss(nn.Module):
    def __init__(self, model_path=None):
        super(NoMaskLoss, self).__init__()

        self.vgg_loss = VggBNLoss(model_path)
        self.sparse_loss = SparseLoss()

    def forward(self, inputs, targets):

        loss = self.vgg_loss(inputs['rgb'], targets['tgt_rgb']) + \
            0.10*self.sparse_loss(inputs['volume'])
        return loss

class VGGOnlyLoss(nn.Module):
    def __init__(self, model_path=None):
        super(VGGOnlyLoss, self).__init__()

        self.vgg_loss = VggBNLoss(model_path)

    def forward(self, inputs, targets):

        loss = self.vgg_loss(inputs['rgb'], targets['tgt_rgb'])
        return loss

loss_dict = {'mse': MSELoss, 'vgg': VggBNLoss, 'l1': L1Loss,
    'masked_l1': MaskedL1Loss, 'vgg_video': VGGVideoLoss,
    'fgbgl1': FgbgL1Loss, 'fgbgvgg': FgbgVGGLoss, 'full_loss': CombinedLoss, 'no_mask': NoMaskLoss, 'vgg_only': VGGOnlyLoss}