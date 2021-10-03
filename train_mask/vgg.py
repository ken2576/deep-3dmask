import torch
import torch.nn as nn
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, model_path, requires_grad=False):
        super().__init__()
        
        if not model_path:
            vgg = models.vgg19(pretrained=True)
        else:
            vgg = models.vgg19()
            vgg.load_state_dict(torch.load(model_path))
        
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class BatchNormVgg19(torch.nn.Module):
    def __init__(self, model_path):
        super(BatchNormVgg19, self).__init__()
        if not model_path:
            vgg = models.vgg19_bn(pretrained=True)
        else:
            vgg = models.vgg19_bn()
            vgg.load_state_dict(torch.load(model_path))
        self.layers = vgg.features
        self.slice1 = nn.Sequential(*list(self.layers.children())[0:6]) 
        self.slice2 = nn.Sequential(*list(self.layers.children())[6:13]) 
        self.slice3 = nn.Sequential(*list(self.layers.children())[13:20]) 
        self.slice4 = nn.Sequential(*list(self.layers.children())[20:33]) 
        self.slice5 = nn.Sequential(*list(self.layers.children())[33:46]) 
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        out_1_2 = self.slice1(x)
        out_2_2 = self.slice2(out_1_2)
        out_3_2 = self.slice3(out_2_2)
        out_4_2 = self.slice4(out_3_2)
        out_5_2 = self.slice5(out_4_2)
        return [out_1_2, out_2_2, out_3_2, out_4_2, out_5_2]