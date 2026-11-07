import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
from model_utils import ConvBlock


class ConvNextFoveaNet(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        self.encoder=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None).features
        self.lateral=nn.ModuleList([nn.Conv2d(c,64,1) for c in [96,192,384,768]])
        self.smooth=nn.ModuleList([ConvBlock(64,64) for _ in range(3)])
        self.head=nn.Sequential(ConvBlock(64,64),nn.Conv2d(64,2,1))
        self.presence=nn.Sequential(nn.Linear(768,128),nn.SiLU(),nn.Linear(128,1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self,x):
        x=(x-self.mean)/self.std; features=[]
        for i,layer in enumerate(self.encoder):
            x=layer(x)
            if i in [1,3,5,7]: features.append(x)
        p=self.lateral[3](features[3])
        for i in [2,1,0]:
            p=self.smooth[i](F.interpolate(p,size=features[i].shape[-2:],mode='bilinear',align_corners=False)+self.lateral[i](features[i]))
        maps=self.head(p)
        return maps[:,0],maps[:,1],self.presence(features[3].mean((2,3))).squeeze(1)

    def freeze_batchnorm(self):
        pass
