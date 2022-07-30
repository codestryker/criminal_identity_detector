import torch
import torch as th
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


CUDA=False
cond_dim = 128
df_dim = 64
gf_dim = 64
z_dim = 50
emb_dim = 256

from cid_stage_1 import Conv_k3, Upblock,G_Stage1, weights_init, get_transform, get_data_loader, CondAugment_Model
from cid_stage_1 import KL_loss, cal_G_loss, cal_D_loss, scale, save_model, Text2FaceDataset

import numpy as np
import matplotlib.pyplot as plt


class ResBlock(nn.Module):
    def __init__(self, plane):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv_k3(plane, plane),
            nn.BatchNorm2d(plane),
            nn.ReLU(True),
            Conv_k3(plane, plane),
            nn.BatchNorm2d(plane)
        )
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        tmp = x
        o = self.block(x)
        o = o + tmp
        return self.relu(o)

class CondAugment_Model2(nn.Module):
    def __init__(self):
        super(CondAugment_Model2,self).__init__()
        self.fc = nn.Linear(in_features=emb_dim, out_features=cond_dim*2,bias=True)
        self.relu = nn.ReLU(True)
        
    def convert(self, embed):
        x = self.relu(self.fc(embed))
        mean, sigma = x[:, : cond_dim], x[:, cond_dim:]
        return mean, sigma
    
    def forward(self, x):
        mean, sigma = self.convert(x)
        diag = torch.exp(sigma*0.5)
        if CUDA:
            normal_dis = (torch.FloatTensor(diag.size()).normal_()).cuda()
        else:
            normal_dis = (torch.FloatTensor(diag.size()).normal_())
        condition = (diag*normal_dis)+mean
        return condition, mean, sigma

class G_Stage2(nn.Module):
    def __init__(self, G_Stage1):
        super(G_Stage2, self).__init__()
        self.G1 = G_Stage1
        self.CA = CondAugment_Model2()
        for p in self.G1.parameters():
            p.requires_grad = False
        self.encoder = nn.Sequential(
            Conv_k3(3, gf_dim),
            nn.ReLU(True),
            nn.Conv2d(gf_dim, gf_dim * 4, 8, 8, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 4),
            nn.ReLU(True))
        self.combine = nn.Sequential(
            Conv_k3(384, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.residual = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256)
        )
        self.decoder = nn.Sequential(
            Upblock(256,128),
            Upblock(128,64),
            Upblock(64,32),
            Upblock(32,16),
            Conv_k3(16,3),
            nn.Tanh()
        )
        
    def forward(self, noise, emb):
        init_image, _, _ = self.G1(noise, emb)
        encoded = self.encoder(init_image)
        
        cond, m, s = self.CA(emb)
        cond = cond.view(-1, 128, 1, 1)
        cond = cond.repeat(1, 1, 4, 4)
        
        encoded_cond = torch.cat([encoded, cond],1)
        img_feature = self.combine(encoded_cond)
        img_feature = self.residual(img_feature)
        img = self.decoder(img_feature)
        
        return init_image, img, m, s

class D_output(nn.Module):
    def __init__(self, have_cond = True):
        super(D_output, self).__init__()
        self.have_cond = have_cond
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2),
            nn.Sigmoid()
        )
        if have_cond:
            cond_part = nn.Sequential(
                Conv_k3(in_p=512+cond_dim, out_p=512),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.classifier = torch.nn.Sequential(*(list(cond_part)+list(self.classifier)))
        print(self.classifier)
            
    def forward(self, encoded_image, encoded_cond=None):
        if self.have_cond and encoded_cond is not None:
            cond = encoded_cond.view(-1,128, 1, 1)
            cond = cond.repeat(1, 1, 4, 4)
            image_with_cond = torch.cat([encoded_image, cond], 1)
        else:
            image_with_cond = encoded_image
        return self.classifier(image_with_cond).view(-1)

class D_Stage2(nn.Module):
    def __init__(self):
        super(D_Stage2, self).__init__()
        self.img_encoder = nn.Sequential(
            # start 3 x 64 x 64
            nn.Conv2d(3, df_dim, 4, 2, 1, bias=False), #=> 64 x 64 x 64
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(df_dim, df_dim*2, 4, 2, 1, bias=False), #=> 128 x 32 x 32
            nn.BatchNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(df_dim*2, df_dim*4, 4, 4, 1, bias=False), #=> 256 x 16 x 16
            nn.BatchNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(df_dim*4, df_dim*8, 2, 2, 1, bias=False), #=> 512 x 8 x 8
            nn.BatchNorm2d(df_dim*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(df_dim*8, df_dim*16, 2, 1, 1, bias=False), #=> 512 x 8 x 8
            nn.BatchNorm2d(df_dim*16),
            nn.LeakyReLU(0.2, True),

            Conv_k3(df_dim*16, df_dim*8), # 512 x 2 x 2
            nn.BatchNorm2d(df_dim*8),
            nn.LeakyReLU(0.2, True),

        )
        
        self.condition_classifier = D_output()
        self.uncondition_classifier = None # D_output(have_cond=False)
        
    def forward(self, img):
        img_feature = self.img_encoder(img)
        return img_feature
