import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable

TEXT_DIMENSION = 768
GAN_CONDITION_DIM = 128
CUDA=False
GAN_GF_DIM = 128
GAN_DF_DIM = 64
Z_DIM=100
GAN_R_NUM = 4

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = TEXT_DIMENSION
        self.c_dim = GAN_CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# ############# Networks for stageI GAN #############
class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = GAN_GF_DIM * 8
        self.ef_dim = GAN_CONDITION_DIM
        self.z_dim = Z_DIM
        self.define_module()

    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar


class STAGE1_D(nn.Module):
    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = GAN_DF_DIM
        self.ef_dim = GAN_CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G):
        super(STAGE2_G, self).__init__()
        self.gf_dim = GAN_GF_DIM
        self.ef_dim = GAN_CONDITION_DIM
        self.z_dim = Z_DIM
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(GAN_R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()
        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> ngf // 4 x 256 x 256
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 256 x 256
        self.img = nn.Sequential(
            conv3x3(ngf // 4, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        _, stage1_img, _, _ = self.STAGE1_G(text_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar


class STAGE2_D(nn.Module):
    def __init__(self):
        super(STAGE2_D, self).__init__()
        self.df_dim = GAN_DF_DIM
        self.ef_dim = GAN_CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            conv3x3(ndf * 32, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding
        
#!/usr/bin/env python
# coding: utf-8


import torch 
import torch.nn as nn


############## Configurations


dim_text_embedding = 768
dim_conditioning_var = 128
dim_noise = 100
channels_gen = 128
channels_discr = 64
upscale_factor = 2


# upsacles image by factor of 2 and also changes number of channels in upscaled image

def upscale(in_channels,out_channels):
    return nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='nearest'),
            nn.Conv2d(in_channels,out_channels,3,1,1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))



# convolutional residual block, keeps number of channels constant

class ResBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.block = nn.Sequential(
                        nn.Conv2d(channels,channels,3,1,1,bias = False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(True),
                        nn.Conv2d(channels,channels,3,1,1,bias = False),
                        nn.BatchNorm2d(channels)
                        )
        self.ReLU = nn.ReLU(True)
        
    def forward(self,x):
        residue = x
        x = self.block(x)
        x = x + residue
        x = self.ReLU(x)
        return x



class Conditional_augmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_fc_inp = 768
        self.dim_fc_out = dim_conditioning_var
        self.fc = nn.Linear(self.dim_fc_inp, self.dim_fc_out*2, bias= True)
        self.relu = nn.ReLU()
            
    def get_mu_logvar(self,textEmbedding):
        x = self.relu(self.fc(textEmbedding))
        
        mu = x[:,:dim_conditioning_var]
        logvar = x[:,dim_conditioning_var:]
        return mu,logvar
        
    
    def get_conditioning_variable(self,mu,logvar):
        epsilon = torch.randn(mu.size())
        std = torch.exp(0.5*logvar)
        
        return mu + epsilon*std
    
    def forward(self,textEmbedding):
        mu, logvar = self.get_mu_logvar(textEmbedding)
        return self.get_conditioning_variable(mu, logvar)


class Discriminator_logit(nn.Module):
    def __init__(self,dim_discr,dim_condVar,concat=False):
        super().__init__()
        self.dim_discr = dim_discr
        self.dim_condVar = dim_condVar
        self.concat = concat
        if concat == True:
            self.logits = nn.Sequential(
                            nn.Conv2d(dim_discr*8 + dim_condVar,dim_discr*8,3,1,1, bias = False),
                            nn.BatchNorm2d(dim_discr*8),
                            nn.LeakyReLU(.2, True),
                            nn.Conv2d(dim_discr*8, 1, kernel_size=4, stride=4),
                            nn.Sigmoid()
                        )
        
        else :
            self.logits = nn.Sequential(
                            nn.Conv2d(dim_discr*8, 1, kernel_size=4, stride=4),
                            nn.Sigmoid()
                        )
        
    def forward(self, hidden_vec, cond_aug=None):
        if self.concat is True and cond_aug is not None:
            cond_aug = cond_aug.view(-1, self.dim_condVar, 1, 1)
            cond_aug = cond_aug.repeat(1, 1, 4, 4)
            hidden_vec = torch.cat((hidden_vec,cond_aug),1)
        
        return self.logits(hidden_vec).view(-1)


class Stage1_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_noise = dim_noise
        self.dim_cond_aug = dim_conditioning_var
        self.channels_fc = channels_gen * 8
        self.cond_aug_net = Conditional_augmentation()
        
        self.fc = nn.Sequential(
                    nn.Linear(self.dim_noise + self.dim_cond_aug, self.channels_fc * 4 * 4, bias = False),
                    nn.BatchNorm1d(self.channels_fc * 4 * 4),
                    nn.ReLU(True)
                    )
        
        self.upsample = nn.Sequential(
                            upscale(self.channels_fc,self.channels_fc//2),
                            upscale(self.channels_fc//2,self.channels_fc//4),
                            upscale(self.channels_fc//4,self.channels_fc//8),
                            upscale(self.channels_fc//8,self.channels_fc//16)
                            )
        
        self.generated_image = nn.Sequential(
                                nn.Conv2d(self.channels_fc//16,3,3,1,1,bias = False),
                                nn.Tanh())
        
        
    def forward(self,noise,text_embedding):
        cond_aug = self.cond_aug_net(text_embedding)
        x = torch.cat((noise,cond_aug),1)
        
        x = self.fc(x)
        x = x.view(-1,self.channels_fc, 4, 4)
        x = self.upsample(x)
        
        image = self.generated_image(x)
        
        return image
        


class Stage1_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_initial = channels_discr
        
        self.downsample = nn.Sequential(
                            nn.Conv2d(3, self.channels_initial, kernel_size=4, stride=2, padding=1),
                            nn.LeakyReLU(0.2,inplace=True),
            
                            nn.Conv2d(self.channels_initial , self.channels_initial*2, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.channels_initial*2),
                            nn.LeakyReLU(0.2,inplace=True),
            
                            nn.Conv2d(self.channels_initial*2, self.channels_initial*4, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.channels_initial*4),
                            nn.LeakyReLU(0.2,inplace=True),
            
                            nn.Conv2d(self.channels_initial*4, self.channels_initial*8, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.channels_initial*8),
                            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.cond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,True)
        self.uncond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,False)
        
    def forward(self,img):
        return self.downsample(img)


class Stage2_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample_channels = channels_gen
        self.dim_embedding = dim_conditioning_var
        self.cond_aug_net = Conditional_augmentation()
        self.Stage1_G = Stage1_Generator()
        self.downsample = nn.Sequential(
                            nn.Conv2d(3, self.downsample_channels, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
            
                            nn.Conv2d(self.downsample_channels, self.downsample_channels*2, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.downsample_channels*2),
                            nn.ReLU(inplace=True),
            
                            nn.Conv2d(self.downsample_channels*2, self.downsample_channels*4, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.downsample_channels*4),
                            nn.ReLU(inplace=True),
                        )
        self.hidden = nn.Sequential(
                        nn.Conv2d(self.downsample_channels*4 + self.dim_embedding, self.downsample_channels*4, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(self.downsample_channels*4),
                        nn.ReLU(True)
                        )
        self.residual = nn.Sequential(
                            ResBlock(self.downsample_channels*4),
                            ResBlock(self.downsample_channels*4),
                            ResBlock(self.downsample_channels*4),
                            ResBlock(self.downsample_channels*4)            
                        )
        self.upsample = nn.Sequential(
                            upscale(self.downsample_channels*4,self.downsample_channels*2),
                            upscale(self.downsample_channels*2,self.downsample_channels),
                            upscale(self.downsample_channels,self.downsample_channels//2),
                            upscale(self.downsample_channels//2,self.downsample_channels//4)
                        )
        self.image = nn.Sequential(
                        nn.Conv2d(self.downsample_channels//4, 3, 3, 1, 1, bias = False),
                        nn.Tanh()
                        )
        
    def forward(self,noise, text_embedding):
        image = self.Stage1_G(noise, text_embedding)
        image = image.detach()
        enc_img = self.downsample(image)
        
        cond_aug = self.cond_aug_net(text_embedding)
        cond_aug = cond_aug.view(-1, self.dim_embedding, 1, 1)
        cond_aug = cond_aug.repeat(1, 1, 16, 16)
        
        x = torch.cat((enc_img, cond_aug),1)
        x = self.hidden(x)
        x = self.residual(x)
        x = self.upsample(x) 
        enlarged_img = self.image(x)
        
        return enlarged_img


class Stage2_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_initial = channels_discr
        self.downsample = nn.Sequential(
                            nn.Conv2d(3, self.channels_initial, 4, 2, 1, bias = False),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial, self.channels_initial*2, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*2),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*2, self.channels_initial*4, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*4),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*4, self.channels_initial*8, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*8),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*8, self.channels_initial*16, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*16),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*16, self.channels_initial*32, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*32),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*32, self.channels_initial*16, 3, 1, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*16),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*16, self.channels_initial*8, 3, 1, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*8),
                            nn.LeakyReLU(0.2, inplace = True)
                            )
        
        self.cond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,True)
        self.uncond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,False)
        
    def forward(self,image):
        return self.downsample(image)
