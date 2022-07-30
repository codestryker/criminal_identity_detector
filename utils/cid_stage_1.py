import torch
import torch as th
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import pickle as pkl

import pickle

def save_pickle(obj, file_name):
    """
    save the given data obj as a pickle file
    :param obj: python data object
    :param file_name: path of the output file
    :return: None (writes file to disk)
    """
    with open(file_name, 'wb') as dumper:
        pickle.dump(obj, dumper, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    """
    load a pickle object from the given pickle file
    :param file_name: path to the pickle file
    :return: obj => read pickle object
    """
    with open(file_name, "rb") as pick:
        obj = pickle.load(pick)

    return obj

from torch.utils.data import Dataset
import os
import PIL

class Text2FaceDataset(Dataset):
    """ PyTorch Dataset wrapper around the Face2Text dataset """

    def __load_data(self,path):
        """
        private helper for loading the data
        :return: data => dict of data objs
        """
        data = load_pickle(path)

        return data

    def __init__(self, desc_path, img_path, img_transform=None):
        """
        constructor of the class
        :param desc_path: path to image description directory
        :param img_path: path to the images directory
        :param img_transform: torch_vision transform to apply
        """

        # create state:
        self.transform = img_transform
        
        # create data object
        self.img_path=img_path
        self.data_obj = self.__load_data(desc_path)

        # extract all the data
        self.encoded_data = self.data_obj['captions']
        self.images = self.data_obj['images']


    def __len__(self):
        """
        obtain the length of the data-items
        :return: len => length
        """
        return len(self.images)

    def __getitem__(self, ix):
        """
        code to obtain a specific item at the given index
        :param ix: index for element query
        :return: (caption, img) => caption and the image
        """

        # read the image at the given index
        img_file_path = os.path.join(self.img_path, self.images[ix])
        img = PIL.Image.open(img_file_path)

        # transform the image if required
        if self.transform is not None:
            img = self.transform(img)

        # get the encoded caption:
        caption = self.encoded_data[ix]

        # return the data element
        return caption, img

def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize

    if new_size is not None:
        image_transform = Compose([
            Resize(new_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    else:
        image_transform = transforms.Compose([
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform


def get_data_loader(dataset, train_sampler, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader
    if train_sampler:
      dl = DataLoader(
          dataset,
          batch_size=batch_size,
          sampler=train_sampler,
          num_workers=num_workers,
          drop_last=True
      )
    else:
      dl = DataLoader(
          dataset,
          shuffle=True,
          batch_size=batch_size,
          num_workers=num_workers,
      )
    return dl

import torch
from torchvision import transforms
from torchvision.utils import make_grid

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


CUDA = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def Conv_k3(in_p, out_p, stride=1):
    return nn.Conv2d(in_p, out_p, kernel_size=3, stride=stride, padding=1, bias=False)

class Upblock(nn.Module):
    def __init__(self, inp, outp):
        super(Upblock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv_k3(inp, outp)
        self.batch = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(True)
        self.drop_out = nn.Dropout(0.2)
        
    def forward(self, x):
        o = self.up(x)
        o = self.relu(self.conv(o))
        o = self.batch(o)
        return o

class D_output(nn.Module):
    def __init__(self, have_cond = True):
        super(D_output, self).__init__()
        self.have_cond = have_cond
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        if have_cond:
            cond_part = nn.Sequential(
                Conv_k3(in_p=256+64, out_p=512),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.classifier = torch.nn.Sequential(*(list(cond_part)+list(self.classifier)))
        print(self.classifier)
            
    def forward(self, encoded_image, encoded_cond=None):
        if self.have_cond and encoded_cond is not None:
            cond = encoded_cond.view(-1, 64, 1, 1)
            cond = cond.repeat(1, 1, 2, 2)
            image_with_cond = torch.cat((encoded_image, 64), 1)
        else:
            image_with_cond = encoded_image
        return self.classifier(image_with_cond).view(-1)

class CondAugment_Model(nn.Module):
    def __init__(self):
        super(CondAugment_Model,self).__init__()
        self.fc = nn.Linear(in_features=256, out_features=64*2,bias=True)
        self.relu = nn.ReLU(True)
        
    def convert(self, embed):
        x = self.relu(self.fc(embed))
        mean, sigma = x[:, : 64], x[:, 64:]
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

class G_Stage1(nn.Module):
    def __init__(self):
        super(G_Stage1, self).__init__()
        self.CA = CondAugment_Model()
        self.fc = nn.Sequential(
            nn.Linear(in_features=114, out_features=32*8*2*2, bias=False),
            nn.BatchNorm1d(32*8*2*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.img = nn.Sequential(
            Upblock(32*8,32*4),
            Upblock(32*4,32*2),
            Upblock(32*2,32),
            Upblock(32,16),
            Conv_k3(16, 3),
            nn.Tanh()
        )
        
    def forward(self, noise, emb):
        cond, mean, sigma = self.CA(emb)
        cond = cond.view(noise.size(0), 64, 1, 1)
        x = torch.cat((noise, cond),1)
        x = x.view(-1, 114) # z_dim + cond_dim
        o = self.fc(x)
        h_code = o.view(-1, 32*8, 2, 2)
        fake_img = self.img(h_code)
        return fake_img, mean, sigma

class D_Stage1(nn.Module):
    def __init__(self):
        super(D_Stage1, self).__init__()
        self.encoder = nn.Sequential(
            #c alucalation output size = [(input_size −Kernal +2Padding )/Stride ]+1
            # input is image 3 x 32 x 32  
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),# => 32 x 16 x 16
            
            nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, inplace=True),# => 64 x 8 x 8
            
            nn.Conv2d(in_channels=32*2, out_channels=32*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, inplace=True),# => 128 x 4 x 4
            
            nn.Conv2d(in_channels=32*4, out_channels=32*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32*8),
            nn.LeakyReLU(0.2, inplace=True),# => 512 x 2 x 2
        )
        self.condition_classifier = D_output()
        self.uncondition_classifier = None
        
    def forward(self, image):
        return self.encoder(image)

def KL_loss(mean, sigma):
    KLD_element = mean.pow(2).add_(sigma.exp()).mul_(-1).add_(1).add_(sigma)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def cal_G_loss(netD, fake_imgs, real_labels, cond):
    criterion = nn.BCELoss()
    cond = cond.detach()
    fake_f = netD(fake_imgs)

    fake_cond_ouput = netD.condition_classifier(fake_f, cond)
    errD_fake = criterion(fake_cond_ouput, real_labels)
    if netD.uncondition_classifier is not None:
        fake_uncond_output = netD.uncondition_classifier(fake_f)
        uncond_errD_fake = criterion(fake_uncond_output, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake

def cal_D_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, cond):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = cond.detach()
    fake = fake_imgs.detach()

    real_img_feature = netD(real_imgs)
    fake_img_feature = netD(fake)

    real_output = netD.condition_classifier(real_img_feature, cond)
    errD_real  = criterion(real_output, real_labels)
    wrong_output = netD.condition_classifier(real_img_feature[:(batch_size-1)], cond[1:])
    errD_wrong = criterion(wrong_output, fake_labels[1:])

    fake_output = netD.condition_classifier(fake_img_feature, cond)
    errD_fake= criterion(fake_output, fake_labels)

    if netD.uncondition_classifier is not None:
        real_uncond_output = netD.uncondition_classifier(real_img_feature)
        errD_real_uncond = criterion(real_uncond_output, real_labels)

        fake_uncond_output = netD.uncondition_classifier(fake_img_feature)
        errD_fake_uncond = criterion(fake_uncond_output, fake_labels)

        errD = (errD_real+errD_real_uncond)/2. + (errD_fake+errD_wrong+errD_fake_uncond)/3.
        errD_real =  (errD_real+errD_real_uncond)/2
        errD_fake = (errD_fake+errD_fake_uncond)/2.
    else:
        errD = errD_real + (errD_fake+errD_wrong)*0.5
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()

# TODO: Complete the scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    return x * (max - min) + min

def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    print('Save G/D models')


# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    #fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for img in samples[:-1]:
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
        plt.imshow(img)
