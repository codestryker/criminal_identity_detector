import torch
import torch as th
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import sys
import os

sys.path.insert(1,os.getcwd())
sys.path.append(".\\utils\\")

params = dict()
params['IMSIZE']=64

CUDA = False
emb_dim = 256

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# helper display function
def imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

from textEncoder import PretrainedEncoder
from embgen import Embedder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
pretrained_encoder_file= "./networks/InferSent/models/infersent2.pkl"
pretrained_embedding_file= "./networks/InferSent/models/glove.840B.300d.txt"

text_encoder = PretrainedEncoder(
                model_file=pretrained_encoder_file,
                embedding_file=pretrained_embedding_file,
                device=device
            )

embedder = Embedder(
                  embedding_size=4096,
                  hidden_size=256,
                  num_layers=1
              )


text_encoder.eval()
embedder = embedder.to(device) 
embedder.eval()
"""

import pickle
def load_pickle(file_name):
    """
    load a pickle object from the given pickle file
    :param file_name: path to the pickle file
    :return: obj => read pickle object
    """
    with open(file_name, "rb") as pick:
        obj = pickle.load(pick)

    return obj

import subprocess
try:
    import transformers
except (ImportError, AttributeError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])

from transformers import BertModel
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder = BertModel.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('./models/bert/')
encoder.save_pretrained('./models/bert/')
encoder.eval()

from cid_stage_1 import load_pickle, G_Stage1
from cid_stage_2 import G_Stage2

from model import STAGE1_G, STAGE2_G
G1 = STAGE1_G()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

G2 = STAGE2_G(G1)
G2.load_state_dict(torch.load('./models/stage2/netG_epoch_last.pth',map_location=torch.device('cpu')))
G2.STAGE1_G.load_state_dict(torch.load('./models/stage1/netG_epoch_last.pth',map_location=torch.device('cpu')))
G2 = G2.to(device)
G2.eval()

#G1.load_state_dict(torch.load('./models/stage1/netG_epoch_last.pth',map_location=torch.device('cpu')))
#G1.eval()

def text2face(image_description = " smiling face "):
    fixed_noise = Variable(torch.FloatTensor(1, 100).normal_(0, 1),
                                volatile=True)
    with torch.no_grad():
        sample_input = image_description
        input_sentence = torch.tensor(tokenizer.encode(sample_input)).unsqueeze(0)
        out = encoder(input_sentence)
        embeddings_of_last_layer = out[0]
        cls_embeddings = embeddings_of_last_layer[0].clone().detach().requires_grad_(False)
        encoded_caps = np.mean(np.array(cls_embeddings),axis=0)
        txt_embedding = Variable(torch.from_numpy(encoded_caps.reshape(1,768)))
        txt_embedding=txt_embedding.type(torch.FloatTensor)
        with torch.no_grad():
            _, fake, _, _  = G2(txt_embedding, fixed_noise)
        return fake
"""
G1 = G_Stage1()
G1.load_state_dict(torch.load('./models/stage1/netG1_epoch_last.pth'))
G1 = G1.to(device)
G1.eval()

G2 = G_Stage2(G1)
G2.load_state_dict(torch.load('./models/stage2/netG2_epoch_last.pth'))
netG = G2.to(device)
netG.eval()

def text2face(image_description = " smiling face "):
    fixed_noise = torch.FloatTensor(1, 50, 1, 1).normal_(0, 1).to(device)
    with torch.no_grad():
        sample_input = image_description
        _input = [sample_input]
        encoding = text_encoder(_input)
        encoding = encoding.reshape((1,encoding.shape[0], encoding.shape[1]))
        embedding = embedder(torch.from_numpy(encoding).float().to(device))
        encoded_caps = embedding.to(device)[0]
        _,fake, _, _  = netG(fixed_noise, encoded_caps)
        return fake
"""
