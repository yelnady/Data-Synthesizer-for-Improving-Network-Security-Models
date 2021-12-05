#!/usr/bin/env python
# coding: utf-8

# # Required Imports

# In[1]:


import sys, random, math, pickle
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss
import seaborn as sns
from tensorboard import default
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
sys.path.append('DG/gan')
import gc
print(device)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# In[2]:


from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')


# # Import Real Training Data to Generate New Data from it.
# 
# ### Actual Distribution
# - Class0: 6250
# - Class1: 16124
# - Class2: 21273
# - Class3: 5278

# In[3]:


def get_one_class(X,Y,mask,class_label): # (X, Y, and mask) are the whole dataset that is consisted of many classes, Y is NOT One-Hot Encoded
    indices_class_label = np.where(Y==class_label)
    X,Y,mask = X[indices_class_label], Y[indices_class_label], mask[indices_class_label] 
    indices_non_zero = torch.nonzero(torch.sum(mask,1)-1).squeeze()
    return X[indices_non_zero], Y[indices_non_zero], mask[indices_non_zero]

def get_n_samples(X,n_samples):
    randomList = random.sample(range(0, X.shape[0]), n_samples)
    return X[randomList]


# In[4]:


training_real = np.load('../data/web/data_train.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]


# # PyTorch Transformer Model
# 
# - Later, we need to remove this from here and put in a separate folder

# In[9]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).to(device)


# In[12]:



class TimeSeriesTransformer(nn.Module):

    def __init__(self, n_features=1, d_model=256, n_heads=8, n_hidden=256, n_layers=6, dropout=0.2):
        super().__init__()
        self.model_type = 'Time Series Transformer Model'
        self.InputLinear = nn.Linear(n_features, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        
        self.d_model = d_model
        self.n_features = n_features
        
        self.OutputLinear = nn.Linear(d_model, n_features) # The output of the encoder is similar to the input of the encoder, both are (B,S,d_model)

        self.init_weights()
        self.activation = nn.Tanh()
            

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(-1e6)).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def init_weights(self):
        initrange = 0.1
        self.InputLinear.weight.data.uniform_(-initrange, initrange)
        self.OutputLinear.bias.data.zero_()
        self.OutputLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask,padding_mask):
        src = self.InputLinear(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src, src_mask,padding_mask)
        output = self.OutputLinear(output)
        output = self.activation(output) # output[...,:9] --> Actual 9 values
        return output


# In[13]:



model = TimeSeriesTransformer().to(device)

model.load_state_dict(torch.load('WWT_weights_new'))
model.eval()


# # Generating New Data

# In[8]:


# All generated Data has at least three timesteps because the seed is 2

# We should stop at 2 or at least if S >= datapoint_len 


# In[15]:


################################# The following is the generating part #################################

def generate_dataset(X,n_seed,n_samples,max_length):
 
    generated_dataset_X= torch.empty([0,max_length,1])
    datapoint = None
    for n in range(n_samples):
        
        datapoint = get_n_samples(X,n_samples=1) # The first 10 timesteps of just one sample

        datapoint = datapoint[:,:n_seed].permute(1,0,2).to(device)

        gc.collect(),torch.cuda.empty_cache()
        E = datapoint.size(2)
        S = datapoint.size(0)
        for t in range(max_length-n_seed): # Loop until 550 timesteps
            src_mask = model.generate_square_subsequent_mask(S).to(device)
            predicted = model(datapoint,src_mask,None).to(device) # [S,B,E] --> We want just the predicted timestep S
            one_new_timestep=predicted[-1].unsqueeze(0)
            
            datapoint = torch.cat((datapoint,one_new_timestep)) # add the forecasted timestep
            S = datapoint.size(0)
            
#             if S == datapoint_len : #FIXED SIZE
#                 datapoint = torch.cat((datapoint.cpu(),torch.zeros((max_length-S,1,E)))) # Pad remainings with zero
#                 break
        
            del one_new_timestep
        
        generated_dataset_X = torch.cat((generated_dataset_X,datapoint.permute(1,0,2).cpu().detach()),axis=0)
        if (n%100==0):
            print(n)
        if (n%1000==0):
             np.savez('WWT_generated_new',X=generated_dataset_X)
    return generated_dataset_X

max_length = 550
n_seed = 2
n_samples=real_train_X.size(0)
final_dataset_class_X = generate_dataset(real_train_X,n_seed=n_seed,n_samples=n_samples,max_length=max_length)

np.savez('WWT_generated_new',X=final_dataset_class_X)


# In[ ]:




