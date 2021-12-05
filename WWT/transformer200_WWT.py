#!/usr/bin/env python
# coding: utf-8

# # Required Imports

# In[34]:


import sys, random, math, pickle
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss
from tensorboard import default
import torch.nn.functional as F
from datetime import timedelta
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('../DG/gan')
import gc
print(device)


# In[6]:


torch.__version__


# In[7]:


from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')


# # Features & Attributes

# In[8]:


with open('../data/web/data_feature_output.pkl', 'rb') as f:
    data_feature = pickle.load(f)    
with open('../data/web/data_attribute_output.pkl', 'rb') as f:
    data_attribute = pickle.load(f)

    
# data_feature is a list of 9 "output.Output" objects, where each object contains attrs -> (is_gen_flag, dim, normalization)
print("X Features")
for i,feature in enumerate(data_feature):
    print("Feature:",i+1," -- Normalization:",feature.normalization, " -- gen_flag:",feature.is_gen_flag, " -- Dim:",feature.dim)

print("\nY Features")
for i,feature in enumerate(data_attribute):
    print("Feature:",i+1," -- Normalization:",feature.normalization, " -- gen_flag:",feature.is_gen_flag, " -- Dim:",feature.dim)


# In[14]:


training_real = np.load('../data/web/data_train.npz')
real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]


# # PyTorch Transformer Model

# In[55]:


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


# S is the source sequence length, T is the target sequence length, B is the batch size, E is the feature number
# 
# - src: (S, B, E)
# - src_mask: (S, S) # For Self-Attention
# - src_key_padding_mask: (B, S)   ---- The positions with the value of "True" will be ignored while the position with the value of False will be unchanged.
# - output: (T, B, E)
# 
# In the paper n_hidden was 64 and d_model is 512, next we try n_hidden 2048, and d_model 512
# 

# In[56]:


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


# In[57]:


B = real_train_X.size(0)
S = real_train_X.size(1)
E = real_train_X.size(2)


Input_shifted = real_train_X[:,1:]
Zero_at_the_end = torch.zeros((B,1,E))
targets = torch.cat((Input_shifted,Zero_at_the_end),1)


# In[58]:


targets.shape


# # Creating Dataset and Dataloader

# In[59]:


params_dataloader = {'shuffle': True,'num_workers': 4,'batch_size':128} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_train_X, targets)
train_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# # Training

# In[60]:


# # import sys

# # These are the usual ipython objects, including this one you are creating
# ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# # Get a sorted list of the objects and their sizes
# sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# In[61]:


torch.cuda.empty_cache() 

from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
print(device)


# In[62]:


model = TimeSeriesTransformer().to(device)

# We need to swap the axis since Transformer takes (S, B, E), we do that using permute(1,0,2)

src_mask = model.generate_square_subsequent_mask(S)

torch.cuda.empty_cache() 

def train(model,train_dataloader,src_mask,n_epochs):
    time_all = time()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  
    all_epochs_loss = []
    model.train()
    for epoch in range(n_epochs):
        print('--------------------------Epoch{}-----------------------------'.format(epoch+1))
        time0 = time()
        one_epoch_loss = []
        for idx,(X,target) in enumerate(train_dataloader):

            X = X.permute(1,0,2) # [S,B,E]
            optimizer.zero_grad(),gc.collect(),torch.cuda.empty_cache()
            Y_predicted = model(X.to(device),src_mask.to(device),None)
            Y_predicted = Y_predicted.permute(1,0,2).to(device)
            
            #--------------------------------------------LOSS MSE---------------------------------------------------#
            mse_loss = nn.MSELoss()
            
            loss = mse_loss(Y_predicted, target.to(device))
            loss.backward()            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            one_epoch_loss.append(loss.item())         
            #------------------------------------------END LOSS-----------------------------------------------------#
            del X
            del target
            
            if ((idx+1)%50==0):
                print("Batch {}/{}".format(idx+1,len(train_dataloader)))
        print("Epoch {} Loss is {}".format(epoch+1,np.mean(one_epoch_loss)))
        print("Epoch {} - Time (in minutes) is {}".format(epoch+1,timedelta(seconds=(time()-time0))))
        all_epochs_loss.append(np.mean(one_epoch_loss))
    
    print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
    print("All Epochs Loss is\n",all_epochs_loss)
    
train(model,train_dataloader,src_mask,n_epochs=300)


# In[ ]:


# mse_loss = nn.MSELoss(reduction='none')

# loss = mse_loss(real_train_X_one_class[:3][...,:9].to(device), targets[:3][...,:9].to(device))
# print(loss.shape)
# loss = (loss * mask.float()).sum() # gives \sigma_euclidean over unmasked elements

# non_zero_elements = mask.sum()
# mse_loss_val = loss / non_zero_elements

# # now doing backpropagation
# mse_loss_val.backward()


# In[ ]:


torch.save(model.state_dict(), 'WWT_weights_new')

