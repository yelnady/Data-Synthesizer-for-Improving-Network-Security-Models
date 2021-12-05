#!/usr/bin/env python
# coding: utf-8

# # Required Imports

# In[22]:


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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('../DG/gan')
import gc
print(device)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# In[23]:


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

# In[24]:


def get_one_class(X,Y_labels,flag,class_label):
    indices_class_label = np.where(Y_labels==class_label)
    return X[indices_class_label], Y_labels[indices_class_label], flag[indices_class_label] 
    
def get_n_samples(X,Y_labels,flag,n_samples):
    randomList = random.sample(range(0, Y_labels.shape[0]), n_samples)
    return X[randomList], Y_labels[randomList], flag[randomList]

# In real data, if flag sum is 1 --> Then no timestep at all. 
            # So we do remove those ones by converting them to zeros, then return only non-zero flags indices
# In real data, there is no flag of length ZERO
def remove_zero_datapoints(X,Y_labels,flag):
    indices_non_zero = torch.nonzero(torch.sum(flag,1)-1).squeeze()
    return X[indices_non_zero], Y_labels[indices_non_zero], flag[indices_non_zero]


# In[25]:


training_real = np.load('../data/google/data_train_reduced.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)

real_train_X,real_train_Y_labels,real_train_flags = remove_zero_datapoints(real_train_X,real_train_Y_labels,real_train_flags)

real_train_lengths = torch.sum(real_train_flags,1).long()

real_train_masks = real_train_flags == 0 # True when padding, False when actual datapoint


# # The Magic Row

# In[26]:


def generate_magic_row(lengths=real_train_lengths, max_length = 400, n_timesteps = real_train_X.shape[1]):
    magic_rows = []
    for n_length in lengths:
        last_number = 1
        n_length=min(n_timesteps,n_length.item()) 
        step = (1-0.5)/(n_length-1) # -1 to make the last element less than 0.5
        magic_row = []
      
        # Fill with magic numbers
        for _ in range(n_length):
            last_number -=step
            magic_row.append((last_number))
        
        # Fill with zeros   
        magic_row.extend([0]*(n_timesteps - n_length))
        magic_rows.append(magic_row)
    magic_rows = np.array(magic_rows)
    magic_rows=np.expand_dims(magic_rows,2)
    return magic_rows

magic_rows = generate_magic_row()


# In[27]:


real_train_X = torch.cat((real_train_X,torch.FloatTensor(magic_rows)),2)


# # PyTorch Transformer Model
# 
# - Later, we need to remove this from here and put in a separate folder

# In[28]:


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


# In[29]:


class TimeSeriesTransformer(nn.Module):

    def __init__(self, n_features=10, d_model=200, n_heads=8, n_hidden=200, n_layers=8, dropout=0.0):
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
        self.activation = nn.Sigmoid()
            

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(-1e6)).masked_fill(mask == 1, float(0.0))
        return mask

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
    


# In[30]:


model = TimeSeriesTransformer().to(device)

model.load_state_dict(torch.load('W_transformer_flags1_V3'))
model.eval()
print()


# # Generating New Data

# In[31]:


# All generated Data has at least three timesteps because the seed is 2

# We should stop at 2 or at least if S >= datapoint_len 
import gc
gc.collect()


# In[39]:


################################# The following is the generating part #################################

# Returns: X (The data)
# Returns: masks (e.g. [False,Flase,True,True,True,....,True]), False is the actual Data

resulted_masks = []
generated_dataset_Y=[]
generated_dataset_X=[]

def generate_dataset(X,Y,masks,n_seed,n_samples,max_length):

    datapoints,labels,sampled_masks = get_n_samples(X,Y,masks,n_samples=n_samples) 
    
    for n,(datapoint,y,mask) in enumerate(zip(datapoints,labels,sampled_masks)):
        datapoint = datapoint[None,:n_seed].permute(1,0,2)

        datapoint_len = torch.sum(~mask) #Flip and count, you will get the actual length to generate likewise
        mask = mask[None,:n_seed].to(device)
        
        gc.collect(),torch.cuda.empty_cache()
        E = datapoint.size(2)
        S = datapoint.size(0)
        for t in range(max_length-n_seed): # Loop until 400 timesteps
            src_mask = model.generate_square_subsequent_mask(S)

            predicted = model(datapoint.to(device),src_mask.to(device),mask.to(device))# [S,B,E] --> We want just the predicted timestep S
            one_new_timestep=predicted[-1][None]

            datapoint = torch.cat((datapoint.to(device),one_new_timestep)) # add the forecasted timestep
            mask = torch.cat((mask,torch.tensor([[False]]).to(device)),1 )
            
            S = datapoint.size(0)
            gc.collect()
            if  one_new_timestep[...,-1].item() <=0.5 : 
#                 print(S,datapoint_len)
                datapoint = torch.cat((datapoint,torch.zeros((max_length-S,1,E)).to(device))) # Pad remainings with zero
                mask =  torch.cat((mask,torch.full((1,max_length-S),True).to(device)),1)
                break
        resulted_masks.append(mask.cpu().numpy())
        generated_dataset_X.append(datapoint.permute(1,0,2).squeeze().cpu().detach().numpy())
        generated_dataset_Y.append(y.item())

        if (n%100==0):
            print('{}/{}'.format(n,n_samples))
        if (n%1000==0):
             np.savez('npz_transformer_flags1_V3',X=generated_dataset_X,masks= resulted_masks,Y=generated_dataset_Y)

max_length = 400
n_seed = 1
# Padding Mask Fed here is the Mask where "False is Real Data", True is masked and ignore them


# In[40]:


real_train_X0 ,real_train_Y_labels0 ,padding_mask0= get_one_class(real_train_X ,real_train_Y_labels ,real_train_masks,0)
real_train_X1 ,real_train_Y_labels1 ,padding_mask1= get_one_class(real_train_X ,real_train_Y_labels ,real_train_masks,1)
real_train_X2 ,real_train_Y_labels2 ,padding_mask2= get_one_class(real_train_X ,real_train_Y_labels ,real_train_masks,2)
real_train_X3 ,real_train_Y_labels3 ,padding_mask3= get_one_class(real_train_X ,real_train_Y_labels ,real_train_masks,3)


generate_dataset(real_train_X0 ,real_train_Y_labels0 ,padding_mask0,n_seed=n_seed,n_samples=real_train_X0.size(0),max_length=max_length)
generate_dataset(real_train_X1 ,real_train_Y_labels1 ,padding_mask1,n_seed=n_seed,n_samples=real_train_X1.size(0),max_length=max_length)
generate_dataset(real_train_X2 ,real_train_Y_labels2 ,padding_mask2,n_seed=n_seed,n_samples=real_train_X2.size(0),max_length=max_length)
generate_dataset(real_train_X3 ,real_train_Y_labels3 ,padding_mask3,n_seed=n_seed,n_samples=real_train_X3.size(0),max_length=max_length)


# In[ ]:


# The variables are global variables

np.savez('npz_transformer_flags1_V3',X=generated_dataset_X,masks= resulted_masks,Y=generated_dataset_Y)


# In[38]:


gc.collect()


# In[ ]:




