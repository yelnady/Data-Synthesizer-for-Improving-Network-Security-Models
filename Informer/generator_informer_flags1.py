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

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import TokenEmbedding, PositionalEmbedding


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

def get_n_samples(X,Y,mask,n_samples):
    randomList = random.sample(range(0, Y.shape[0]), n_samples)
    return X[randomList], Y[randomList], mask[randomList]

# In real data, if flag sum is 1 --> Then no timestep at all. --> So we do remove those ones by converting them to zeros, then remove from the list
# In real data, there is no flag of length ZERO
def remove_zero_datapoints(X,Y,mask):
    indices_non_zero = torch.nonzero(torch.sum(mask,1)-1).squeeze()
    return X[indices_non_zero], Y[indices_non_zero], mask[indices_non_zero]


# In[4]:


training_real = np.load('../data/google/data_train.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)

#------------------------------------------------------------------Loading One Class------------------------------------------------
real_train_X, real_train_Y_labels, real_train_flags= remove_zero_datapoints(real_train_X, real_train_Y_labels, real_train_flags)

# The pading mask need to be inverted 

padding_mask = real_train_flags == 0 # True when padding, False when considering
real_train_lengths = torch.sum(real_train_flags,1).long()


# In[5]:


window_size = 400


# # The Magic Row

# In[11]:


max_length = 2500


# In[12]:


magic_rows = []
for n_length in real_train_lengths:
    last_number = 1
    n_length=min(max_length,n_length.item())
    step = (1-0.5)/n_length
    magic_row = []
    # Fill with magic numbers
    for _ in range(n_length-1):
        last_number -=step
        magic_row.append(last_number)
    # Fill with zeros   
    magic_row.extend([0]*(max_length - (n_length-1)))
    magic_rows.append(magic_row)
magic_rows = np.array(magic_rows)
magic_rows=np.expand_dims(magic_rows,2)
np.savez('magic_rows',magic=magic_rows)


# In[13]:


magic_rows.shape


# In[14]:


real_train_X = torch.cat((real_train_X,torch.FloatTensor(magic_rows)),2)


# In[15]:


real_train_X.shape


# # Informer Model
# 
# - Later, we need to remove this from here and put in a separate folder

# In[6]:


class MyInformer(nn.Module):

    def __init__(self, n_features=10, d_model=512, n_heads=8, n_hidden=512, e_layers=3, dropout=0.0,seq_length = window_size,
                 attn='prob', mask_flag = True, factor=5,activation='gelu',distil=False, mix=False, output_attention = False):
        
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=n_features, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.InputLinear = nn.Linear(n_features, d_model)
        self.attn = attn
        self.d_model = d_model
        Attn = ProbAttention if attn=='prob' else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(mask_flag=mask_flag, factor=factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix),
                    d_model,
                    n_hidden,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers-1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        #-----------------------------------------------------------------------------------------------------------------------------#
        
        #-----------------------------------------------------------------------------------------------------------------------------#
        self.dropout = nn.Dropout(p=dropout)
        
        self.OutputLinear = nn.Linear(d_model, n_features)
        self.init_weights()
        self.activation= nn.Sigmoid()
        self.end_conv1 = nn.Conv1d(in_channels=102, out_channels=seq_length, kernel_size=1, bias=True)
        
    def init_weights(self):
        initrange = 0.1
        self.InputLinear.weight.data.uniform_(-initrange, initrange)
        self.OutputLinear.bias.data.zero_()
        self.OutputLinear.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, attn_mask): #attn_mask is one mask that provides everything for us
#         src = self.InputLinear(src) * math.sqrt(self.d_model)
        src = self.value_embedding(src) + self.position_embedding(src) # + self.temporal_embedding(x_mark)
        src = self.dropout(src)
        
        output,attns = self.encoder(src,attn_mask) #attn_mask is being passed to FullAttention or ProbAttention

#         output = self.end_conv1(output)
        
        output = self.OutputLinear(output)
        output = self.activation(output)
        return output


# In[8]:



model = MyInformer().to(device)

model.load_state_dict(torch.load('class_all_weights_informer_flags1'))
model.eval()
print()


# # Generating New Data

# In[18]:


################################# The following is the generating part #################################

# Returns: X (The data)
# Returns: masks (e.g. [False,Flase,True,True,True,....,True]), False is the actual Data

resulted_masks = []
generated_dataset_Y=[]
generated_dataset_X=[]

def generate_dataset(X,Y,masks,n_seed,n_samples,max_length):
    datapoint = None
    for n in range(n_samples):
        
        datapoint,y,mask = get_n_samples(X,Y,masks,n_samples=1) # The first 10 timesteps of just one sample
        datapoint = datapoint[:,:n_seed]
        datapoint_len = torch.sum(~mask) #Flip and count, you will get the actual length to generate likewise
        mask = mask[:,:n_seed]
        gc.collect(),torch.cuda.empty_cache()
        E = datapoint.size(2)
        S = datapoint.size(0)
        for t in range(max_length-n_seed): # Loop until 400 timesteps

            predicted = model(datapoint.to(device),None).cpu() # [S,B,E] --> We want just the predicted timestep S
            
            one_new_timestep=predicted[:,-1].unsqueeze(0)
            
            
            datapoint = torch.cat((datapoint,one_new_timestep),1) # add the forecasted timestep
            mask = torch.cat((mask,torch.tensor([[False]])),1)
            S = datapoint.size(1)
            
            if one_new_timestep[...,-1]<0.5 :
#                 print(datapoint.shape[1],datapoint_len)
                datapoint = torch.cat((datapoint,torch.zeros((1,max_length-S,E))),1).cpu() # Pad remainings with zero
                mask =  torch.cat((mask,torch.full((1,max_length-S),True)),1)
                break
            del one_new_timestep
        
        resulted_masks.append(mask.numpy())
        generated_dataset_X.append(datapoint.squeeze().detach().numpy())
        generated_dataset_Y.append(y.item())
        if (n%100==0):
            print('{}/{}'.format(n,n_samples))
        if (n%1000==0):
             np.savez('generated_informer_flags1',X=generated_dataset_X,masks= resulted_masks,Y=generated_dataset_Y)

max_length = 400
n_seed = 2
# Padding Mask Fed here is the Mask where "False is Real Data", True is masked and ignore them


# In[19]:


real_train_X0 ,real_train_Y_labels0 ,padding_mask0= get_one_class(real_train_X ,real_train_Y_labels ,padding_mask,0)
real_train_X1 ,real_train_Y_labels1 ,padding_mask1= get_one_class(real_train_X ,real_train_Y_labels ,padding_mask,1)
real_train_X2 ,real_train_Y_labels2 ,padding_mask2= get_one_class(real_train_X ,real_train_Y_labels ,padding_mask,2)
real_train_X3 ,real_train_Y_labels3 ,padding_mask3= get_one_class(real_train_X ,real_train_Y_labels ,padding_mask,3)


generate_dataset(real_train_X0 ,real_train_Y_labels0 ,padding_mask0,n_seed=n_seed,n_samples=real_train_X0.size(0),max_length=max_length)
generate_dataset(real_train_X1 ,real_train_Y_labels1 ,padding_mask1,n_seed=n_seed,n_samples=real_train_X1.size(0),max_length=max_length)
generate_dataset(real_train_X2 ,real_train_Y_labels2 ,padding_mask2,n_seed=n_seed,n_samples=real_train_X2.size(0),max_length=max_length)
generate_dataset(real_train_X3 ,real_train_Y_labels3 ,padding_mask3,n_seed=n_seed,n_samples=real_train_X3.size(0),max_length=max_length)


# In[ ]:


np.savez('generated_informer_flgas1',X=generated_dataset_X,masks= resulted_masks,Y=generated_dataset_Y)


# In[ ]:




