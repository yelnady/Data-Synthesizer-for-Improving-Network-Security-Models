#!/usr/bin/env python
# coding: utf-8

# # Required Imports

# In[ ]:


# exactly same as V2, but we have added a weight to the loss function.


# In[1]:


import sys, random, math, pickle
from time import time
import numpy as np
import gc
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss
import torch.nn.functional as F
from datetime import timedelta
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
sys.path.append('../DG/gan')
print(device)


# In[2]:


from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')


# In[3]:


with open('../data/google/data_feature_output.pkl', 'rb') as f:
    data_feature = pickle.load(f)    
with open('../data/google/data_attribute_output.pkl', 'rb') as f:
    data_attribute = pickle.load(f)

    
# data_feature is a list of 9 "output.Output" objects, where each object contains attrs -> (is_gen_flag, dim, normalization)
print("X Features")
for i,feature in enumerate(data_feature):
    print("Feature:",i+1," -- Normalization:",feature.normalization, " -- gen_flag:",feature.is_gen_flag, " -- Dim:",feature.dim)

print("\nY Features")
for i,feature in enumerate(data_attribute):
    print("Feature:",i+1," -- Normalization:",feature.normalization, " -- gen_flag:",feature.is_gen_flag, " -- Dim:",feature.dim)


# # Loading Real Train Data

# In[4]:


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


# In[12]:



training_real = np.load('../data/google/data_train_reduced.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)

real_train_X,real_train_Y_labels,real_train_flags = remove_zero_datapoints(real_train_X,real_train_Y_labels,real_train_flags)

real_train_lengths = torch.sum(real_train_flags,1).long()

real_train_masks = real_train_flags == 0 # True when padding, False when actual datapoint


# # The Magic Row

# In[13]:



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


# In[14]:


real_train_X = torch.cat((real_train_X,torch.FloatTensor(magic_rows)),2)


# # TST

# In[16]:



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


# In[17]:




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
    


# In[18]:


B = real_train_X.size(0)
S = real_train_X.size(1)
E = real_train_X.size(2)

# 1- Shift the targets
Input_shifted = real_train_X[:,1:]
Zero_at_the_end = torch.zeros((B,1,E))
targets = torch.cat((Input_shifted,Zero_at_the_end),1) # real_train_X shifted to the left one timestep


# In[19]:


targets=  targets[:,:400]
real_train_masks = real_train_masks[:,:400]
real_train_X = real_train_X[:,:400]

S = real_train_X.size(1)


# # DataSet and DataLoader

# In[20]:


params_dataloader = {'shuffle': True,'num_workers': 2,'batch_size':64} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_train_X, targets, real_train_masks)
train_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# # Training

# In[21]:


torch.cuda.empty_cache() 

from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
print(device)


# In[ ]:



model = TimeSeriesTransformer().to(device)
model.load_state_dict(torch.load('W_transformer_flags1_V4'))

# We need to swap the axis since Transformer takes (S, B, E), we do that using permute(1,0,2)

src_mask = model.generate_square_subsequent_mask(S)

torch.cuda.empty_cache() 

def train(model,train_dataloader,src_mask,n_epochs):
    time_all = time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  
    losses = []
    all_epochs_loss = []
    model.train()
    for epoch in range(n_epochs):
        print('--------------------------Epoch{}-----------------------------'.format(epoch+1))
        time0 = time()
        one_epoch_loss = []
        for idx,(X,target,padding_mask) in enumerate(train_dataloader):
            
            X = X.permute(1,0,2) # [S,B,E]
            optimizer.zero_grad(),gc.collect(),torch.cuda.empty_cache()
            Y_predicted = model(X.to(device),src_mask.to(device),padding_mask.to(device))
            Y_predicted = Y_predicted.permute(1,0,2).cpu()
            
            #--------------------------------------------LOSS MSE---------------------------------------------------#
            mse_loss = nn.MSELoss(reduction='none')
            loss = mse_loss(Y_predicted, target)
            
            loss[...,-1] *= 2.5
            
            # 1- Use reduction='none' loss, and calculate MSE for the first 9 features only
            # 2- Sum the loss across features -> (B,S)
            # 3- Unsqueeze to use bmm -> loss: (B,S,1) , ~padding_mask.float(): (B,S,1)
            # 4- Transpose loss, and bmm(loss,padding_mask) -> (B,1,1)
            # 5- Calculate mean or sum of the batch losses
            
            loss = torch.sum(loss,2)
            padding_mask = (~padding_mask).unsqueeze(2).float()
            loss = loss.unsqueeze(2).permute(0,2,1)

            loss = torch.bmm(loss,padding_mask).mean()

            loss.backward()            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses.append(loss.item())
            one_epoch_loss.append(loss.item())
        
           
            #------------------------------------------END LOSS-----------------------------------------------------#
            del X
            del target
            
            if ((idx+1)%50==0):
                print("Batch {}/{}".format(idx+1,len(train_dataloader)))
        if ((epoch+1)%100==0):
            torch.save(model.state_dict(), 'W_transformer_flags1_V4')

        print("Epoch {} Loss is {}".format(epoch+1,np.mean(one_epoch_loss)))
        print("Epoch {} - Time (in minutes) is {}".format(epoch+1,timedelta(seconds=(time()-time0))))
        all_epochs_loss.append(np.mean(one_epoch_loss))
    
    print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
    print("All Epochs Loss is\n",all_epochs_loss)
    
train(model,train_dataloader,src_mask,n_epochs=400)


# In[ ]:


torch.save(model.state_dict(), 'W_transformer_flags1_V4')


# In[ ]:




