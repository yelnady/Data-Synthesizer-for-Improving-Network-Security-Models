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
from tensorboard import default
import torch.nn.functional as F
from datetime import timedelta
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
sys.path.append('DG/gan')
import gc
print(device)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# In[2]:


torch.__version__


# In[3]:


from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')


# # Features & Attributes

# In[4]:


with open('data/google/data_feature_output.pkl', 'rb') as f:
    data_feature = pickle.load(f)    
with open('data/google/data_attribute_output.pkl', 'rb') as f:
    data_attribute = pickle.load(f)

    
# data_feature is a list of 9 "output.Output" objects, where each object contains attrs -> (is_gen_flag, dim, normalization)
print("X Features")
for i,feature in enumerate(data_feature):
    print("Feature:",i+1," -- Normalization:",feature.normalization, " -- gen_flag:",feature.is_gen_flag, " -- Dim:",feature.dim)

print("\nY Features")
for i,feature in enumerate(data_attribute):
    print("Feature:",i+1," -- Normalization:",feature.normalization, " -- gen_flag:",feature.is_gen_flag, " -- Dim:",feature.dim)


# # Loading Real Train Data
# 
# - Class0: 6250
# - Class1: 16124
# - Class2: 21273
# - Class3: 5278

# In[33]:


# Returns all samples from this class that has two timesteps or more, class0 was 6529 data points, now it's 6250
# 1- Get indices of current class_label
# 2- Calculate lengths of the sequence samples
# 3- Choose samples that are nonzero

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
def remove_zero_datapoints(X,Y,Y_labels,mask):
    indices_non_zero = torch.nonzero(torch.sum(mask,1)-1).squeeze()
    return X[indices_non_zero], Y[indices_non_zero], Y_labels[indices_non_zero], mask[indices_non_zero]


# In[34]:


training_real = np.load('data/google/data_train.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)


real_train_X, real_train_Y, real_train_Y_labels, real_train_flags = remove_zero_datapoints(
    real_train_X, real_train_Y, real_train_Y_labels, real_train_flags.float())

real_train_lengths = torch.sum(real_train_flags,1).long()


# # PyTorch Transformer Model

# In[36]:


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

# In[63]:


class TimeSeriesTransformer(nn.Module):

    def __init__(self, n_features=9,n_classes = 4, d_model=512, n_heads=8, n_hidden=512, n_layers=8, dropout=0.1):
        super().__init__()
        self.model_type = 'Time Series Transformer Model'
        self.PromptLinear = nn.Linear(n_classes, d_model)
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
        return mask.to(device)

    def init_weights(self):
        initrange = 0.1
        self.InputLinear.weight.data.uniform_(-initrange, initrange)
        self.OutputLinear.bias.data.zero_()
        self.OutputLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, prompt, src_mask,padding_mask):
        p = self.PromptLinear(prompt).unsqueeze(0) * math.sqrt(self.d_model)
        v = self.InputLinear(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(torch.cat((p,v)))
        output = self.transformer_encoder(src, src_mask,padding_mask)
        output = self.OutputLinear(output)
        output = self.activation(output) # output[...,:9] --> Actual 9 values
        return output
    


# # Preparing Inputs, Masks, and Targets
# 
# - Inputs, will be the original data, except the last actual timestep will be masked in the padding_mask
# - Targets, will be the original data, but shifted to the left one step, and added zero at the end

# In[64]:


B = real_train_X.size(0)
S = real_train_X.size(1)
E = real_train_X.size(2)

# 1- Shift the targets
Input_shifted = real_train_X[:,1:]
Zero_at_the_end = torch.zeros((B,1,E))
targets = torch.cat((Input_shifted,Zero_at_the_end),1)

real_train_masks = real_train_flags == 0 # True when padding, False when actual datapoint


# In[83]:


#############################################--------WINDOW SIZE-------------###########################################

targets=  targets[:,:401]
real_train_masks = real_train_masks[:,:400]
real_train_X = real_train_X[:,:400]


# We will need to add one more step for the padding mask to work as our prompt flag for first time step
real_train_masks = torch.cat((real_train_masks.cpu(),torch.full((real_train_X.size(0),1),False)),1)


# # Creating Dataset and Dataloader

# In[84]:


params_dataloader = {'shuffle': True,'num_workers': 2,'batch_size':100} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_train_X, real_train_Y.float() ,targets, real_train_masks)
train_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# # Training

# In[86]:


torch.cuda.empty_cache() 

from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
print(device)


# In[87]:


model = TimeSeriesTransformer().to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_parameters(model)


# In[88]:


model = TimeSeriesTransformer().to(device)

# We need to swap the axis since Transformer takes (S, B, E), we do that using permute(1,0,2)

S = real_train_X.size(1) + 1 #1 is the prompt 

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
        for idx,(X, label, target, padding_mask) in enumerate(train_dataloader):
            X = X.permute(1,0,2) # [S,B,E]
            optimizer.zero_grad(),gc.collect(),torch.cuda.empty_cache()
            Y_predicted = model(X.to(device),label.to(device),src_mask.to(device),padding_mask.to(device))
            Y_predicted = Y_predicted.permute(1,0,2).to(device)
            
            #--------------------------------------------LOSS MSE---------------------------------------------------#
            mse_loss = nn.MSELoss(reduction='none')
            
            loss = mse_loss(Y_predicted, target.to(device))
            # 1- Use reduction='none' loss, and calculate MSE for the first 9 features only
            # 2- Sum the loss across features -> (B,S)
            # 3- Unsqueeze to use bmm -> loss: (B,S,1) , ~padding_mask.float(): (B,S,1)
            # 4- Transpose loss, and bmm(loss,padding_mask) -> (B,1,1)
            # 5- Calculate mean or sum of the batch losses
            
            loss = torch.sum(loss,2)
            padding_mask = (~padding_mask).unsqueeze(2).float().to(device)
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

        print("Epoch {} Loss is {}".format(epoch+1,np.mean(one_epoch_loss)))
        print("Epoch {} - Time (in minutes) is {}".format(epoch+1,timedelta(seconds=(time()-time0))))
        all_epochs_loss.append(np.mean(one_epoch_loss))
    
    print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
    print("All Epochs Loss is\n",all_epochs_loss)
    
train(model,train_dataloader,src_mask,n_epochs=400)


# In[ ]:


torch.save(model.state_dict(), 'class_all_weights_prompt')


# In[ ]:




