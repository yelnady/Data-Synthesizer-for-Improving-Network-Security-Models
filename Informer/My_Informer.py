#!/usr/bin/env python
# coding: utf-8

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
from torch.nn import TransformerEncoder, TransformerEncoderLayer

sys.path.append('../DG/gan')
import gc
print(device)


# ------------------------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import TokenEmbedding, PositionalEmbedding


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
# 
# - Class0: 6250
# - Class1: 16124
# - Class2: 21273
# - Class3: 5278

# In[5]:


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
def remove_zero_datapoints(X,Y,mask):
    indices_non_zero = torch.nonzero(torch.sum(mask,1)-1).squeeze()
    return X[indices_non_zero], Y[indices_non_zero], mask[indices_non_zero]


def prepare_attn_mask(padding_mask):
    attn_mask = torch.zeros((padding_mask.shape[0],padding_mask.shape[1],padding_mask.shape[1]), dtype=torch.bool) #[B,L,L]
    for idx in range(len(padding_mask)):
        x = attn_mask.shape[1]
        attn_mask[idx] = (~torch.logical_and(~padding_mask[idx].view(-1,1).expand(x,x),
                                              ~padding_mask[idx].expand(x,x))).logical_or(torch.ones(x,x).triu(1))
    return attn_mask #[B,L,L]


# In[6]:


training_real = np.load('../data/google/data_train.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)


real_train_X,real_train_Y_labels,real_train_flags = remove_zero_datapoints(real_train_X,real_train_Y_labels,real_train_flags)

real_train_lengths = torch.sum(real_train_flags,1).long()


# # Preparing Inputs, Masks, and Targets
# 
# - Inputs, will be the original data, except the last actual timestep will be masked in the padding_mask
# - Targets, will be the original data, but shifted to the left one step, and added zero at the end

# In[7]:


B = real_train_X.size(0)
S = real_train_X.size(1)
E = real_train_X.size(2)

# 1- Shift the targets
Input_shifted = real_train_X[:,1:]
Zero_at_the_end = torch.zeros((B,1,E))
targets = torch.cat((Input_shifted,Zero_at_the_end),1)

# 2- Shift the masks to be the same as targets

real_train_masks = real_train_flags == 0 # True when padding, False when actual datapoint
real_train_masks = real_train_masks[:,1:]
Zero_at_the_end = torch.zeros((B,1))==0
real_train_masks = torch.cat((real_train_masks,Zero_at_the_end),1)


# In[8]:



#############################################--------WINDOW SIZE-------------###########################################

window_size = S = 400
targets=  targets[:,:window_size]
real_train_masks = real_train_masks[:,:window_size]
real_train_X = real_train_X[:,:window_size]

# S = real_train_X.size(1)


# # Creating Dataset and Dataloader

# In[9]:


B = 64
params_dataloader = {'shuffle': True,'num_workers':2,'batch_size':B} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_train_X, targets, real_train_masks)
train_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# In[10]:


from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(1)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')


# # Informer

# In[11]:


# Informer: DataEmbedding(x_enc, x_mark_enc):enc_out  --> encoder(enc_out,enc_self_mask):enc_out, attns 
#           DataEmbedding(x_dec, x_mark_dec):dec_out  --> decoder(dec_out, enc_out, dec_self_mask, dec_enc_mask):dec_out
#           projection(dec_out): dec_out [Batch, n_timesteps , n_features]

# Encoder takes 3 params (list of attn_layers/Encoder_Layer, List of Conv Layer if distil, normalization layer/nn.LayerNorm) -> returns x, attn
# EncoderLayer takes 4 params (AttentionLayer, d_model, n_hidden = n_hidden or 4*d_model, dropout=0.1, activation="relu") -> returns x, attns
# Attn takes 5 params (mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False)



#  d_model=512; n_heads=8; e_layers=3; d_ff=512; n_features = 9; dropout=0.0; attn='full'; activation='gelu'; ; distil=False; mix=True;
        
# We need to change the gelu and relu to allow for Sigmoid and Tanh in the Encoder
# output_attention = Flase because Encoder returns  (enc_out, attns)
# d_ff is like d_hidden


# Before Encoder we need to feed a positional encoding and linear layer to map the dimensions to that encoder
# After Encoder we need to remap the dimensiosn 


# In[12]:


class MyInformer(nn.Module):

    def __init__(self, n_features=9, d_model=512, n_heads=8, n_hidden=512, e_layers=3, dropout=0.0,seq_length = window_size,
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


# # Training

# In[13]:


gc.collect(),torch.cuda.empty_cache()


# In[16]:


# x,target,mask_sample =iter(train_dataloader).next()


# In[19]:


model = MyInformer().to(device)

torch.cuda.empty_cache() 

def train(model,train_dataloader,n_epochs):
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

            optimizer.zero_grad(),gc.collect(),torch.cuda.empty_cache()

            attn_mask = prepare_attn_mask(padding_mask)
            
            Y_predicted = model(X.to(device),attn_mask)

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
            #-------------------------------------LOSS Cross Entropy-----------------------------------------------#
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
    if ((epoch+1)%100==0):
            torch.save(model.state_dict(), 'class_all_weights_informer')
       
    print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
    print("All Epochs Loss is\n",all_epochs_loss)
    
train(model,train_dataloader,n_epochs=400)


# In[ ]:


# prob gives 0.282603288663466 in 0:08:23 minutes 
# full gives 0.13638 in 0:10:14

# Full gives 0.1431730 using just n_layers = 3 in just 0:07:09 minutes
#prob gives 0.1298 using  just n_layers = 3 in just 0:07:11.3


# Using distill and 3 layers only I got -> 1.43092 then 0.078535


# In[ ]:


torch.save(model.state_dict(), 'class_all_weights_informer')


# In[ ]:


# mask_flag doesn't provide anything related to our paddings, but we can provide our own matrix
# The weight mask, which is the combination of the padding and causal masks, is used to know which positions to fill with 
# −∞ before computing the softmax, which will be zero after it.

# first src_mask is used to block specific positions from attending and then key_padding_mask is used to block attending to pad tokens.


# In ProbAttention or FullAttention, if mask_flag=True and attn_mask {given in encoder forward} = None --> Informer makes Attention Matrix Mask for you
#                                  , if mask_flag=True and given mask --> Informer just converts True values to (-np.inf)
#                                  , if mask_flag=False doesn't care about any type of masks at all

#IMPORTANT -> We need just to send a attn_mask where False is Important, and True to ignore and it will be set to (-np.inf) by mask_flag=True

# torch.masked_fill_(matrix,value): Fills elements of tensor with value where mask is True. 

# Mask Shape -> mask_shape = [B, 1, L, L] -> [B, n_heads, L, L]

# The mask we provide need to have an attribute called mask as done in utils/masking so I build class MyProbMask


# ## We will pass everything ready in the attn_mask from EncoderLayer.forward() -> AttentionLayer.forward() -> ProbAttention.forward(attn_mask) -> _update_context(attn_mask)

# - output_seq_len=label_len+pred_len because in decoder we have label_len is the history provided to be the X_token as in Figure1, then we tell it to predict the following lengths, with a total of output_seq_len

# In[ ]:


# We need to see difference between PropAttention and FullAttention
# We need to know the difference implementations for attn_matrix in Prob and Full (Currently using Full)
# We need to see why distill = False, makes the dimensions correct and ignores the conv layers

# We need to understand factor and mix params
# We need to know differnce between Encoder  and EncoderStack

# Current Implementation -> No Linear Layer at all! and uses Full Attention which causes problems

