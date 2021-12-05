#!/usr/bin/env python
# coding: utf-8

# # Required Imports

# In[1]:


# V2: same as V1, but 256 and PL
# V3: same as V2 but 128 datapoints in batch, 256 the dim size, and 16 heads per layer

# In[1]:


import sys, random, math, pickle
from time import time
import numpy as np
import gc
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from datetime import timedelta
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
import torchmetrics.functional as FM
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
sys.path.append('../DG/gan')


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

# In[5]:


def get_one_class(X,Y,Y_labels,flag,class_label):
    indices_class_label = np.where(Y_labels==class_label)
    return X[indices_class_label], Y[indices_class_label], Y_labels[indices_class_label], flag[indices_class_label] 
    
def get_n_samples(X,Y,Y_labels,flag,n_samples):
    randomList = random.sample(range(0, Y_labels.shape[0]), n_samples)
    return X[randomList], Y[randomList], Y_labels[randomList], flag[randomList]

# In real data, if flag sum is 1 --> Then no timestep at all. 
            # So we do remove those ones by converting them to zeros, then return only non-zero flags indices
# In real data, there is no flag of length ZERO
def remove_zero_datapoints(X,Y,Y_labels,flag):
    indices_non_zero = torch.nonzero(torch.sum(flag,1)-1).squeeze()
    return X[indices_non_zero],  Y[indices_non_zero], Y_labels[indices_non_zero], flag[indices_non_zero]


# In[46]:


training_real = np.load('../data/google/data_train_reduced.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)

real_train_X,real_train_Y,real_train_Y_labels,real_train_flags = remove_zero_datapoints(real_train_X,real_train_Y,
                                                                           real_train_Y_labels,real_train_flags)

real_train_lengths = torch.sum(real_train_flags,1).long()

real_train_masks = real_train_flags == 0 # True when padding, False when actual datapoint


# In[47]:


val_real = np.load('../data/google/data_train_val.npz')

real_val_X = torch.from_numpy(val_real['data_feature']).float() #[50000, 2500, 9]
real_val_Y = torch.from_numpy(val_real['data_attribute']) #[50000,4]
real_val_Y_labels = torch.argmax(real_val_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_val_flags = torch.from_numpy(val_real['data_gen_flag'])   # (50000, 2500)

real_val_X, real_val_Y, real_val_Y_labels, real_val_flags = remove_zero_datapoints(real_val_X,real_val_Y,
                                                                     real_val_Y_labels,real_val_flags)

real_val_lengths = torch.sum(real_val_flags,1).long()

real_val_masks = real_val_flags == 0 # True when padding, False when actual datapoint


# # DataSet and DataLoader

# In[48]:


B = real_train_X.size(0)
S = real_train_X.size(1)
E = real_train_X.size(2)

# 1- Shift the targets
Input_shifted = real_train_X[:,1:]
Zero_at_the_end = torch.zeros((B,1,E))
targets = torch.cat((Input_shifted,Zero_at_the_end),1) # real_train_X shifted to the left one timestep

targets=  targets[:,:400]
real_train_masks = real_train_masks[:,:400]
real_train_X = real_train_X[:,:400]

S = real_train_X.size(1)

# We will need to add one more step for the padding mask to work as our prompt flag for first time step
real_train_masks = torch.cat((real_train_masks.cpu(),torch.full((real_train_X.size(0),1),False)),1)

params_dataloader = {'shuffle': True,'num_workers':4 ,'batch_size':128} # No need to shuffle rn, they are all the same class
# "num_workers" is how many subprocesses to use for data loading.
dataset = torch.utils.data.TensorDataset(real_train_X,  real_train_Y.float(), targets, real_train_masks)
train_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# In[61]:


# Validation Dataset and DataLoader 

B = real_val_X.size(0)
S = real_val_X.size(1)
E = real_val_X.size(2)

Input_shifted = real_val_X[:,1:]
Zero_at_the_end = torch.zeros((B,1,E))
targets = torch.cat((Input_shifted,Zero_at_the_end),1) # real_train_X shifted to the left one timestep

targets=  targets[:,:400]
real_val_masks = real_val_masks[:,:400]
real_val_X = real_val_X[:,:400]

S = real_val_X.size(1)

real_val_masks = torch.cat((real_val_masks.cpu(),torch.full((real_val_X.size(0),1),False)),1)

params_dataloader = {'shuffle': False,'num_workers':16 ,'batch_size':512} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_val_X, real_val_Y.float(), targets, real_val_masks)
val_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# In[62]:


real_val_masks.shape


# # TST

# In[63]:


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
        return self.dropout(x)


# In[68]:


class TimeSeriesTransformer(pl.LightningModule):

    def __init__(self, n_features=9,n_classes = 4, d_model=256, n_heads=16, n_hidden=256, n_layers=12, dropout=0,
                 S=400,n_metadata=1):
        super().__init__()
        self.model_type = 'Time Series Transformer Model - PromptV3'
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
        self.n_metadata = n_metadata
        self.S =  S
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(-1e7)).masked_fill(mask == 1, float(0.0))
        return mask 

    def init_weights(self):
        initrange = 0.1
        self.InputLinear.weight.data.uniform_(-initrange, initrange)
        self.OutputLinear.bias.data.zero_()
        self.OutputLinear.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, prompt, src_mask,padding_mask):
        p = self.PromptLinear(prompt).unsqueeze(0) * math.sqrt(self.d_model)
        v = self.InputLinear(src)  * math.sqrt(self.d_model)
        src = torch.cat((p,v))
        src += self.positional_encoding(torch.cat((p,v)))
        output = self.transformer_encoder(src, src_mask,padding_mask)
        output = self.OutputLinear(output)
        output = self.activation(output) # output[...,:9] --> Actual 9 values
        return output
    
    
    def training_step(self, batch, batch_idx):

        X,y_one_hot,target,padding_mask = batch
        src_mask = self.generate_square_subsequent_mask(self.S + self.n_metadata).cuda()
        X = X.permute(1,0,2)
        Y_predicted = self(X,y_one_hot,src_mask,padding_mask)
        Y_predicted = Y_predicted.permute(1,0,2)
            
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(Y_predicted[:,self.n_metadata:], target)
        loss = torch.sum(loss,2)
        padding_mask = (~padding_mask[:,self.n_metadata:]).unsqueeze(2).float()
        loss = loss.unsqueeze(2).permute(0,2,1)
        loss = torch.bmm(loss,padding_mask).mean()
            
#         acc = FM.accuracy(F.softmax(y_hat,1), y)
        return {'loss': loss,} # will call loss.backward() on what we return exactly. 
    
    def training_epoch_end(self, outputs):
        print("Epoch Loss:",torch.stack([x["loss"] for x in outputs]).mean().item())
    
    # Lightning disables gradients, puts model in eval mode, and does everything needed for validation.
    def validation_step(self, batch, batch_idx):
        X,y_one_hot,target,padding_mask = batch
        src_mask = self.generate_square_subsequent_mask(self.S + self.n_metadata).cuda()
        X = X.permute(1,0,2)
        Y_predicted = self(X,y_one_hot,src_mask,padding_mask)
        Y_predicted = Y_predicted.permute(1,0,2)
            
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(Y_predicted[:,self.n_metadata:], target)
        loss = torch.sum(loss,2)
        padding_mask = (~padding_mask[:,self.n_metadata:]).unsqueeze(2).float()
        loss = loss.unsqueeze(2).permute(0,2,1)
        loss = torch.bmm(loss,padding_mask).mean()
        self.log('val_loss', loss)
        return {'val_loss': loss,} # We may return the predictions themselves
    
    def validation_epoch_end(self, outputs):
        print("Validation Loss:",torch.stack([x["val_loss"] for x in outputs]).mean().item())
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    


# # Training

# **Notes on EarlyStopping:**
# - The EarlyStopping callback runs at the **end of every validation epoch**, which, under the default configuration, happen after **every training epoch**.
# -  However, the frequency of validation can be modified by setting various parameters in the Trainer, for example **check_val_every_n_epoch and val_check_interval**.
# - Note that the **patience** parameter counts the number of **validation epochs with no improvement**, and **not the number of training epochs**. 
#     - Therefore, with parameters **check_val_every_n_epoch=10 and patience=3**, the trainer will perform at least **40 training epochs before being stopped**. 

# In[69]:



def main():
    # pl.seed_everything(42, workers=True) --> sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    time_all = time()

    model = TimeSeriesTransformer()
    early_stop_callback = EarlyStopping(monitor='val_loss',patience=3, verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint()
    trainer = pl.Trainer(gpus=1,max_epochs=400, progress_bar_refresh_rate=50,
                        callbacks=[checkpoint_callback]
                         )
    
    trainer.fit(model,train_dataloader,val_dataloader)
    print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
    print(checkpoint_callback.best_model_path)

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




