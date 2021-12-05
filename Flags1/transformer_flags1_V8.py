#!/usr/bin/env python
# coding: utf-8

# # Required Imports

# In[1]:


# V4: exactly same as V2, but we have added a weight to the loss function.
# V6: Same as V4, but with PyTorch Lightning
# V7: new loss flatten, 128 batch size on 8 workers, 256 dimensions, no weighted flags, patience = 5
# V8: same as V7, but loss*2.5

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


# In[5]:


training_real = np.load('../data/google/data_train_reduced.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)

real_train_X,real_train_Y_labels,real_train_flags = remove_zero_datapoints(real_train_X,real_train_Y_labels,real_train_flags)

real_train_lengths = torch.sum(real_train_flags,1).long()

real_train_masks = real_train_flags == 0 # True when padding, False when actual datapoint


# In[6]:


val_real = np.load('../data/google/data_train_val.npz')

real_val_X = torch.from_numpy(val_real['data_feature']).float() #[50000, 2500, 9]
real_val_Y = torch.from_numpy(val_real['data_attribute']) #[50000,4]
real_val_Y_labels = torch.argmax(real_val_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_val_flags = torch.from_numpy(val_real['data_gen_flag'])   # (50000, 2500)

real_val_X,real_val_Y_labels,real_val_flags = remove_zero_datapoints(real_val_X,real_val_Y_labels,real_val_flags)

real_val_lengths = torch.sum(real_val_flags,1).long()

real_val_masks = real_val_flags == 0 # True when padding, False when actual datapoint


# # The Magic Row

# In[7]:



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


# In[8]:


magic_rows_train = generate_magic_row(real_train_lengths)
real_train_X = torch.cat((real_train_X,torch.FloatTensor(magic_rows_train)),2)

magic_rows_val = generate_magic_row(real_val_lengths)
real_val_X = torch.cat((real_val_X,torch.FloatTensor(magic_rows_val)),2)


# In[9]:


real_val_X.shape


# # DataSet and DataLoader

# In[10]:


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

params_dataloader = {'shuffle': True,'num_workers':2 ,'batch_size':64} # No need to shuffle rn, they are all the same class
# "num_workers" is how many subprocesses to use for data loading.
dataset = torch.utils.data.TensorDataset(real_train_X, targets, real_train_masks)
train_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# In[11]:


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

params_dataloader = {'shuffle': False,'num_workers':8 ,'batch_size':128} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_val_X, targets, real_val_masks)
val_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# # TST

# In[12]:


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


# In[13]:


class TimeSeriesTransformer(pl.LightningModule):

    def __init__(self, n_features=10, d_model=256, n_heads=8, n_hidden=256, n_layers=8, dropout=0.0,S=400):
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
        self.n_features = n_features
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(-1e7)).masked_fill(mask == 1, float(0.0))
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
    
    def training_step(self, batch, batch_idx):

        X,target,padding_mask = batch
        src_mask = self.generate_square_subsequent_mask(S).cuda()
        X = X.permute(1,0,2)
        Y_predicted = self(X,src_mask,padding_mask)
        Y_predicted = Y_predicted.permute(1,0,2)
            
        mse_loss = nn.MSELoss(reduction='none')
        diff2 = mse_loss(Y_predicted, target)
        diff2[...,-1] *= 2
        diff2= diff2.flatten()
        flags = (~padding_mask).unsqueeze(2).expand(-1,-1,self.n_features).float().flatten()
        loss = diff2*flags
        loss = torch.sum(loss) / torch.sum(flags)
#         acc = FM.accuracy(F.softmax(y_hat,1), y)
        return {'loss': loss,} # will call loss.backward() on what we return exactly. 
    
    def training_epoch_end(self, outputs):
        if((self.current_epoch+1)%100==0):
            torch.save(self.state_dict(), 'W_transformer_flags1_V8')
        print("Epoch Loss:",torch.stack([x["loss"] for x in outputs]).mean().item())
    
    # Lightning disables gradients, puts model in eval mode, and does everything needed for validation.
    def validation_step(self, batch, batch_idx):
        X,target,padding_mask = batch
        src_mask = self.generate_square_subsequent_mask(S).cuda()
        X = X.permute(1,0,2)
        Y_predicted = self(X,src_mask,padding_mask)
        Y_predicted = Y_predicted.permute(1,0,2)
            
        mse_loss = nn.MSELoss(reduction='none')
        diff2 = mse_loss(Y_predicted, target)
        diff2[...,-1] *= 2 
        diff2= diff2.flatten()
        flags = (~padding_mask).unsqueeze(2).expand(-1,-1,self.n_features).float().flatten()
        loss = diff2*flags
        loss = torch.sum(loss) / torch.sum(flags)
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

# In[15]:



def main():
    # pl.seed_everything(42, workers=True) --> sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    time_all = time()

    model = TimeSeriesTransformer()
    early_stop_callback = EarlyStopping(monitor='val_loss',patience=5, verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint()
#     trainer = pl.Trainer(gpus=2,max_epochs=400, progress_bar_refresh_rate=50,accelerator ='ddp',
#                         callbacks=[early_stop_callback,checkpoint_callback]
#                          ,plugins=DDPPlugin(find_unused_parameters=False,check_val_every_n_epoch=2))
    
    
    trainer = pl.Trainer(gpus=2,max_epochs=500, progress_bar_refresh_rate=50, accelerator ='dp',
                        callbacks=[checkpoint_callback])
    trainer.fit(model,train_dataloader)
    print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
    print(checkpoint_callback.best_model_path)

if __name__ == '__main__':
    main()

