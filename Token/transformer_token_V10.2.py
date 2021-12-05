#!/usr/bin/env python
# coding: utf-8

# # Required Imports

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
from sklearn.preprocessing import KBinsDiscretizer
sys.path.append('../DG/gan')


# # Loading Real Train Data

# In[2]:


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


# In[3]:


training_real = np.load('../data/google/data_train_reduced.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)

real_train_X,real_train_Y_labels,real_train_flags = remove_zero_datapoints(real_train_X,real_train_Y_labels,real_train_flags)

real_train_lengths = torch.sum(real_train_flags,1).long()

real_train_masks = real_train_flags == 0


# In[8]:


val_real = np.load('../data/google/data_train_val.npz')

real_val_X = torch.from_numpy(val_real['data_feature']).float() #[50000, 2500, 9]
real_val_Y = torch.from_numpy(val_real['data_attribute']) #[50000,4]
real_val_Y_labels = torch.argmax(real_val_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_val_flags = torch.from_numpy(val_real['data_gen_flag'])   # (50000, 2500)

real_val_X,real_val_Y_labels,real_val_flags = remove_zero_datapoints(real_val_X,real_val_Y_labels,real_val_flags)

real_val_masks = real_val_flags == 0

real_val_lengths = torch.sum(real_val_flags,1).long()


# # Ideas of V10

# - Feed Only data (real-valued), have two linear outputs, one is softmax as a classifier --> The last one is end token, another class --> We will discretize (Is it shifted or not)
# - We will need also to have the real-valued data outputted.
# - We need to see how to combine the nine features using mean.

# ## Important
# 
# - We don't shift the mask, so (last real value) should give (zero) and (end token).
# - We add the end token for the first zero, after all values
# - We also shift the nlp
# - Only 53 and 56 are greater than 400 timesteps for both train and val, respectively

# # Discretizaiton

# 1. Get Discretized Version (300-1)  --> Classes [0,298]
# 2. Add end token using the lengths array after the last time step --> EOS = 299
# 3. Shift one step, so the last timestep is expected to predict (end token).
# 4. We don't need to change the mask at all. Since everything is taken into consideration

# In[4]:


n_classes_disc = 300


# In[5]:


#Discretizing
est_train = KBinsDiscretizer(n_bins=n_classes_disc-1,encode='ordinal', strategy='uniform')

real_train_X_nlp = real_train_X.mean(2)
real_train_X_nlp = est_train.fit_transform(real_train_X_nlp.view(-1,1))
real_train_X_nlp = torch.Tensor(real_train_X_nlp).view(real_train_X.shape[0], real_train_X.shape[1])
# est_train.bin_edges_

#Adding End Token
for i,after_last in enumerate(real_train_lengths):
    real_train_X_nlp[i,after_last]  = n_classes_disc - 1   
    
B = real_train_X.size(0)
S = real_train_X.size(1)
E = real_train_X.size(2)

#Shifting NLP Target
Input_shifted = real_train_X_nlp[:,1:]
Zero_at_the_end = torch.zeros((B,1))
targets_nlp = torch.cat((Input_shifted,Zero_at_the_end),1)


# In[6]:


est_train.bin_edges_, torch.max(real_train_X.mean(2)), torch.max(real_train_X.sum(2))


# In[15]:


real_train_X_nlp = real_train_X_nlp[:,:400]
real_train_masks = real_train_masks[:,:400]
targets_nlp= targets_nlp[:,:400]

params_dataloader = {'shuffle': True,'num_workers':8 ,'batch_size':128} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_train_X_nlp.long(), targets_nlp.long(), real_train_masks)
train_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# In[8]:


#Discretizing
est_val = KBinsDiscretizer(n_bins=n_classes_disc-1,encode='ordinal', strategy='uniform')

real_val_X_nlp = real_val_X.sum(2)
real_val_X_nlp = est_val.fit_transform(real_val_X_nlp.view(-1,1))
real_val_X_nlp = torch.Tensor(real_val_X_nlp).view(real_val_X.shape[0], real_val_X.shape[1])
# est.bin_edges_

#Adding End Token
for i,after_last in enumerate(real_val_lengths):
    real_val_X_nlp[i,after_last]  = n_classes_disc - 1   
    
#Shifting Real Target
B = real_val_X.size(0)
S = real_val_X.size(1)
E = real_val_X.size(2)

Input_shifted = real_val_X[:,1:]
Zero_at_the_end = torch.zeros((B,1,E))
targets_real = torch.cat((Input_shifted,Zero_at_the_end),1) 

#Shifting NLP Target
Input_shifted = real_val_X_nlp[:,1:]
Zero_at_the_end = torch.zeros((B,1))
targets_nlp = torch.cat((Input_shifted,Zero_at_the_end),1)


# In[37]:


real_val_X = real_val_X[:,:400]
real_val_masks = real_val_masks[:,:400]
targets_real = targets_real[:,:400]
targets_nlp = targets_nlp[:,:400]


params_dataloader = {'shuffle': False,'num_workers':8 ,'batch_size':128} # No need to shuffle rn, they are all the same class
dataset = torch.utils.data.TensorDataset(real_val_X, targets_real, targets_nlp.long(), real_val_masks)
val_dataloader  = torch.utils.data.DataLoader(dataset, **params_dataloader)


# In[15]:


targets_nlp[200:240,:10]


# In[9]:


S = 400


# # TST

# In[10]:


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


# In[16]:


class TimeSeriesTransformer(pl.LightningModule):

    def __init__(self, n_features=9, d_model=256, n_heads=8, n_hidden=256, n_layers=8, dropout=0.0, S=400):
        super().__init__()
        self.model_type = 'Time Series Transformer Model'
        self.InputLinear = nn.Linear(n_features, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout, batch_first=True)
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
        output_nlp = self.OutputLinear(output)
        return   output_nlp 
    
    def training_step(self, batch, batch_idx):

        X,target_nlp,padding_mask = batch
        src_mask = self.generate_square_subsequent_mask(S).cuda()
        #print(X.shape,src_mask.shape)
        class_probs  = self(X,src_mask,padding_mask)
        loss_nlp = nn.CrossEntropyLoss()(class_probs.view(-1,n_classes_disc), target_nlp.flatten() )
         
        loss = loss_nlp 
        return {'loss': loss} # will call loss.backward() on what we return exactly. 
    
    def training_epoch_end(self, outputs):
        if((self.current_epoch+1)%100==0):
            torch.save(self.state_dict(), 'W_transformer_token_V10.2')
        print("Epoch Loss:",torch.stack([x["loss"] for x in outputs]).mean().item())
        
    # Lightning disables gradients, puts model in eval mode, and does everything needed for validation.
    def validation_step(self, batch, batch_idx):
        X,target,lengths,padding_mask = batch
        X = X.permute(1,0,2)
        padding_mask = torch.cat((torch.zeros((X.shape[1],2),dtype=torch.bool), (torch.ones((X.shape[1],398),dtype=torch.bool))),1).cuda()
        class_probs  = self(X,None,padding_mask)
        lengths -=1
        loss = nn.CrossEntropyLoss()(class_probs, lengths )
        
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

# In[17]:


d_model = 256
model = TimeSeriesTransformer() 

ck = torch.load('../lightning_logs/version_19/checkpoints/epoch=399-step=76799.ckpt')['state_dict']
model.load_state_dict(ck)
for param in model.parameters():
    param.requires_grad = False

model.OutputLinear  = nn.Linear(d_model,n_classes_disc) 
model.InputLinear  = nn.Embedding(n_classes_disc, d_model)  #(num_embeddings, embedding_dim)


# model.load_state_dict(torch.load('W_transformer_token_V10.1'))
# model.eval()
print()


# In[18]:


# RuntimeError: CUDA error: device-side assert triggered --> The problem it needs to be 0-399 not 1-400
def main():
    # pl.seed_everything(42, workers=True) --> sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    time_all = time()

    early_stop_callback = EarlyStopping(monitor='val_loss',patience=5, verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint()
#     trainer = pl.Trainer(gpus=2,max_epochs=400, progress_bar_refresh_rate=50,accelerator ='ddp',
#                         callbacks=[early_stop_callback,checkpoint_callback]
#                          ,plugins=DDPPlugin(find_unused_parameters=False,check_val_every_n_epoch=2))
    
    
    trainer = pl.Trainer(gpus=1,max_epochs=400, progress_bar_refresh_rate=50,check_val_every_n_epoch=3,
                        callbacks=[checkpoint_callback],)
    trainer.fit(model,train_dataloader)
    print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
    print(checkpoint_callback.best_model_path)

if __name__ == '__main__':
    main()


# In[37]:


gc.collect()


# In[ ]:





# In[ ]:





# In[ ]:




