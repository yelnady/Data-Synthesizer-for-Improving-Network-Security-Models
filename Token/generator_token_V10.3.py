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
from sklearn.preprocessing import KBinsDiscretizer


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


def get_one_class(X,Y,nlp, flag, class_label): # (X, Y, and flag) are the whole dataset that is consisted of many classes, Y is NOT One-Hot Encoded
    indices_class_label = np.where(Y==class_label)
    X,Y,nlp,flag = X[indices_class_label], Y[indices_class_label], nlp[indices_class_label], flag[indices_class_label] 
    indices_non_zero = torch.nonzero(torch.sum(flag,1)-1).squeeze()
    return X[indices_non_zero], Y[indices_non_zero], nlp[indices_non_zero], flag[indices_non_zero]

def get_n_samples(X,Y,nlp,flag,n_samples):
    randomList = random.sample(range(0, Y.shape[0]), n_samples)
    return X[randomList], Y[randomList], nlp[randomList], flag[randomList]

# In real data, if flag sum is 1 --> Then no timestep at all. --> So we do remove those ones by converting them to zeros, then remove from the list
# In real data, there is no flag of length ZERO
def remove_zero_datapoints(X,Y,flag):
    indices_non_zero = torch.nonzero(torch.sum(flag,1)-1).squeeze()
    return X[indices_non_zero], Y[indices_non_zero], flag[indices_non_zero]


# In[45]:


training_real = np.load('../data/google/data_train_reduced.npz')

real_train_X = torch.from_numpy(training_real['data_feature']).float() #[50000, 2500, 9]
real_train_Y = torch.from_numpy(training_real['data_attribute']) #[50000,4]
real_train_Y_labels = torch.argmax(real_train_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
real_train_flags = torch.from_numpy(training_real['data_gen_flag'])   # (50000, 2500)

#------------------------------------------------------------------Loading One Class------------------------------------------------
real_train_X, real_train_Y_labels, real_train_flags= remove_zero_datapoints(real_train_X, real_train_Y_labels, real_train_flags)

# The pading mask need to be inverted 

real_train_masks = real_train_flags == 0 # True when padding, False when considering

real_train_lengths = torch.sum(real_train_flags,1).long()


# In[46]:


# test_real = np.load('../data/google/data_test_reduced.npz')

# real_test_X = torch.from_numpy(test_real['data_feature']).float() #[50000, 2500, 9]
# real_test_Y = torch.from_numpy(test_real['data_attribute']) #[50000,4]
# real_test_Y_labels = torch.argmax(real_test_Y,1) #[50000,]  returns a list of the class label, no one hot encoding any more
# real_test_flags = torch.from_numpy(test_real['data_gen_flag'])   # (50000, 2500)

# real_test_X,real_test_Y_labels,real_test_flags = remove_zero_datapoints(real_test_X,real_test_Y_labels,real_test_flags)

# real_test_masks = real_test_flags == 0

# real_test_lengths = torch.sum(real_test_flags,1).long()


# In[47]:


n_classes_disc = 100


# In[48]:


est_train = KBinsDiscretizer(n_bins=n_classes_disc-1,encode='ordinal', strategy='uniform')

real_train_X_nlp = real_train_X.mean(2)
real_train_X_nlp = est_train.fit_transform(real_train_X_nlp.view(-1,1))
real_train_X_nlp = torch.Tensor(real_train_X_nlp).view(real_train_X.shape[0], real_train_X.shape[1])
# est.bin_edges_

#Adding End Token
for i,after_last in enumerate(real_train_lengths):
    real_train_X_nlp[i,after_last]  = n_classes_disc - 2   
    
real_train_X_nlp = real_train_X_nlp.long()


# # PyTorch Transformer Model
# 
# - Later, we need to remove this from here and put in a separate folder

# In[49]:


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


# In[50]:


class TimeSeriesTransformer(nn.Module):

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
        output_real = self.OutputLinear(output)
        output_real = self.activation(output_real)
        return output_real 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


# In[51]:



model = TimeSeriesTransformer().to(device)

ck = torch.load('../lightning_logs/version_19/checkpoints/epoch=399-step=76799.ckpt')['state_dict']
model.load_state_dict(ck)

# model.load_state_dict(torch.load('W_transformer_token_V10.1_masked'))
model.eval()
print()


# # NLP Model

# In[52]:


n_classes_disc


# In[53]:


class TimeSeriesTransformerNLP(nn.Module):

    def __init__(self, n_features=9, d_model=256, n_heads=8, n_hidden=256, n_layers=8, dropout=0.0, S=400):
        super().__init__()
        self.model_type = 'Time Series Transformer Model'
        self.InputLinear =   nn.Linear(n_features, d_model) 
        
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        
        self.d_model = d_model
        self.n_features = n_features
        
        self.OutputLinear = nn.Sequential(nn.Linear(d_model,512) , nn.ReLU(), nn.Linear(512,n_classes_disc) )  # The output of the encoder is similar to the input of the encoder, both are (B,S,d_model)
        self.init_weights()
        self.activation = nn.Sigmoid() 
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(-1e6)).masked_fill(mask == 1, float(0.0))
        return mask 

    def init_weights(self):
        initrange = 0.1
        self.InputLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask,padding_mask):
        src = self.InputLinear(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src, src_mask,padding_mask)
        output_nlp = self.OutputLinear(output)
        return   output_nlp 


# In[54]:


model_nlp = TimeSeriesTransformerNLP().to(device)
model_nlp.load_state_dict(torch.load('W_transformer_token_V10.3'))

model_nlp.eval()
print()


# In[55]:


gc.collect()


# In[56]:


src_mask = model.generate_square_subsequent_mask(400)
x = model_nlp(real_train_X[200:220,:400].to(device),src_mask.to(device),real_train_masks[200:220,:400].to(device)).cpu()
F.softmax(x.squeeze()).argmax(2)[:,:40]


# In[44]:


real_train_X_nlp[200:220,:10]


# # Generating New Data

# In[80]:


################################# The following is the generating part #################################

# Returns: X (The data)
# Returns: masks (e.g. [False,Flase,True,True,True,....,True]), False is the actual Data

resulted_masks = []
generated_dataset_Y=[]
generated_dataset_X=[]

def generate_dataset(X,Y,X_nlp,masks,n_seed,n_samples,max_length):
    for n in range(n_samples):
        datapoint,y,nlp, mask = get_n_samples(X,Y,X_nlp,masks,n_samples=1) # The first 10 timesteps of just one sample
        datapoint = datapoint[:,:n_seed] 
        nlp = nlp[:,:n_seed]
        datapoint_len = torch.sum(~mask) #Flip and count, you will get the actual length to generate likewise
        mask = mask[:,:n_seed] 
        E = datapoint.size(2)
        S = datapoint.size(1)
        for t in range(max_length-n_seed): # Loop until 400 timesteps
            src_mask = model.generate_square_subsequent_mask(S)
            Y_predicted = model(datapoint.to(device),src_mask.to(device),mask.to(device)).cpu()
            nlp_predicted = model_nlp(datapoint.to(device),src_mask.to(device),mask.to(device)).cpu()
            one_new_timestep=Y_predicted[:,-1].unsqueeze(0)

            one_new_nlp = F.softmax(nlp_predicted.squeeze()).argmax(1)[-1].item()
            #print(F.softmax(nlp_predicted.squeeze()).argmax(1))
            if one_new_nlp == n_classes_disc-2:
#                 print(S, datapoint_len )
                datapoint = torch.cat((datapoint,torch.zeros((1,max_length-S,E))),1).cpu()# Pad remainings with zero
                mask =  torch.cat((mask,torch.full((1,max_length-S),True)),1)
        
                break
            
            datapoint = torch.cat((datapoint,one_new_timestep),1) # add the forecasted timestep
            mask = torch.cat((mask,torch.tensor([[False]])),1 )
            nlp = torch.cat((nlp,torch.LongTensor([[one_new_nlp]])),1)
            #print(F.softmax(nlp_predicted.squeeze()).argmax(1), "Last", one_new_nlp)
            #print("New:", nlp)
            S = datapoint.size(1)
            
        resulted_masks.append(mask.numpy())
        generated_dataset_X.append(datapoint.squeeze().detach().numpy())
        generated_dataset_Y.append(y.item())
        del mask
        del datapoint
        del one_new_timestep
        gc.collect(),torch.cuda.empty_cache()
        
        if (n%100==0):
            print('{}/{}'.format(n,n_samples))
        if ((n+1)%1000==0):
             np.savez('npz_transformer_token_V10.2_masked',X=generated_dataset_X,masks= resulted_masks,Y=generated_dataset_Y)

max_length = 400
n_seed = 2
# Padding Mask Fed here is the Mask where "False is Real Data", True is masked and ignore them


# In[81]:


gc.collect()
real_train_X0 ,real_train_Y_labels0 , real_train_X_nlp_0 ,padding_mask0= get_one_class(real_train_X ,real_train_Y_labels, real_train_X_nlp ,real_train_masks,0)
real_train_X1 ,real_train_Y_labels1 , real_train_X_nlp_1 ,padding_mask1= get_one_class(real_train_X ,real_train_Y_labels, real_train_X_nlp ,real_train_masks,1)
real_train_X2 ,real_train_Y_labels2 , real_train_X_nlp_2 ,padding_mask2= get_one_class(real_train_X ,real_train_Y_labels, real_train_X_nlp ,real_train_masks,2)
real_train_X3 ,real_train_Y_labels3 , real_train_X_nlp_3, padding_mask3= get_one_class(real_train_X ,real_train_Y_labels, real_train_X_nlp ,real_train_masks,3)


generate_dataset(real_train_X0 ,real_train_Y_labels0 , real_train_X_nlp_0, padding_mask0,n_seed=n_seed,n_samples=real_train_X0.size(0),max_length=max_length)
generate_dataset(real_train_X1 ,real_train_Y_labels1 , real_train_X_nlp_1, padding_mask1,n_seed=n_seed,n_samples=real_train_X1.size(0),max_length=max_length)
generate_dataset(real_train_X2 ,real_train_Y_labels2 , real_train_X_nlp_2, padding_mask2,n_seed=n_seed,n_samples=real_train_X2.size(0),max_length=max_length)
generate_dataset(real_train_X3 ,real_train_Y_labels3 , real_train_X_nlp_3, padding_mask3,n_seed=n_seed,n_samples=real_train_X3.size(0),max_length=max_length)


# In[ ]:


gc.collect()


# In[ ]:


# Uncomment to make sure everything is calculated correctly in softmax and argmax

# x = torch.rand((2,3))
# print(x)
# print(F.softmax(x).argmax(1))


# In[ ]:


# gc.collect()
# real_test_X0 ,real_test_Y_labels0 ,padding_mask0= get_one_class(real_test_X ,real_test_Y_labels ,real_test_masks,0)
# real_test_X1 ,real_test_Y_labels1 ,padding_mask1= get_one_class(real_test_X ,real_test_Y_labels ,real_test_masks,1)
# real_test_X2 ,real_test_Y_labels2 ,padding_mask2= get_one_class(real_test_X ,real_test_Y_labels ,real_test_masks,2)
# real_test_X3 ,real_test_Y_labels3 ,padding_mask3= get_one_class(real_test_X ,real_test_Y_labels ,real_test_masks,3)


# generate_dataset(real_test_X0 ,real_test_Y_labels0 ,padding_mask0,n_seed=n_seed,n_samples=real_test_X0.size(0),max_length=max_length)
# generate_dataset(real_test_X1 ,real_test_Y_labels1 ,padding_mask1,n_seed=n_seed,n_samples=real_test_X1.size(0),max_length=max_length)
# generate_dataset(real_test_X2 ,real_test_Y_labels2 ,padding_mask2,n_seed=n_seed,n_samples=real_test_X2.size(0),max_length=max_length)
# generate_dataset(real_test_X3 ,real_test_Y_labels3 ,padding_mask3,n_seed=n_seed,n_samples=real_test_X3.size(0),max_length=max_length)


# In[ ]:


np.savez('npz_transformer_token_V10.2',X=generated_dataset_X,masks= resulted_masks,Y=generated_dataset_Y)


# In[ ]:


import gc


# In[ ]:


gc.collect(),torch.cuda.empty_cache()


# In[ ]:


# What is the average values of all samples in variable length sequence

mean_of_samples = []
for x,length in zip(real_train_X, real_train_lengths):
    mean_of_samples.append(torch.sum(x)/(9*length)) # torch.Size([2500])
    
print(np.mean(mean_of_samples))


# In[ ]:


# torch.max(input) → Tensor
# Returns the maximum value of all elements in the input tensor.

torch.max(real_train_X.flatten())


# In[ ]:


torch.min(real_train_X.flatten())


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
plt.plot(real_train_X.flatten(), 'o', color='black');
plt.xscale("log")
ax.set_title('CDF - Class all')
ax.set_xlabel('The Sequence Length')


# In[ ]:


torch.mean(real_train_X,2).shape


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
plt.hist(torch.mean(real_train_X,2).flatten(), 'o', color='black');
plt.xscale("log")
ax.set_title('CDF - Class all')
ax.set_xlabel('The Sequence Length')


# In[ ]:


plt.hist([1,1,1,2])


# In[ ]:




