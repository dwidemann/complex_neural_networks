'''
Steps:

1. read in the data
2. build the model
3. train the model
4. test the model

TODO: Either use pytorch-lightning or pytorch-template for the repo. 
'''
import os, sys
import numpy as np
from utils.utils import ComplexLayer, crelu, complexDataset, CompFlatten
from data_readers import deepSigDataReaders as dr
from pytorch_complex_tensor import ComplexTensor
from torch.utils.data import Dataset, DataLoader
from torch.nn import Conv1d, Conv2d, Linear, ConvTranspose2d
import torch.nn as nn

#%% read in the data
data_path = '/data/DeepSig/RML2016.10a_dict.pkl'
data = dr.DeepSig_2016a(data_path=data_path)
train_data = data['train_data']

#%% right now DataLoader does not work with the ComplexTensor
# The output has dimension [bz,1,2,128]
# when it should be [bz,1,128] where each sample is a ComplexTensor

dataset = complexDataset(train_data)

'''
trainLoader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

I = iter(trainLoader)
X,y = I.next()
X.shape
'''

#%%
class SimpleCNN(nn.Module):

    def __init__(self, num_output=11):
        super(SimpleCNN, self).__init__()
        self.num_output = num_output
        dct = {'in_channels':1, 'out_channels':16, 'kernel_size':(3,), 'padding':1, 'bias':False}
        self.conv1 = ComplexLayer(Conv1d,dct)
        dct = {'in_channels':16, 'out_channels':32, 'kernel_size':(3,), 'padding':1,'bias':False}
        self.conv2 = ComplexLayer(Conv1d,dct)
        dct = {'in_channels':32, 'out_channels':64, 'kernel_size':(3,), 'padding':1,'bias':False}
        self.conv3 = ComplexLayer(Conv1d,dct)
        self.Flatten = CompFlatten()
        dct = {'in_features':64*128, 'out_features':num_output,'bias':True}
        self.fc1 = ComplexLayer(Linear,dct)

    def forward(self,x):
        x = crelu(self.conv1(x))
        x = crelu(self.conv2(x))
        x = crelu(self.conv3(x))
        x = self.Flatten(x)
        x = self.fc1(x)
        x = F.softmax(x.abs(),dim=1)
        return x

model = SimpleCNN(num_output=11)
x,y = dataset.__getitem__(10)
out = model(x)
print(out.shape)

#%% create the loss, optimizer and scheduler
loss = nn.CrossEntropyLoss()
l = loss(out,y)
lr = .001
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

#%%
num_epochs = 2
for epoch in range(num_epochs):
    perm = np.random.permutation(len(dataset.data))
    running_loss = 0.0
    for idx, i in enumerate(perm,1):
        x,y = dataset.__getitem__(i)
        optimizer.zero_grad()
        out = model(x)
        l = loss(out,y)
        l.backward()
        optimizer.step()    

        running_loss += l.item()
        if (idx) % 100 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, idx, running_loss/idx))
 
