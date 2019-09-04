import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_complex_tensor import ComplexTensor
from torch.utils.data import Dataset, DataLoader


class ComplexLayer(nn.Module):
    '''
    This class wraps a pytorch layer and turns it into the equivalent 
    complex layer. So far it works for Linear, Conv and ConvTranspose
    
    TODO:   1. Code it for RNN layers.
    
    '''
    def __init__(self, Layer,kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bias = kwargs.get('bias',False)
        # turn the bias off so as to only do matrix multiplication 
        # if you leave the bias on, then the complex arithmetic does not 
        # work out correctly
        kwargs['bias'] = False
        self.f_re = Layer(**kwargs)
        self.f_im = Layer(**kwargs)
        self.b = None
        out_dim_keyNames = set(['out_channels', 'out_features'])
        self.outType = list(out_dim_keyNames.intersection(kwargs.keys()))[0]
        self.out_dim = kwargs[self.outType]
        if self.bias:
            b_r = np.random.randn(self.out_dim,1).astype('float32')
            b_i = np.random.randn(self.out_dim,1).astype('float32')
            z = b_r + 1j*b_i
            self.b = ComplexTensor(z)    

    def forward(self, x): 
        real = self.f_re(x.real) - self.f_im(x.imag)
        imaginary = self.f_re(x.imag) + self.f_im(x.real)
        if self.bias:
            if self.outType == 'out_channels':
                # expand the dims
                b_r = self.b.real.reshape(1,len(self.b),1,1)
                b_i = self.b.imag.reshape(1,len(self.b),1,1)
            else:
                b_r = self.b.real.reshape(len(self.b),)
                b_i = self.b.imag.reshape(len(self.b),)
            real = real + b_r
            imaginary = imaginary + b_i
        result = torch.cat([real, imaginary], dim=-2)
        result.__class__ = ComplexTensor
        return result
    
    def __call__(self,x):
        result = self.forward(x)
        return result

def crelu(T):
    '''
    complex relu. T is a complex tensor
    '''
    real = F.relu(T.real)
    imag = F.relu(T.imag)
    result = torch.cat([real, imag], dim=-2)
    result.__class__ = ComplexTensor
    return result


class Flatten(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,x):
        x = x.reshape(x.size(0),-1)
        return x


class CompFlatten(Flatten):

    def __init__(self,):
        super().__init__()
        self.fl = Flatten()

    def forward(self,z):
        x = self.fl(z.real)
        y = self.fl(z.imag)
        z = torch.cat([x, y], dim=-2)
        z.__class__ = ComplexTensor
        return z
        
class complexDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X,y = self.data[idx]
        X = X.astype('float32')
        y = torch.from_numpy(np.array(y))

        if len(X.shape) == 2:
            X = np.expand_dims(X,axis=0)
 
        if self.transform:
            X = self.transform(X)
        X =  ComplexTensor(X)
        return X,y    


