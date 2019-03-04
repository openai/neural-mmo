import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

#Print model size
def modelSize(net):
   params = 0
   for e in net.parameters():
      params += np.prod(e.size())
   params = int(params/1000)
   print("Network has ", params, "K params")

#Same padded (odd k)
def Conv2d(fIn, fOut, k, stride=1):
   pad = int((k-1)/2)
   return torch.nn.Conv2d(fIn, fOut, k, stride=stride, padding=pad)

def Pool(k, stride=1, pad=0):
   #pad = int((k-1)/2)
   return torch.nn.MaxPool2d(k, stride=stride, padding=pad)

def Relu():
   return torch.nn.ReLU()

class FCRelu(nn.Module):
   def __init__(self, xdim, ydim):
      super().__init__()
      self.fc = torch.nn.Linear(xdim, ydim)
      self.relu = Relu()

   def forward(self, x):
      x = self.fc(x)
      x = self.relu(x)
      return x

class ConvReluPool(nn.Module):
   def __init__(self, fIn, fOut, k, stride=1, pool=2):
      super().__init__()
      self.conv = Conv2d(fIn, fOut, k, stride)
      self.relu = Relu()
      self.pool = Pool(k)

   def forward(self, x):
      x = self.conv(x)
      x = self.relu(x)
      x = self.pool(x)
      return x

#ModuleList wrapper
def moduleList(module, *args, n=1):
   return nn.ModuleList([module(*args) for i in range(n)])

#Variable wrapper
def var(xNp, volatile=False, cuda=False):
   x = Variable(torch.from_numpy(xNp), volatile=volatile).float()
   if cuda:
      x = x.cuda()
   return x

#Full-network initialization wrapper
def initWeights(net, scheme='orthogonal'):
   print('Initializing weights. Warning: may overwrite sensitive bias parameters (e.g. batchnorm)')
   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            init.orthogonal(e)
      elif scheme == 'normal':
         init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         init.xavier_normal(e)

