from pdb import set_trace as T
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.distributions import Categorical

import numpy as np
import time

#Same padded (odd k)
def Conv2d(fIn, fOut, k, stride=1):
   pad = int((k-1)/2)
   return torch.nn.Conv2d(fIn, fOut, k, stride=stride, padding=pad)

class StimNet(nn.Module):
   def __init__(self, xdim, h, ydim):
      super().__init__()
      self.conv1 = Conv2d(8, int(h/2), 3, stride=2)
      self.conv2 = Conv2d(int(h/2), h, 3, stride=2)
      self.fc1 = torch.nn.Linear(5+4*4*h, h)
      self.fc2 = torch.nn.Linear(h, ydim)

   def forward(self, conv, flat):
      if len(conv.shape) == 3:
         conv = conv.view(1, *conv.shape)
         flat = flat.view(1, *flat.shape)
      x, batch = conv, conv.shape[0]
      x = torch.nn.functional.relu(self.conv1(x))
      x = torch.nn.functional.relu(self.conv2(x))
      x = x.view(batch, -1)

      x = torch.cat((x, flat), dim=1)
      x = torch.nn.functional.relu(self.fc1(x))
      x = self.fc2(x)

      pi = x.view(batch, -1)

      return pi

def classify(logits):
   #logits = logits + 0.15*torch.norm(logits)
   distribution = Categorical(F.softmax(logits, dim=1))
   atn = distribution.sample()
   return atn

class ANN(nn.Module):
   def __init__(self, xdim, h, ydim):
      super().__init__()
      self.stimNet = StimNet(xdim, 24, ydim)
      self.valNet = StimNet(xdim, 24, 1)
      #self.curNet  = CurNet(xdim, 24, ydim)
      self.conv, self.flat, self.ent, self.stim, self.idx = [], [], [], [], []

   def recv(self, conv, flat, ent, stim, idx):
      self.conv.append(conv)
      self.flat.append(flat)
      self.ent.append(ent)
      self.stim.append(stim)
      self.idx.append(idx)

   def send(self):
      conv = torch.stack(self.conv, dim=0)
      flat = torch.stack(self.flat, dim=0)

      pi, val, atn = [], [], []
      #for c, f in zip(conv, flat): 
      #    p, v, a = self.forward(c, f)
      #    pi.append(p)
      #    val.append(v)
      #     atn.append(a)
      pi, val, atn = self.forward(conv, flat)

      pi  = [e.view(1, -1) for e in pi]
      val = [e.view(1, -1) for e in val]
      atn = [e.view(1) for e in atn]

      ret = list(zip(pi, val, self.ent, self.stim, atn, self.idx))
      self.conv, self.flat, self.ent, self.stim, self.idx = [], [], [], [], []
      return ret

   def forward(self, conv, flat):
      pi   = self.stimNet(conv, flat)
      val  = self.valNet(conv, flat)
      atn  = classify(pi)
      #ri, li = self.curNet(ents, entID, atn, conv, flat)

      return pi, val, atn

if __name__ == '__main__':
   ann = ANN(1850, 32, 6)#.cuda()
   batch = 100

   conv = torch.rand(batch, 8, 15, 15)#.cuda()
   flat = torch.rand(batch, 5)#.cuda()

   while True:
      start = time.time()
      _ = ann(conv, flat)
      print(1.0 / (time.time() - start))

