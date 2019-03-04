import torch
from torch import nn
from torch.autograd import Variable

from copy import deepcopy
import numpy as np
from pdb import set_trace as T

def var(xNp, volatile=False, cuda=False):
   x = Variable(torch.from_numpy(xNp), volatile=volatile).float()
   if cuda:
      x = x.cuda()
   return x

class StimNet(nn.Module):
   def __init__(self, xdim, h, ydim):
      super().__init__()
      self.fc1 = torch.nn.Linear(xdim, h)
      self.fc2 = torch.nn.Linear(h, ydim)

   def forward(self, x):
      a = self.fc1(x)
      a = torch.nn.functional.relu(a)
      a = self.fc2(a)
      return a

def randomMutation(ann, sigma):
   annNew = deepcopy(ann)
   for e in annNew.parameters():
     e.data = e.data + torch.Tensor(sigma*np.random.randn(*e.size()))
   return annNew

def GA(fitness, generations, n, t, sigma, dims):
   P, F = [], []
   for g in range(generations):
      Pn, Fn = [], []
      for i in range(n):
         if g == 0:
            P.append(StimNet(*dims))
            F.append(fitness(P[-1]))
         elif i == 0:
            Pn.append(P[0])
            Fn.append(F[0])
         else:
            k = np.random.randint(0, t)
            Pn.append(randomMutation(P[k], sigma))
            Fn.append(fitness(Pn[-1]))
      #Sort dec by F
      if g > 0:
         inds = np.argsort(Fn)[::-1]
         F = np.asarray(Fn)[inds].tolist()
         P = np.asarray(Pn)[inds].tolist()
      print(F[0])
            
if __name__ == '__main__':
   generations = 100
   n = 1000
   t = 10
   sigma = 0.01
   dims = (847, 16, 6)

   def fitness(ann):
      inp = var(np.random.randn(dims[0]))
      out = ann(inp)
      loss = -torch.sum((1 - out)**2)
      return loss.data[0]

   ret = GA(fitness, generations, n, t, sigma, dims)
