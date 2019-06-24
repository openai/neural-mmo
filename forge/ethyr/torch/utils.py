from pdb import set_trace as T
import numpy as np

import torch
from torch import nn


def modelSize(net):
   '''Print model size'''
   params = 0
   for e in net.parameters():
      params += np.prod(e.size())
   params = int(params/1000)
   print("Network has ", params, "K params")

def ModuleList(module, *args, n=1):
   '''Repeat module n times'''
   return nn.ModuleList([module(*args) for i in range(n)])

#Full-network initialization wrapper
def initWeights(net, scheme='orthogonal'):
   '''Provides multiple weight initialization schemes

   Args:
      net: network to initialize
      scheme: otrhogonal, normal, or xavier
   '''
   print('Initializing weights. Warning: may overwrite sensitive bias parameters (e.g. batchnorm)')
   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            init.orthogonal(e)
      elif scheme == 'normal':
         init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         init.xavier_normal(e)

def pack(val):
   '''Pack values to tensor'''
   seq_lens   = torch.LongTensor(list(map(len, val)))
   seq_tensor = torch.zeros((len(val), seq_lens.max()))
   for idx, (seq, seqlen) in enumerate(zip(val, seq_lens)):
      seq_tensor[idx, :seqlen] = torch.Tensor(seq)

   #Todo: reintroduce sort
   #seq_lens, perm_idx = seq_lens.sort(0, descending=True)
   #seq_tensor = seq_tensor[perm_idx]

   return seq_tensor, seq_lens

#Be sure to unsort these
def unpack(vals, lens):
   '''Unpack value tensor using provided lens'''
   ret = []
   for idx, l in enumerate(lens):
      e = vals[idx, :l]
      ret.append(e)
   return ret

