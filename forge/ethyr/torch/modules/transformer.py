from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.scale = np.sqrt(h)

   def forward(self, Q, K, V):
      Kt = K.transpose(1, 2)
      QK = torch.matmul(Q, Kt)
      QK = torch.softmax(QK / self.scale, 2)
      QKV = torch.matmul(QK, V)
      return QKV

class MultiHeadAttention(nn.Module):
   def __init__(self, h, nHeads):
      super().__init__()
      self.headDim = int(h / nHeads)
      self.nHeads = nHeads

      self.Q = nn.Linear(h, h)
      self.K = nn.Linear(h, h)
      self.V = nn.Linear(h, h)

      self.attention = ScaledDotProductAttention(h)

   def reperm(self, x, src, perm, dst):
      return  x.view(*src).permute(*perm).reshape(*dst)

   def forward(self, x):
      batch, seq, _ = x.size()
      headDim, nHead = self.headDim, self.nHeads

      perm = (2, 0, 1, 3)
      src  = (batch, seq, nHead, headDim)
      dst  = (batch*nHead, seq, headDim)

      q = self.reperm(self.Q(x), src, perm, dst)
      k = self.reperm(self.K(x), src, perm, dst)
      v = self.reperm(self.V(x), src, perm, dst)

      perm = (1, 2, 0, 3)
      src  = (nHead, batch, seq, headDim)
      dst  = (batch, seq, headDim*nHead)

      attn = self.attention(q, k, v)
      #attn = q+k+v
      attn = self.reperm(attn, src, perm, dst)

      return attn

class Block(nn.Module):
   def __init__(self, h, nHeads, norm=False):
      super().__init__()
      self.attn = MultiHeadAttention(h, nHeads)
      #self.attn = ScaledDotProductAttention(h)

      self.fc   = nn.Linear(h, h)

      self.normalize = norm
      if norm:
         self.norm = nn.LayerNorm(h)

   def forward(self, x):
      #x = self.attn(x, x, x) + x
      x = self.attn(x) + x
      if self.normalize:
         x = self.norm(x)

      x = self.fc(x) + x
      if self.normalize:
         x = self.norm(x)

      return x

class Transformer(nn.Module):
   def __init__(self, h, nHeads, nLayers=1, flat=True):
      super().__init__()
      modules = [Block(h, nHeads) for i in range(nLayers)]
      self.attns = nn.ModuleList(modules)
      self.flat = True

   def forward(self, x):
      for attn in self.attns:
         x = attn(x)

      if self.flat:
         x = x.mean(1)

      return x

'''
class Transformer(nn.Module):
   def __init__(self, h, nHeads, nLayers=1, flat=True):
      super().__init__()

   def forward(self, x):
      return x.mean(1)
'''
