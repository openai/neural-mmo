from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.scale = np.sqrt(h)

   def forward(self, Q, K, V):
      Kt = K.transpose(-2, -1)
      QK = torch.matmul(Q, Kt)
      QK = QK / self.scale
      #QK = torch.softmax(QK / self.scale, -1)
      #QK = torch.nn.functional.softmax(QK, dim=-2)
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

   def forward(self, q, kv=None):
      if kv is None:
         k = v = q
      else:
         k = v = kv

      assert k.shape == v.shape

      perm = (3, 0, 1, 2, 4)
      headDim, nHead = self.headDim, self.nHeads

      batch1, batch2, seq, _ = k.size()
      src  = (batch1, batch2, seq, nHead, headDim)
      dst  = (batch1*batch2*nHead, seq, headDim)

      k = self.reperm(self.K(k), src, perm, dst)
      v = self.reperm(self.V(v), src, perm, dst)

      batch1, batch2, seq, _ = q.size()
      src  = (batch1, batch2, seq, nHead, headDim)
      dst  = (batch1*batch2*nHead, seq, headDim)

      q = self.reperm(self.Q(q), src, perm, dst)

      perm = (1, 2, 0, 3)
      src  = (nHead, batch1*batch2, seq, headDim)
      dst  = (batch1, batch2, seq, headDim*nHead)

      attn = self.attention(q, k, v)
      attn = self.reperm(attn, src, perm, dst)

      return attn

class Block(nn.Module):
   def __init__(self, h, nHeads, norm=False):
      super().__init__()
      self.attn = MultiHeadAttention(h, nHeads)
      self.fc   = nn.Linear(h, h)

      self.normalize = norm
      if norm:
         self.norm = nn.LayerNorm(h)

   def forward(self, x, kv=None):
      x = self.attn(x, kv) + x
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
      self.flat = flat

   def forward(self, x, kv=None):
      for attn in self.attns:
         x = attn(x, kv)

      if self.flat:
         x = x.mean(-2)

      return x
