from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from neural_mmo.forge.ethyr.torch.policy import linear, functional

class DecomposedAttention(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.scale = h

   def forward(self, X, Q, K, V):
      X   = Q.transpose(-2, -1)
      QX  = torch.matmul(Q, X)
      KXT = torch.matmul(K, X).transpose(-2, -1)
      VX  = torch.matmul(V, X) / self.scale 
      
      KXTVX   = torch.matmul(KXT, VX)
      QXKXTVX = torch.matmul(QX, KXTVX)

      return QXKXTVX

class ScaledDotProductAttention(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.scale = np.sqrt(h)

   def forward(self, Q, K, V):
      Kt  = K.transpose(-2, -1)
      QK  = torch.matmul(Q, Kt)

      #Original is attending over hidden dims?
      #QK  = torch.softmax(QK / self.scale, dim=-2)
      QK    = torch.softmax(QK / self.scale, dim=-1)
      score = torch.sum(QK, -2)
      QKV = torch.matmul(QK, V)
      return QKV, score

class SelfAttention(nn.Module):
   def __init__(self, xDim, yDim, flat=True):
      super().__init__()

      self.Q = torch.nn.Linear(xDim, yDim)
      self.K = torch.nn.Linear(xDim, yDim)
      self.V = torch.nn.Linear(xDim, yDim)

      self.attention = ScaledDotProductAttention(yDim)
      self.flat = flat

   def forward(self, q):
      Q = self.Q(q)
      K = self.K(q)
      V = self.V(q)

      attn, scores = self.attention(Q, K, V)

      if self.flat:
         attn, _ = torch.max(attn, dim=-2)

      return attn, scores

class Attention(nn.Module):
   def __init__(self, xDim, yDim, flat=True):
      super().__init__()

      self.Q = torch.nn.Linear(xDim, yDim)
      self.K = torch.nn.Linear(xDim, yDim)
      self.V = torch.nn.Linear(xDim, yDim)

      self.attention = ScaledDotProductAttention(yDim)
      self.flat = flat

   def forward(self, q, v):
      Q = self.Q(q)
      K = self.K(v)
      V = self.V(v)

      #??? Doing 100x100 mat here, not good
      attn, scores = self.attention(Q, K, V)

      if self.flat:
         attn, _ = torch.max(attn, dim=-2)

      return attn, scores

class MaxReluBlock(nn.Module):
   def __init__(self, h, layers=2):
      super().__init__()
      self.net = linear.ReluBlock(
         h, layers, postRelu=False)
 
   def forward(self, x):
      x    = self.net(x)
      x, _ = torch.max(x, dim=-2)
      return x

class DotReluBlock(nn.Module):
   def __init__(self, h, layers=2):
      super().__init__()
      self.key = linear.ReluBlock(
         h, layers, postRelu=False)

      self.val = linear.ReluBlock(
         h, layers, postRelu=False)
   
   def forward(self, k, v):
      k = self.key(k).unsqueeze(-2)
      v = self.val(v)
      x = torch.sum(k * v, -1)
      return x
