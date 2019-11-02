from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from forge.ethyr.torch.policy import linear, functional


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
      QK  = torch.softmax(QK / self.scale, dim=-2)
      QKV = torch.matmul(QK, V)
      return QKV

class MultiLinear(nn.Module):
   def __init__(self, xDim, yDim, n):
      super().__init__()
      self.fc = nn.ModuleList([
         nn.Linear(xDim, yDim) for _ in range(n)
      ])

   def forward(self, x):
      x = [fc(x) for fc in self.fc]
      x = torch.stack(x, -3)
      x = torch.max(x, -2)[0]
      return x

class Attention(nn.Module):
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

      attn = self.attention(Q, K, V)

      if self.flat:
         attn, _ = torch.max(attn, dim=-2)

      return attn

class FactorizedAttention(nn.Module):
   def __init__(self, xDim, yDim, h, flat=True):
      super().__init__()

      self.Q = MultiLinear(xDim, yDim, h)
      self.K = MultiLinear(xDim, yDim, h)
      self.V = MultiLinear(xDim, yDim, h)

      self.attention = ScaledDotProductAttention(yDim)
      self.flat = flat

   def forward(self, q):
      Q = self.Q(q)
      K = self.K(q)
      V = self.V(q)

      attn = self.attention(Q, K, V)

      if self.flat:
         attn, _ = torch.max(attn, dim=-2)

      return attn

class Attention2(nn.Module):
   def __init__(self, xDim, yDim):
      super().__init__()
      self.attn1 = Attention(xDim, yDim, flat=False)
      self.attn2 = Attention(xDim, yDim)
   
   def forward(self, x):
      x = self.attn1(x)
      x = self.attn2(x)
      return x

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
      k = self.key(k)
      v = self.val(v)
      x = torch.sum(k * v, -1)
      #x = functional.dot(k, v)
      return x
      return x.squeeze(-1)


class MiniAttend(nn.Module):
   def __init__(self, h, flat=True):
      super().__init__()
      self.fc1   = nn.Linear(h, h)
      self.fc2   = nn.Linear(h, h)
      self.flat = flat

   def forward(self, x, kv=None):
      if kv is not None:
         x = x * kv
   
      x = self.fc1(x)

      #New test
      x = torch.relu(x)
      x = self.fc2(x)

      if self.flat:
         #x = x.mean(-2)
         x, _ = torch.max(x, dim=-2)

      return x


class BareAttend(nn.Module):
   def __init__(self, h, flat=True):
      super().__init__()
      self.flat = flat

   def forward(self, x, kv=None):
      if kv is not None:
         x = x * kv
   
      if self.flat:
         x = torch.mean(x, dim=-2)

      return x


class AttnCat(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.attn = MiniAttend(h, flat=False)
      self.fc   = nn.Linear(h, 1)
      self.h = h

   def forward(self, key, vals):
      K, V = key, vals
      if len(K.shape) == 1:
         K = K.unsqueeze(0).unsqueeze(0).unsqueeze(0)
         V = V.unsqueeze(0).unsqueeze(0)

      #K = K.expand_as(V)
      #attn = functional.dot(K, V)

      #Yes, V, K. Otherwise all keys are equiv
      attn = self.attn(V, K)
      attn = self.fc(attn)
      attn = attn.squeeze(-1)

      #attn = self.attn(K, V).mean(-1)
      attn = attn.squeeze(0).squeeze(0)
      return attn

class BareMetal(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.h = h

   def forward(self, key, vals):
      K, V = key, vals
      if len(K.shape) == 1:
         K = K.unsqueeze(0)

      attn = (K*V).mean(-1)
      return attn


