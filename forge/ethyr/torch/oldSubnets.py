#Previous networks tried. Fully connected nets are much
#easier to get working than conv nets.

from pdb import set_trace as T
import torch
from torch import nn
from torch.nn import functional as F
from forge.ethyr.torch import utils as tu

class Attention(nn.Module):
   def __init__(self, xDim, yDim):
      super().__init__()
      self.fc = torch.nn.Linear(2*xDim, yDim)

   #Compute all normalized scores
   def score(args, x, normalize=True, scale=False):

      scores = torch.matmul(args, x.transpose(0, 1))

      if scale:
         scores = scores / (32**0.5)

      if normalize:
         scores = Attention.normalize(scores)
      return scores.view(1, -1)

   #Normalize exp
   def normalize(x):
      b = x.max()
      y = torch.exp(x - b)
      return y / y.sum()

   def attend(self, args, scores):
      attn = args * scores
      return torch.sum(attn, dim=0).view(1, -1)

   def forward(self, args, x, normalize=True):
      scores = Attention.score(args, x, normalize)
      scores = self.attend(args, scores)
      scores = torch.cat((scores, x), dim=1)
      scores = torch.nn.functional.tanh(self.fc(scores))
      return scores

class AttnCat(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.fc1 = torch.nn.Linear(2*h, h)
      self.fc2 = torch.nn.Linear(h, 1)
      self.h = h

   def forward(self, x, args):
      n = args.shape[0]
      x = x.expand(n, self.h)
      xargs = torch.cat((x, args), dim=1)

      x = F.relu(self.fc1(xargs))
      x = self.fc2(x)
      return x.view(1, -1)

class AtnNet(nn.Module):
   def __init__(self, h, nattn):
      super().__init__()
      self.fc1 = torch.nn.Linear(h, nattn)

   def forward(self, stim, args):
      atn = self.fc1(stim)
      return atn

class ArgNet(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.attn = AttnCat(h)

   #Arguments: stim, action/argument embedding
   def forward(self, key, atn, args):
      atn = atn.expand_as(args)
      vals = torch.cat((atn, args), 1)
      arg = self.attn(key, vals)
      argIdx = classify(arg)
      return argIdx, argd

class Env(nn.Module):
   def __init__(self,  config):
      super().__init__()
      h = config.HIDDEN
      entDim = 12 + 255
      self.fc1  = torch.nn.Linear(entDim+1800+2*h, h)
      self.embed = torch.nn.Embedding(7, 7)
      self.ent1 = torch.nn.Linear(entDim, 2*h)

   def forward(self, conv, flat, ents):
      tiles, nents = conv[0], conv[1]
      tiles = self.embed(tiles.view(-1).long()).view(-1)
      nents = nents.view(-1)
      conv = torch.cat((tiles, nents)) 
      ents = self.ent1(ents)
      ents, _ = torch.max(ents, 0)
      x = torch.cat((conv.view(-1), flat, ents)).view(1, -1)
      x = torch.nn.functional.relu(self.fc1(x))
      return x

class Full(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.conv1 = tu.Conv2d(8, int(h/2), 3, stride=2)
      self.conv2 = tu.Conv2d(int(h/2), h, 3, stride=2)
      self.fc1 = torch.nn.Linear(6+4*4*h + h, h)
      #self.fc1 = torch.nn.Linear(5+4*4*h, h)
      self.ent1 = torch.nn.Linear(6, h)

   def forward(self, conv, flat, ents):
      if len(conv.shape) == 3:
         conv = conv.view(1, *conv.shape)
         flat = flat.view(1, *flat.shape)

      x, batch = conv, conv.shape[0]
      x = torch.nn.functional.relu(self.conv1(x))
      x = torch.nn.functional.relu(self.conv2(x))
      x = x.view(batch, -1)

      ents = self.ent1(ents)
      ents, _ = torch.max(ents, 0)
      ents = ents.view(batch, -1)

      #x = torch.cat((x, flat), dim=1)
      x = torch.cat((x, flat, ents), dim=1)

      x = torch.nn.functional.relu(self.fc1(x))
      return x

class FC1(nn.Module):
   def __init__(self, h):
      super().__init__()
      #h = config.HIDDEN
      self.fc1 = torch.nn.Linear(12+1800, h)

   def forward(self, conv, flat, ents):
      x = torch.cat((conv.view(-1), flat)).view(1, -1)
      x = torch.nn.functional.relu(self.fc1(x))
      return x

class FC2(nn.Module):
   def __init__(self,  config):
      super().__init__()
      h = config.HIDDEN
      self.fc1  = torch.nn.Linear(12+1800+2*h, h)
      self.ent1 = torch.nn.Linear(12, 2*h)
      self.embed = torch.nn.Embedding(7, 7)

   def forward(self, conv, flat, ents):
      tiles, nents = conv[0], conv[1]
      tiles = self.embed(tiles.view(-1).long()).view(-1)
      nents = nents.view(-1)
      conv = torch.cat((tiles, nents)) 
      ents = self.ent1(ents)
      ents, _ = torch.max(ents, 0)
      x = torch.cat((conv.view(-1), flat, ents)).view(1, -1)
      x = torch.nn.functional.relu(self.fc1(x))
      return x

#Use this one. Nice embedding property.
#They all work pretty well
class FC3(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.conv1 = tu.Conv2d(8, 8, 1)
      self.fc1   = tu.FCRelu(5+1800, h)

   def forward(self, conv, flat, ents):
      conv = conv.view(1, *conv.shape)
      x = self.conv1(conv)
      x = torch.cat((x.view(-1), flat)).view(1, -1)
      x = self.fc1(x)
      return x

class FC4(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.conv1 = tu.Conv2d(8, 4, 1)
      self.fc1   = tu.FCRelu(5+900, h)

   def forward(self, conv, flat, ents):
      conv = conv.view(1, *conv.shape)
      x = self.conv1(conv)
      x = torch.cat((x.view(-1), flat)).view(1, -1)
      x = self.fc1(x)
      return x

class FCEnt(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.fc1 = torch.nn.Linear(5+1800+h, h)
      self.ent1 = torch.nn.Linear(5, h)

   def forward(self, conv, flat, ents):
      ents = self.ent1(ents)
      ents, _ = torch.max(ents, 0)
      ents = ents.view(-1)

      x = torch.cat((conv.view(-1), flat, ents)).view(1, -1)
      x = torch.nn.functional.relu(self.fc1(x))
      return x

class FCAttention(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.fc1 = torch.nn.Linear(5+1800+h, h)
      self.ent1 = torch.nn.Linear(5, h)
      self.attn = Attention(h, h)

   def forward(self, conv, flat, ents):
      ents = self.ent1(ents)
      T()
      ents = self.attn(ents)
      ents = ents.view(-1)

      x = torch.cat((conv.view(-1), flat, ents)).view(1, -1)
      x = torch.nn.functional.relu(self.fc1(x))
      return x


class CNN1(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.conv1 = tu.Conv2d(8, h, 5, stride=3)
      self.conv2 = tu.Conv2d(h, h, 5, stride=3)
      self.fc1 = torch.nn.Linear(4*h+5, h)

   def forward(self, conv, flat, ents):
      x = conv.view(1, *conv.shape)
      x = torch.nn.functional.relu(self.conv1(x))
      x = torch.nn.functional.relu(self.conv2(x))
      x = x.view(-1)

      x = torch.cat((x, flat))
      x = torch.nn.functional.relu(self.fc1(x))
      return x.view(1, -1)

class CNN2(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.conv1 = tu.Conv2d(8, h, 3, stride=2)
      self.conv2 = tu.Conv2d(h, h, 3, stride=2)
      self.fc1 = torch.nn.Linear(16*h+5, h)

   def forward(self, conv, flat, ents):
      x = conv.view(1, *conv.shape)
      x = torch.nn.functional.relu(self.conv1(x))
      x = torch.nn.functional.relu(self.conv2(x))
      x = x.view(-1)

      x = torch.cat((x, flat))
      x = torch.nn.functional.relu(self.fc1(x))
      return x.view(1, -1)

class CNN3(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.conv1 = tu.ConvReluPool(8, h, 5)
      self.conv2 = tu.ConvReluPool(h, h//2, 5)
      self.fc1 = torch.nn.Linear(h//2*7*7+5, h)

   def forward(self, conv, flat, ents):
      x = conv.view(1, *conv.shape)
      x = torch.nn.functional.relu(self.conv1(x))
      x = torch.nn.functional.relu(self.conv2(x))
      x = x.view(-1)

      x = torch.cat((x, flat))
      x = torch.nn.functional.relu(self.fc1(x))
      return x.view(1, -1)


