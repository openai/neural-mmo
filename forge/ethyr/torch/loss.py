from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

def advantage(returns, val):
   A = returns - val
   adv = (A - A.mean()) / (1e-4 + A.std())
   adv = adv.detach()
   return adv

def policyLoss(logProb, atn, adv):
   pgLoss = -logProb.gather(1, atn.view(-1, 1))
   return (pgLoss * adv).mean()

def valueLoss(v, returns):
   return (0.5 * (v - returns) **2).mean()

def entropyLoss(prob, logProb):
   return (prob * logProb).sum(1).mean()

def pad(seq):
   seq = [e.view(-1) for e in seq]
   lens = [(len(e), idx) for idx, e in enumerate(seq)]
   lens, idx = zip(*sorted(lens, reverse=True))
   seq = np.array(seq)[np.array(idx)]

   seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)
   #seq = seq.squeeze(dim=1)
   idx = torch.tensor(idx).view(-1, 1).expand_as(seq)
   seq = seq.gather(0, idx)

   return seq

def PG(pi, atn, val, returns):
   prob = pad([F.softmax(e, dim=1) for e in pi])
   logProb = pad([F.log_softmax(e, dim=1) for e in pi])

   adv = advantage(returns, val)
   polLoss = policyLoss(logProb, atn, adv)
   entLoss = entropyLoss(prob, logProb)
   return polLoss, entLoss

#A bit stale--not currently used because
#Vanilla PG is equivalent with 1 update per minibatch
class PPO(nn.Module):
   def forward(self, pi, v, atn, returns):
      prob, logProb = F.softmax(pi, dim=1), F.log_softmax(pi, dim=1)
      atnProb = prob.gather(1, atn.view(-1, 1))

      #Compute advantage
      A = returns - v
      adv = (A - A.mean()) / (1e-4 + A.std())
      adv = adv.detach()

      #Clipped ratio loss
      prob, probOld, logProb = F.softmax(pi, dim=1), F.softmax(piOld, dim=1), F.log_softmax(pi, dim=1)
      atnProb, atnProbOld = prob.gather(1, atn), probOld.gather(1, atn)

      ratio = atnProb / (atnProbOld + 1e-6)
      surr1 = ratio*adv
      surr2 = torch.clamp(ratio, min=1. - self.clip, max=1. + self.clip) * adv 
      policyLoss = -torch.min(surr1, surr2)

      #Compute value loss
      valueLoss = (0.5 * (v - returns) **2).mean()

      #Compute entropy loss
      entropyLoss = (prob * logProb).sum(1).mean()

      return policyLoss, valueLoss, entropyLoss
