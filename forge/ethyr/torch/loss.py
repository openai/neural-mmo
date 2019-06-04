from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

#One issue with doing .backward here is you don't
#get to mean and std norm
def advantage(returns, val):
   A = returns - val
   adv = A
   #Potentially problematic line
   adv = (A - A.mean()) / (1e-4 + A.std())
   adv = adv.detach()
   return adv

def policyLoss(logProb, atn, adv):
   pgLoss = -logProb.gather(1, atn.view(-1, 1))
   return (pgLoss * adv).mean()

def valueLoss(v, returns):
   return (0.5 * (v - returns) **2).mean()

def entropyLoss(prob, logProb):
   #logProb = logProb * (logProb != -float('inf')).float()
   loss = (prob * logProb)
   loss[torch.isnan(loss)] = 0
   #loss = loss[:, 0:1]
   #mask = ~torch.isnan(loss)
   #loss = torch.where(mask, loss, torch.zeros_like(loss))
   #loss = loss[mask]
   return loss.sum(1).mean()
   loss = loss.sum(1)
   loss = loss[loss!=0]
   loss = loss.mean()
   return loss

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

#Assumes pi is already -inf padded
def PG(pi, atn, val, returns):
   prob = [F.softmax(e, dim=-1) for e in pi]
   logProb = [F.log_softmax(e, dim=-1) for e in pi]

   prob = torch.nn.utils.rnn.pad_sequence(prob, batch_first=True)
   logProb = torch.nn.utils.rnn.pad_sequence(logProb, batch_first=True)

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
