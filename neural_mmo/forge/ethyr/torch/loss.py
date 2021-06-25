from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

#NaN grad errors here usually mean too small
#of a batch for good advantage estimation
#(or dummy fixed action selection)
def advantage(returns, val):
   '''Computes a mean centered advantage function
   using a value function baseline.

   Args:
      returns: Tensor of trajectory returns
      vals: Tensor of value function outpus

   Returns:
      advantage: Advantage estimate tensor
   '''
   A = returns - val
   adv = A
   adv = (A - A.mean())
   #adv = (A - A.mean()) / (1e-4 + A.std())
   #adv = (A - A.mean()) / (1e-8 + A.std())
   adv = adv.detach()
   return adv

def policyLoss(logProb, atn, adv):
   '''Policy gradient loss
   
   Args:
      logProb: Log probability tensor
      atn: Action index tensor
      adv: Advantage estimate tensor

   Returns:
      loss: Mean policy gradient loss
   '''
   pgLoss = -logProb.gather(1, atn.view(-1, 1))
   return (pgLoss * adv).mean()

def valueLoss(val, returns):
   '''Value function loss

   Args:
      val: Value tensor
      returns: Return tensor

   Returns:
      value: Mean value loss
   '''
   return (0.5 * (val - returns) **2).mean()

def entropyLoss(prob, logProb):
   '''Entropy computation
   
   Args:
      prob    : Probability tensor
      logProb : Log probability tensor
   
   Returns:
      entropy: Mean entropy 
   '''
   loss = (prob * logProb)
   loss[torch.isnan(loss)] = 0
   return loss.sum(1).mean()

#Assumes pi is already -inf padded
def PG(pi, atn, val, returns):
   '''Computes losses for the policy gradient algorithm
   
   Args:
      pi     : Logit tensor (network outputs, pre softmax)
      atn    : Index tensor of selected actions
      val    : Value tensor
      return : Return tensor 
   
   Returns:
      polLoss : Policy loss
      entLoss : Entropy loss
   '''
   prob = [F.softmax(e, dim=-1) for e in pi]
   logProb = [F.log_softmax(e, dim=-1) for e in pi]

   prob = torch.nn.utils.rnn.pad_sequence(prob, batch_first=True)
   logProb = torch.nn.utils.rnn.pad_sequence(logProb, batch_first=True)

   adv = advantage(returns, val)
   polLoss = policyLoss(logProb, atn, adv)
   entLoss = entropyLoss(prob, logProb)
   valLoss = valueLoss(val, returns)
   return polLoss, valLoss, entLoss

#Not currently used because Vanilla PG is equivalent with 1 update/minbatch
class PPO(nn.Module):
   def forward(self, pi, v, atn, returns):
      prob, logProb = F.softmax(pi, dim=1), F.log_softmax(pi, dim=1)
      atnProb = prob.gather(1, atn.view(-1, 1))

      #Compute advantage
      A = returns - v
      adv = (A - A.mean()) / (1e-4 + A.std())
      adv = adv.detach()

      #Clipped ratio loss
      prob    = F.softmax(pi, dim=1)
      probOld = F.softmax(piOld, dim=1)
      logProb = F.log_softmax(pi, dim=1)

      atnProb    = prob.gather(1, atn)
      atnProbOld = probOld.gather(1, atn)

      ratio = atnProb / (atnProbOld + 1e-6)
      surr1 = ratio*adv
      surr2 = torch.clamp(ratio, min=1. - self.clip, max=1. + self.clip) * adv 
      policyLoss = -torch.min(surr1, surr2)

      #Compute value loss
      valueLoss = (0.5 * (v - returns) **2).mean()

      #Compute entropy loss
      entropyLoss = (prob * logProb).sum(1).mean()

      return policyLoss, valueLoss, entropyLoss
