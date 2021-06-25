import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from collections import defaultdict
from neural_mmo.forge.ethyr.torch import loss


class ManualAdam(optim.Adam):
   '''Adam wrapper that accepts gradient lists'''
   def step(self, grads):
      '''Takes an Adam step on the parameters using
      the user provided gradient list provided.
   
      Args:
         grads: A list of gradients
      '''
      grads = Variable(torch.Tensor(np.array(grads)))
      self.param_groups[0]['params'][0].grad = grads
      super().step()

class ManualSGD(optim.SGD):
   '''SGD wrapper that accepts gradient lists'''
   def step(self, grads):
      '''Takes an SGD step on the parameters using
      the user provided gradient list provided.
   
      Args:
         grads: A list of gradients
      '''
      grads = Variable(torch.Tensor(np.array(grads)))
      self.param_groups[0]['params'][0].grad = grads
      super().step()

def merge(rollouts):
   '''Merges all collected rollouts for batched
   compatibility with optim.backward'''
   n = 0
   outs = defaultdict(lambda: defaultdict(list))
   for rollout in rollouts.values():
      for idx in range(rollout.time):
         for out in rollout.actions[idx]:
            if len(out.atnLogits) == 1:
               continue
            outk = outs[out.atnArgKey]
            outk['atns'].append(out.atnLogits)
            outk['idxs'].append(out.atnIdx)
            outk['vals'].append(out.value)
            outk['rets'].append(out.returns)
            n += 1

   return outs, n


def backward(rollouts, config):
   '''Computes gradients from a list of rollouts

   Args:
      rolls: A list of rollouts
      valWeight (float): Scale to apply to the value loss
      entWeight (float): Scale to apply to the entropy bonus
      device (str): Hardware to run backward on

   Returns:
      reward: Mean reward achieved across rollouts
      val: Mean value function estimate across rollouts
      pg: Policy gradient loss
      valLoss: Value loss
      entropy: Entropy bonus      
   '''
   device = config.DEVICE
   outs, n = merge(rollouts)
   pgLoss, valLoss, entLoss = 0, 0, 0
   for k, out in outs.items():
      atns = out['atns']
      vals = torch.stack(out['vals'])
      idxs = torch.tensor(out['idxs']).to(device)
      rets = torch.tensor(out['rets']).view(-1, 1).to(device)

      l, v, e = loss.PG(atns, idxs, vals, rets)

      #Averaging results in no learning. Need to retune LR?
      pgLoss  += l# / n
      valLoss += v# / n
      entLoss += e# / n

   totLoss = (
         config.PG_WEIGHT*pgLoss + 
         config.VAL_WEIGHT*valLoss + 
         config.ENTROPY*entLoss)

   totLoss.backward(retain_graph=True)

   return pgLoss, valLoss, entLoss

