import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from pdb import set_trace as T

from forge.ethyr import rollouts
from forge.ethyr.torch import loss

class ManualAdam(optim.Adam):
   def step(self, grads):
      grads = Variable(torch.Tensor(np.array(grads)))
      self.param_groups[0]['params'][0].grad = grads
      super().step()

class ManualSGD(optim.SGD):
   def step(self, grads):
      grads = Variable(torch.Tensor(np.array(grads)))
      self.param_groups[0]['params'][0].grad = grads
      super().step()

def backward(rolls, valWeight=0.5, entWeight=0):
   outs = rollouts.mergeRollouts(rolls.values())
   pg, entropy, attackentropy = 0, 0, 0
   for k, out in outs['action'].items():
      atns = out['atns']
      vals = torch.stack(out['vals'])
      idxs = torch.stack(out['idxs'])
      rets = torch.tensor(out['rets']).view(-1, 1)
      l, e = loss.PG(atns, idxs, vals, rets)
      pg += l
      entropy += e

   returns = torch.stack(outs['value'])
   values  = torch.tensor(outs['return']).view(-1, 1)
   valLoss = loss.valueLoss(values, returns)
   totLoss = pg + valWeight*valLoss + entWeight*entropy

   totLoss.backward()
   reward = np.mean(outs['return'])

   return reward, vals.mean(), pg, valLoss, entropy

