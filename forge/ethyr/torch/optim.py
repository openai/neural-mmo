import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from pdb import set_trace as T

from forge.ethyr import rollouts
from forge.ethyr.torch import loss
from forge.ethyr.torch import param

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

def backward(rolls, anns, valWeight=0.5, entWeight=0):
   atns, vals, rets = rollouts.mergeRollouts(rolls.values())
   returns = torch.tensor(rets).view(-1, 1).float()
   vals = torch.cat(vals)
   pg, entropy, attackentropy = 0, 0, 0
   for i, atnList in enumerate(atns):
      aArg, aArgIdx = list(zip(*atnList))
      aArgIdx = torch.stack(aArgIdx)
      l, e = loss.PG(aArg, aArgIdx, vals, returns)
      pg += l
      entropy += e

   valLoss = loss.valueLoss(vals, returns)
   totLoss = pg + valWeight*valLoss + entWeight*entropy

   totLoss.backward()
   grads = [param.getGrads(ann) for ann in anns]
   reward = np.mean(rets)

   return reward, vals.mean(), grads, pg, valLoss, entropy

