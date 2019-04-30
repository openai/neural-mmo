from pdb import set_trace as T
import numpy as np
import torch

def zeroGrads(ann):
   ind = 0
   for e in ann.parameters():
      if e.grad is None:
          continue
      shape = e.size()
      nParams = np.prod(shape)
      e.grad.data *= 0
      ind += nParams

def setParameters(ann, meanVec):
   ind = 0
   for e in ann.parameters():
      shape = e.size()
      nParams = np.prod(shape)
      e.data = torch.Tensor(np.array(meanVec[ind:ind+nParams]).reshape(*shape))
      ind += nParams

def setGrads(ann, grads):
   ind = 0
   for e in ann.parameters():
      shape = e.size()
      nParams = np.prod(shape)
      e.grad.data = torch.Tensor(np.array(grads[ind:ind+nParams]).reshape(*shape))
      ind += nParams

def getParameters(ann):
   ret = []
   for e in ann.parameters():
      ret += e.data.view(-1).numpy().tolist()
   return ret

def getGrads(ann, warn=True):
   ret = []
   for param, e in ann.named_parameters():
      if e.grad is None:
         if warn:
            print(str(param), ': GRADIENT NOT FOUND. Possible causes: (1) you have loaded a model with a different architecture. (2) This layer is not differentiable or not in use.')
         ret += np.zeros(e.shape).ravel().tolist()
      else:
         #nan = torch.sum(e.grad != e.grad) > 0:
         dat = e.grad.data.view(-1).numpy().tolist()
         if sum(np.isnan(dat))> 0 : 
            T()
         ret += dat
   return ret

