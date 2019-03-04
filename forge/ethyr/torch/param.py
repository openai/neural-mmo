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

def getGrads(ann):
   ret = []
   for param, e in ann.named_parameters():
      try:
         ret += e.grad.data.view(-1).numpy().tolist()
      except:
         print('Gradient dimension mismatch. This usually means you have either (1) loaded a model with a different architecture or (2) have a layer for which gradients are not available (e.g. not differentiable or more commonly not being used)')
   return ret

