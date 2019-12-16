from pdb import set_trace as T
import numpy as np
import torch

def zeroGrads(ann):
   '''Zeros out gradients

   Args:
      ann: a model
   '''
   ind = 0
   for e in ann.parameters():
      if e.grad is None:
          continue
      shape = e.size()
      nParams = np.prod(shape)
      e.grad.data *= 0
      ind += nParams

def setParameters(ann, meanVec):
   '''Sets the parameters of the ann

   Args:
      meanVec: A list of parameters
   '''
   if meanVec is None:
      return

   ind = 0
   for e in ann.parameters():
      shape = e.size()
      nParams = np.prod(shape)
      data = np.array(meanVec[ind:ind+nParams]).reshape(*shape)
      e.data = torch.Tensor(data).to(e.device)
      ind += nParams

def setGrads(ann, grads):
   '''Sets the gradients of the ann
      
   Args:
      grads: A list of gradients
   '''
   ind = 0
   for e in ann.parameters():
      shape = e.size()
      nParams = np.prod(shape)
      e.grad.data = torch.Tensor(np.array(grads[ind:ind+nParams]).reshape(*shape))
      ind += nParams

def getParameters(ann):
   '''Get the parameters of the ann

   Args:
      ann: The model to get parameters from

   Returns:
      ret: A list of parameters
   '''
   ret = []
   for e in ann.parameters():
      ret += e.data.cpu().view(-1).numpy().tolist()
   return ret

def getGrads(ann, warn=True):
   '''Get the gradients of the ann

   Args:
      ann: The model to get gradients from
      warn (bool): Whether to warn when gradients are None.
         This is a common red flag that there is something
         wrong with the network or training

   Returns:
      ret: A list of gradients
   '''
   ret = []
   for param, e in ann.named_parameters():
      if e.grad is None:
         if warn:
            print(str(param), ': GRADIENT NOT FOUND. Possible causes: (1) you have loaded a model with a different architecture. (2) This layer is not differentiable or not in use.')
         ret += np.zeros(e.shape).ravel().tolist()
      else:
         #nan = torch.sum(e.grad != e.grad) > 0:
         dat = e.grad.data.cpu().view(-1).numpy().tolist()
         assert sum(np.isnan(dat)) == 0
         ret += dat
   return ret

