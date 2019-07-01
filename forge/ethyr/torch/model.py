from pdb import set_trace as T
import numpy as np
import torch
import time

from collections import defaultdict
from torch.nn.parameter import Parameter

from forge.ethyr.torch import save
from forge.ethyr.torch.optim import ManualAdam

class Model:
   '''Model manager class

   Convenience class wrapping saving/loading,
   model initialization, optimization, and logging.

   Args:
      ann: Model to optimize. Used to initialize weights.
      config: A Config specification
      args: Hook for additional user arguments
   '''
   def __init__(self, ann, config, args):
      self.saver = save.Saver(config.MODELDIR,
            'models', 'bests', resetTol=256)
      self.config, self.args = config, args

      self.init(ann)
      if self.config.LOAD or self.config.BEST:
         self.load(self.config.BEST)

   def init(self, ann):
      print('Initializing new model...')
      self.initModel(ann)

      self.opt = None
      if not self.config.TEST:
         self.opt = ManualAdam([self.params], lr=0.001, weight_decay=0.00001)

   #Initialize a new network
   def initModel(self, ann):
      self.models = ann(self.config).params()
      self.params = Parameter(torch.Tensor(np.array(self.models)))

   #Grads and clip
   def stepOpt(self, gradList):
      '''Clip the provided gradients and step the optimizer

      Args:
         gradList: a list of gradients
      '''
      grad = np.array(gradList)
      grad = np.mean(grad, 0)
      grad = np.clip(grad, -5, 5)

      gradAry = torch.Tensor(grad)
      self.opt.step(gradAry)

   def checkpoint(self, reward):
      '''Save the model to checkpoint

      Args:
         reward: Mean reward of the model
      '''
      if self.config.TEST:
         return
      self.saver.checkpoint(self.params, self.opt, reward)

   def load(self, best=False):
      '''Load a model from file

      Args:
         best (bool): Whether to load the best (True)
             or most recent (False) checkpoint
      '''
      print('Loading model...')
      epoch = self.saver.load(
            self.opt, self.params, best)

   @property
   def nParams(self):
      '''Print the number of model parameters'''
      nParams = len(self.model)
      print('#Params: ', str(nParams/1000), 'K')

   @property
   def model(self):
      '''Get model parameters

      Returns:
         a numpy array of model parameters
      '''
      return self.params.detach().numpy()

