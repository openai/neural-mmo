from pdb import set_trace as T
import numpy as np
import torch
import time

from collections import defaultdict
from torch.nn.parameter import Parameter

from forge.ethyr.torch import save
from forge.ethyr.torch.optim import ManualAdam, ManualSGD
from forge.ethyr.torch.param import getParameters 
from forge.blade.lib.log import Quill
from forge import trinity
from forge.trinity import Base

class Model:
   def __init__(self, config, args):
      self.saver = save.Saver(config.NPOP, config.MODELDIR, 
            'models', 'bests', resetTol=256)
      self.config, self.args = config, args

      self.init()
      if self.config.LOAD or self.config.BEST:
         self.load(self.config.BEST)

   def init(self):
      print('Initializing new model...')
      self.initModel()

      self.opt = None
      if not self.config.TEST:
         self.opt = ManualAdam([self.params], lr=0.001, weight_decay=0.00001)

   #Initialize a new network
   def initModel(self):
      self.models = trinity.ANN(self.config).params()
      self.params = Parameter(torch.Tensor(np.array(self.models)))

   #Grads and clip
   def stepOpt(self, gradList):
      grad = np.array(gradList)
      grad = np.mean(grad, 0)
      grad = np.clip(grad, -5, 5)

      gradAry = torch.Tensor(grad)
      self.opt.step(gradAry)

   def checkpoint(self, reward):
      if self.config.TEST:
         return
      self.saver.checkpoint(self.params, self.opt, reward)

   def load(self, best=False):
      print('Loading model...')
      epoch = self.saver.load(
            self.opt, self.params, best)

   @property
   def nParams(self):
      nParams = len(self.model)
      print('#Params: ', str(nParams/1000), 'K')
      
   @property
   def model(self):
      return self.params.detach().numpy()

class Pantheon(Base.Pantheon):
   def __init__(self, trinity, config, args):
      super().__init__(trinity, config, args)      
      self.start, self.tick, self.nANN = time.time(), 0, config.NPOP
      self.config, self.args = config, args
      self.net = Model(config, args)
      self.quill = Quill(config.MODELDIR)
      self.log = defaultdict(list)
      self.net.nParams

   @property 
   def model(self):
      return self.net.model

   def processRecvs(self, recvs):
      recvs, logs = list(zip(*recvs))

      #Write logs
      self.quill.scrawl(logs)
      self.tick += 1

      if not self.config.TEST:
         lifetime = self.quill.latest()
         self.net.stepOpt(recvs)
         self.net.checkpoint(lifetime)
         self.net.saver.print()
      else:
         self.quill.print()

      return self.model

   def run(self):
      model = self.model
      while True:
         recvs = self.step(model)
         model = self.processRecvs(recvs)
         
