from pdb import set_trace as T
import numpy as np
import torch
import time
from forge.blade.lib.utils import EDA

class Resetter:
   def __init__(self, resetTol):
      self.resetTicks, self.resetTol = 0, resetTol 
   def step(self, best=False):
      if best:
         self.resetTicks = 0
      elif self.resetTicks < self.resetTol:
         self.resetTicks += 1
      else:
         self.resetTicks = 0
         return True
      return False

class Saver:
   def __init__(self, nANN, root, savef, bestf, resetTol):
      self.bestf, self.savef = bestf, savef,
      self.root, self.extn = root, '.pth'
      self.nANN = nANN
      
      self.resetter = Resetter(resetTol)
      self.rewardAvg, self.best = EDA(), 0
      self.start, self.epoch = time.time(), 0
      self.resetTol = resetTol

   def save(self, params, opt, fname):
      data = {'param': params,
              'opt' : opt.state_dict(),
              'epoch': self.epoch}
      torch.save(data, self.root + fname + self.extn) 

   def checkpoint(self, params, opt, reward):
      self.save(params, opt, self.savef)
      best = reward > self.best
      if best: 
         self.best = reward
         self.save(params, opt, self.bestf)

      self.time  = time.time() - self.start
      self.start = time.time()
      self.reward = reward
      self.epoch += 1

      if self.epoch % 100 == 0:
         self.save(params, opt, 'model'+str(self.epoch))

      return self.resetter.step(best)

   def load(self, opt, param, best=False):
      fname = self.bestf if best else self.savef
      data = torch.load(self.root + fname + self.extn)
      param.data = data['param']
      if opt is not None:
         opt.load_state_dict(data['opt'])
      epoch = data['epoch']
      return epoch

   def print(self):
      print(
            'Tick: ', self.epoch,
            ', Time: ', str(self.time)[:5],
            ', Lifetime: ', str(self.reward)[:5],
            ', Best: ', str(self.best)[:5]) 

