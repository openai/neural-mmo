from pdb import set_trace as T
import numpy as np
import torch
import time

from neural_mmo.forge.blade.lib.utils import EDA
import os

class Resetter:
   '''Utility for model stalling that keeps track
   of the time since the model has improved

   Args:
      resetTol: ticks before resetting
   '''
   def __init__(self, resetTol):
      self.resetTicks, self.resetTol = 0, resetTol 

   def step(self, best=False):
      '''Step the resetter

      Args:
         best (bool): whether this is the best model
            performance thus far

      Returns:
         bool: whether the model should be reset to
            the previous best checkpoint
      '''
      if best:
         self.resetTicks = 0
      elif self.resetTicks < self.resetTol:
         self.resetTicks += 1
      else:
         self.resetTicks = 0
         return True
      return False

class Saver:
   '''Model save/load class
   
   Args:
      root: Path for save
      savef: Name of the save file
      bestf: Name of the best checkpoint file
      resetTol: How often to reset if training stalls
   '''
   def __init__(self, root, savef, bestf, resetTol):
      self.bestf, self.savef = bestf, savef,
      self.root, self.extn = root, '.pth'
      
      self.resetter = Resetter(resetTol)
      self.rewardAvg, self.best = EDA(), 0
      self.start, self.epoch = time.time(), 0
      self.resetTol = resetTol

   def save(self, params, opt, fname):
      data = {'param': params,
              'opt' : opt,
              'epoch': self.epoch}
      path = os.path.join(self.root, fname + self.extn)
      torch.save(data, path)

   def checkpoint(self, params, opt, lifetime):
      '''Save the model to file

      Args:
         params: Parameters to save
         opt: Optimizer to save
         fname: File to save to
      '''
      #self.save(params, opt, self.savef)
      best = lifetime > self.best
      if best: 
         self.best = lifetime
         self.save(params, opt, self.bestf)

      self.time     = time.time() - self.start
      self.start    = time.time()
      self.lifetime = lifetime
      self.epoch += 1

      if self.epoch % 100 == 0:
         self.save(params, opt, 'model'+str(self.epoch))

      return self.perf(), self.resetter.step(best)

   def load(self, opt, param, best=False):
      '''Load the model from file

      Args:
         opt: Optimizer to load
         params: Parameters to load 
         best: Whether to load the best or latest checkpoint

      Returns:
         epoch: The epoch of the loaded checkpoint
      '''
      fname = self.bestf if best else self.savef
      path  = os.path.join(self.root, fname) + self.extn
      data  = torch.load(path)
      param.data = data['param']
      if opt is not None:
         opt.load(data['opt'])
      epoch = data['epoch']
      return epoch

   def perf(self):
      '''Print stats for the latest epoch'''
      return ''.join([
            'Tick: ', str(self.epoch),
            ', Time: ', str(self.time)[:5],
            ', Lifetime: ', str(self.lifetime)[:5],
            ', Best: ', str(self.best)[:5]])

