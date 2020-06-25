from pdb import set_trace as T

import time
from tqdm import tqdm
import numpy as np

import ray
import sys
import os


class Bar(tqdm):
   def __init__(self, position=0):
      lbar = '{desc}: {percentage:3.0f}%|'
      bar = '{bar}'
      rbar  = '| [{elapsed}, ' '{rate_fmt}{postfix}]'
      fmt = ''.join([lbar, bar, rbar])
      super().__init__(total=100, position=position, bar_format=fmt)

   def percent(self, val):
      self.update(val - self.n)

   def title(self, txt):
      self.set_description(txt)

   def set(self,):
      tags = 'a b c d'.split()
      i = np.random.randint(4)
      util = 100*np.random.rand()

      self.percent(util)
      self.title(tags[i])

@ray.remote
class Actor:
   def __init__(self):
      self.bar1 = Bar(position=0)
      self.bar2 = Bar(position=1)

   def step(self):
      time.sleep(0.5)

      self.bar1.set()
      self.bar2.set()

if __name__ == '__main__':
   ray.init()
   actor = Actor.remote()

   while True:
      sys.stdout.flush()
      os.system('clear')

      ray.get(actor.step.remote())

