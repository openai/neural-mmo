from pdb import set_trace as T
from collections import defaultdict
from forge.blade.lib.enums import Material
from forge.blade.lib import enums
from copy import deepcopy
import os

import numpy as np
import json, pickle
import time
import ray

#Static blob analytics
class InkWell:
   def unique(blobs):
      tiles = defaultdict(list)
      for blob in blobs:
          for t, v in blob.unique.items():
             tiles['unique_'+t.tex].append(v)
      return tiles

   def counts(blobs):
      tiles = defaultdict(list)
      for blob in blobs:
          for t, v in blob.counts.items():
             tiles['counts_'+t.tex].append(v)
      return tiles

   def explore(blobs):
      tiles = defaultdict(list)
      for blob in blobs:
          for t in blob.counts.keys():
             counts = blob.counts[t]
             unique = blob.unique[t]
             if counts != 0:
                tiles['explore_'+t.tex].append(unique / counts)
      return tiles

   def lifetime(blobs):
      return {'lifetime':[blob.lifetime for blob in blobs]}
 
   def reward(blobs):
      return {'reward':[blob.reward for blob in blobs]}
  
   def value(blobs):
      return {'value': [blob.value for blob in blobs]}

#Agent logger
class Blob:
   def __init__(self): 
      self.unique = {Material.GRASS.value: 0,
                     Material.SCRUB.value: 0,
                     Material.FOREST.value: 0}
      self.counts = deepcopy(self.unique)
      self.lifetime = 0

      self.reward, self.ret = [], []
      self.value, self.entropy= [], []
      self.pg_loss, self.val_loss = [], []

   def finish(self):
      self.lifetime = len(self.reward)
      self.reward   = np.sum(self.reward)
      self.value    = np.mean(self.value)

class Quill:
   def __init__(self, modeldir):
      self.time = time.time()
      self.dir = modeldir
      self.index = 0
      try:
         os.remove(modeldir + 'logs.p')
      except:
         pass
 
   def timestamp(self):
      cur = time.time()
      ret = cur - self.time
      self.time = cur
      return str(ret)

   def print(self):
      print(
            'Time: ', self.timestamp(), 
            ', Iter: ', str(self.index))

   def scrawl(self, logs):
      #Collect log update
      self.index += 1
      rewards, blobs = [], []
      for blobList in logs:
         blobs += blobList
         for blob in blobList:
            rewards.append(float(blob.lifetime))
            
      self.lifetime = np.mean(rewards)   
      blobRet = []
      for e in blobs:
          if np.random.rand() < 0.1:
              blobRet.append(e)
      self.save(blobRet)

   def latest(self):
      return self.lifetime

   def save(self, blobs):
      with open(self.dir + 'logs.p', 'ab') as f:
         pickle.dump(blobs, f)

   def scratch(self):
      pass

#Log wrapper and benchmarker
class Benchmarker:
   def __init__(self, logdir):
      self.benchmarks = {}

   def wrap(self, func):
      self.benchmarks[func] = Utils.BenchmarkTimer()
      def wrapped(*args):
         self.benchmarks[func].startRecord()
         ret = func(*args)
         self.benchmarks[func].stopRecord()
         return ret
      return wrapped

   def bench(self, tick):
      if tick % 100 == 0:
         for k, benchmark in self.benchmarks.items():
            bench = benchmark.benchmark()
            print(k.__func__.__name__, 'Tick: ', tick,
                  ', Benchmark: ', bench, ', FPS: ', 1/bench)
 

