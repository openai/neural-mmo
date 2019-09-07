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

class BlobSummary:
   def __init__(self):
      self.nRollouts = 0
      self.nUpdates  = 0
      self.blobs     = []

   def merge(blobs):
      summary = BlobSummary()
      for blob in blobs:
         summary.nRollouts += blob.nRollouts
         summary.nUpdates  += blob.nUpdates
         summary.blobs     += blob.blobs

      return summary

#Agent logger
class Blob:
   def __init__(self, entID, annID): 
      self.unique = {Material.GRASS.value: 0,
                     Material.SCRUB.value: 0,
                     Material.FOREST.value: 0}
      self.counts = deepcopy(self.unique)
      self.lifetime = 0

      self.reward, self.ret       = None, []
      self.value, self.entropy    = None, []
      self.pg_loss, self.val_loss = []  , []

      self.entID = entID 
      self.annID = annID

   def update(self):
      self.lifetime += 1

class Quill:
   def __init__(self, config):
      self.config = config
      modeldir = config.MODELDIR

      self.time = time.time()
      self.dir = modeldir
      self.index = 0

      self.curUpdates = 0
      self.curRollouts = 0
      self.nUpdates = 0
      self.nRollouts = 0
      try:
         os.remove(modeldir + 'logs.p')
      except:
         pass
 
   def timestamp(self):
      cur = time.time()
      ret = cur - self.time
      self.time = cur
      return str(ret)

   def stats(self):
      updates  = 'Updates:  (Total) ' + str(self.nUpdates)
      rollouts = 'Rollouts: (Total) ' + str(self.nRollouts)

      padlen   = len(updates)
      updates  = updates.ljust(padlen)  
      rollouts = rollouts.ljust(padlen) 

      updates  += '  |  (Epoch) ' + str(self.curUpdates)
      rollouts += '  |  (Epoch) ' + str(self.curRollouts)

      return updates + '\n' + rollouts

   def scrawl(self, logs):
      #Collect experience information
      self.nUpdates     += logs.nUpdates
      self.nRollouts    += logs.nRollouts
      self.curUpdates   =  logs.nUpdates
      self.curRollouts  =  logs.nRollouts

      #Collect log update
      rewards = []
      self.index += 1
      for blob in logs.blobs:
         rewards.append(float(blob.lifetime))

      self.lifetime = np.mean(rewards)   

      if not self.config.SAVE_BLOBS:
         return
      
      blobRet = []
      for e in logs.blobs:
         if np.random.rand() < self.config.BLOB_FRAC:
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
 

