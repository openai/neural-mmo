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

from forge.blade.core.market.new_visualizer import Middleman, Market, BokehServer

class Logger:                                                                 
   def __init__(self, middleman):                                             
      self.items     = 'reward lifetime value'.split()                              
      self.middleman = middleman                                              
      self.tick      = 0                                                      
                                                                              
   def update(self, lifetime_mean, reward_mean, value_mean,
              lifetime_std, reward_std, value_std):
      data = {}                                                               
      data['lifetime'] = lifetime_mean
      data['reward']   = reward_mean
      data['value']    = value_mean
      data['lifetime_std']  = lifetime_std
      data['reward_std']    = reward_std
      data['value_std']     = value_std
      data['tick']     = self.tick                                            
                                                                              
      self.tick += 1                                                          
      self.middleman.setData.remote(data)

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

      self.lifetime = []
      self.reward   = [] 
      self.value    = []

   def add(self, blobs):
      for blob in blobs:
         self.nRollouts += blob.nRollouts
         self.nUpdates  += blob.nUpdates

         self.lifetime += blob.lifetime
         self.reward   += blob.reward
         self.value    += blob.value

      return self

#Agent logger
class Blob:
   def __init__(self, entID, annID): 
      self.lifetime = 0
      self.reward   = [] 
      self.value    = []

      self.entID = entID 
      self.annID = annID

   def inputs(self, reward):
      if reward is not None:
         self.reward.append(reward)

   def outputs(self, value):
      self.value.append(value)
      self.lifetime += 1

   def finish(self):
      self.reward   = [np.mean(self.reward)]
      self.value    = [np.mean(self.value)]
      self.lifetime = [self.lifetime]

      self.nUpdates  = self.lifetime[0]
      self.nRollouts = 1

class Quill:
   def __init__(self, config):
      self.config = config
      modeldir = config.MODELDIR

      self.time = time.time()
      self.dir = modeldir

      self.curUpdates = 0
      self.curRollouts = 0
      self.nUpdates = 0
      self.nRollouts = 0
      try:
         os.remove(modeldir + 'logs.p')
      except:
         pass

      middleman   = Middleman.remote()                                        
      self.logger = Logger(middleman)                                         
      visualizer  = BokehServer.remote(middleman, self.logger.items)          

 
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

      self.value_mean    = np.mean(logs.value)
      self.reward_mean   = np.mean(logs.reward)
      self.lifetime_mean = np.mean(logs.lifetime)

      self.value_std    = np.std(logs.value)
      self.reward_std   = np.std(logs.reward)
      self.lifetime_std = np.std(logs.lifetime)

      print('Value Function: ', self.value_mean)
      self.logger.update(self.lifetime_mean, self.reward_mean, self.value_mean,
                         self.lifetime_std, self.reward_std, self.value_std)

      return self.stats(), self.lifetime_mean

   def latest(self):
      return self.lifetime_mean, self.reward_mean

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
 

