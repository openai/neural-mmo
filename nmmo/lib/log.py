from pdb import set_trace as T
from collections import defaultdict
from nmmo.lib import material
from copy import deepcopy
import os

import numpy as np
import json, pickle
import time
import ray

from nmmo.lib import utils

class Quill:
   def __init__(self):
      self.blobs = defaultdict(Blob)
      self.stats = defaultdict(list)

   def stat(self, key, val):
      self.stats[key].append(val)

   def register(self, key, tick, *plots):
      if key in self.blobs:
         blob = self.blobs[key]
      else:
         blob = Blob()
         self.blobs[key] = blob

      blob.tick = tick
      return blob

   @property
   def packet(self):
      logs = {key: blob.packet for key, blob in self.blobs.items()}
      return {'Log': logs, 'Stats': self.stats}

class Blob:
   def __init__(self):
      self.tracks = defaultdict(Track)
 
   def log(self, value, key=None):
      self.tracks[key].update(self.tick, value)

   @property
   def packet(self):
      return {key: track.packet for key, track in self.tracks.items()}

#Static blob analytics
class Track:
   def __init__(self):
      self.data = defaultdict(list)

   def update(self, tick, value):
      if type(value) != list:
         value = [value]

      self.data[tick] += value

   @property
   def packet(self):
      return self.data

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
 

