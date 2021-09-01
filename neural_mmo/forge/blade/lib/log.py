from pdb import set_trace as T
from collections import defaultdict
from neural_mmo.forge.blade.lib import material
from copy import deepcopy
import os

import numpy as np
import json, pickle
import time
import ray

from neural_mmo.forge.blade.lib import utils
from neural_mmo.forge.blade.systems import visualizer
from neural_mmo.forge.blade.systems.visualizer import plot

class Quill:
   TRAINING     = -1
   LINE         = 0
   SCATTER      = 1
   HISTOGRAM    = 2
   GANTT        = 3
   STATS        = 4
   RADAR        = 5
   STACKED_AREA = 6

   PLOTS = {
      TRAINING:     plot.Training,
      LINE:         plot.Line,
      SCATTER:      plot.Scatter,
      HISTOGRAM:    plot.Histogram,
      GANTT:        plot.Gantt,
      STATS:        plot.Stats,
      RADAR:        plot.Radar,
      STACKED_AREA: plot.StackedArea,
   }

   def plot(idx):
      return Quill.PLOTS[idx] 

   def __init__(self):
      self.blobs = defaultdict(Blob)
      self.stats = defaultdict(list)

   def stat(self, key, val):
      self.stats[key].append(val)

   def register(self, key, tick, *plots):
      if key in self.blobs:
         blob = self.blobs[key]
      else:
         blob = Blob(*plots)
         self.blobs[key] = blob

      blob.tick = tick
      return blob

   @property
   def packet(self):
      logs = {key: blob.packet for key, blob in self.blobs.items()}
      return {'Log': logs, 'Stats': self.stats}

class Blob:
   def __init__(self, *args):
      self.tracks = defaultdict(Track)
      self.plots  = args
 
   def log(self, value, key=None):
      self.tracks[key].update(self.tick, value)

   @property
   def packet(self):
      return {key: (self.plots, track.packet) for key, track in self.tracks.items()}

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

class InkWell:
   def __init__(self):
      self.log   = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
      self.stats = defaultdict(list)

   def update(self, quill, realm=0):
      '''Realm param unused -- for future interactive expansions'''
      log, stats = quill['Log'], quill['Stats']
      for key, blob in log.items():
         for subkey, (plots, track) in blob.items():
            for time, vals in track.items():
               self.log[realm][key, tuple(plots)][subkey][time] += vals
      for key, stat in stats.items():
         self.stats[key] += stat

   @property
   def packet(self):
      return {'Log': utils.default_to_regular(self.log),
              'Stats': utils.default_to_regular(self.stats)}
 
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
 

