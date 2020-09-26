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

from forge.blade.lib import utils
from forge.blade.systems import visualizer
from forge.blade.systems.visualizer import plot

class Quill:
   LINE         = 0
   SCATTER      = 1
   HISTOGRAM    = 2
   GANTT        = 3
   STATS        = 4
   RADAR        = 5
   STACKED_AREA = 6

   PLOTS = {
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

   def __init__(self, realm, tick):
      self.blobs = defaultdict(Blob)
      self.realm = realm #Game map index
      self.tick  = tick  #Current game tick

   def register(self, key, *plots):
      if key in self.blobs:
         blob = self.blobs[key]
      else:
         blob = Blob(*plots)
         self.blobs[key] = blob

      blob.tick = self.tick
      return blob

   @property
   def packet(self):
      return {key: blob.packet for key, blob in self.blobs.items()}

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

'''
flat = []
for key, val in sorted(self.data.items()):
   flat += val

if Quill.LINE in plots:
   data['line']      = flat
if Quill.SCATTER in plots:
   data['scatter']   = self.data
if Quill.HISTOGRAM in plots:
   data['histogram'] = flat
'''

class InkWell:
   def __init__(self):
      self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

   def update(self, packets):
      for realm, quill in enumerate(packets):
         for key, blob in quill.items():
            for subkey, (plots, track) in blob.items():
               for time, vals in track.items():
                  self.data[realm][key, tuple(plots)][subkey][time] += vals

   @property
   def packet(self):
      return utils.default_to_regular(self.data)
 
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
 

