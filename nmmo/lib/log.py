from pdb import set_trace as T
from collections import defaultdict
from nmmo.lib import material
from copy import deepcopy
import os

import logging
import numpy as np
import json, pickle
import time

from nmmo.lib import utils

class Logger:
    def __init__(self):
        self.stats = defaultdict(list)

    def log(self, key, val):
        try:
            int_val = int(val)
        except TypeError as e:
            print(f'{val} must be int or float')
            raise e
        self.stats[key].append(val)
        return True

class MilestoneLogger(Logger):
    def __init__(self, log_file):
        super().__init__()
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=log_file, filemode='w')

    def log_min(self, key, val):
        if key in self.stats and val >= self.stats[key][-1]:
            return False

        self.log(key, val)
        return True

    def log_max(self, key, val):
        if key in self.stats and val <= self.stats[key][-1]:
            return False

        self.log(key, val)
        return True

class Quill:
   def __init__(self, config):
      self.config = config

      self.env    = Logger()
      self.player = Logger()
      self.event  = Logger()

      self.shared = {}

      if config.LOG_MILESTONES:
          self.milestone = MilestoneLogger(config.LOG_FILE)

   def register(self, key, fn):
       assert key not in self.shared, f'Log key {key} already exists'
       self.shared[key] = fn

   def log_env(self, key, val):
      self.env.log(key, val)

   def log_player(self, key, val):
      self.player.log(key, val)

   @property
   def packet(self):
      packet = {'Env':    self.env.stats,
              'Player': self.player.stats}

      if self.config.LOG_EVENTS:
          packet['Event'] = self.event.stats
      else:
          packet['Event'] = 'Unavailable: config.LOG_EVENTS = False'

      if self.config.LOG_MILESTONES:
          packet['Milestone'] = self.event.stats
      else:
          packet['Milestone'] = 'Unavailable: config.LOG_MILESTONES = False'

      return packet

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
 

