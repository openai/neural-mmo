from pdb import set_trace as T
import ray
import numpy as np
from sim.lib import Utils
from collections import defaultdict
from itertools import chain
from sim.lib.PriorityQueue import PriorityQueue
from scipy.stats import rankdata

class ES:
   def __init__(self, foo, hyperparams, test=False):
      #self.nPop = 2048
      self.nPop = 512
      self.nRollouts = 1
      self.minRollouts = 1
      self.nDims = 25248 + 16 + 96 + 6
      self.nNoise = 10000
      self.sigma = hyperparams['sigma']
      self.alpha = hyperparams['alpha']
      self.topP  = hyperparams['topP']
      self.tstep = hyperparams['step']
      #0.0020*51.2
      self.tick = 0
      self.test = test
      self.desciples= {}
      self.elite = defaultdict(list)
      #self.topk = PriorityQueue(20)
      self.priorities = []
      self.bench = Utils.BenchmarkTimer()

      #if not test:
      self.meanVec = self.sigma*np.random.randn(self.nDims)
      self.resetNoise()
      self.data = (self.meanVec, self.noise)
 
   def collectRollout(self, iden, ent):
      if self.test:
         return
      f, mut = (ent.timeAlive, ent.packet)
      mut = mut[0]
      self.elite[mut].append(f)

   def stepES(self):
      if self.tick % self.tstep != 0:
         return 
      meanOff, n = 0, 0
      noise = self.noise

      #Mean over rollouts
      elite = [(np.min(v), k) for k, v in self.elite.items() if len(v) >= self.minRollouts]
      if len(elite) == 0:
         return
      elite = sorted(elite, reverse=True)
      self.elite = defaultdict(list)
      Fs, mutations = list(zip(*elite))
      topP = int(self.topP * len(Fs))
      Fs, mutations = Fs[:topP], mutations[:topP]
 
      Fs = rankdata(Fs, method='dense')
      Fs = np.asarray(Fs)[:, np.newaxis]
      mutations = noise[mutations, :]

      if len(Fs) == 0:
         return
      #Weighted mean vec update
      meanOff = np.mean(Fs*mutations, 0)
      meanVec = self.meanVec
      self.meanVec = meanVec + self.alpha * meanOff

      self.data = self.meanVec, self.noise

   def getParams(self):
      return self.meanVec, self.priorities

   def setParams(self, meanVec, hyperparams):
      self.meanVec = meanVec
      self.data = (self.meanVec, self.shared[1])

   def print(self):
      print(sorted(self.priorities, reverse=True)[:20])

   @property
   def n(self):
      return len(self.desciples)

   def step(self):
      self.tick += 1
      self.cullDead()
      #self.priorities = self.topk.priorities()
      #self.priorities = self.priorities[:len(self.priorities)]
      elite = [np.min(v) for v in self.elite.values() if len(v) >= self.minRollouts]
      elite = sorted(elite, reverse=True)
      priorities = int(self.topP * len(elite))
      self.priorities = elite[:priorities]
      self.stepES()
      pass

   @property
   def shared(self):
      ret = self.data
      self.data = (None, None)
      return ret

   def cullDead(self):
      for k in list(self.desciples.keys()):
         if not self.desciples[k].alive:
            del self.desciples[k]

   def resetNoise(self):
      self.noiseInd = 0
      self.noiseIncInd = 0
      if self.test:
         self.noise = np.zeros((self.nNoise, self.nDims))
      else:
         self.noise = 0.1*np.random.randn(self.nNoise, self.nDims)

   def save(self):
      return self.meanVec

   def load(self, meanVec):
      self.meanVec = meanVec
 
   #Returns a seed to use for spawning
   def spawn(self):
      iden = Utils.uniqueKey(self.desciples)
     
      self.noiseIncInd += 1
      if self.noiseIncInd == self.nRollouts:
         self.noiseIncInd = 0
         self.noiseInd += 1

      if self.noiseInd == self.nNoise:
         self.noiseInd = 0
         self.noiseIncInd = 0
 
      packet = self.noiseInd
      return iden, [packet]
