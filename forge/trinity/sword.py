from pdb import set_trace as T
from collections import defaultdict
import numpy as np

from forge import trinity
from forge.ethyr.torch.param import setParameters, zeroGrads
from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout

class Sword:
   def __init__(self, config, args):
      self.config, self.args = config, args
      self.nANN, self.h = config.NPOP, config.HIDDEN
      self.anns  = [trinity.ANN(config)
            for i in range(self.nANN)]

      self.init, self.nRollouts = True, 32
      self.networksUsed = set()
      self.updates, self.rollouts = defaultdict(Rollout), {}
      self.ents, self.rewards, self.grads = {}, [], None
      self.nGrads = 0

   def backward(self):
      ents = self.rollouts.keys()
      anns = [self.anns[idx] for idx in self.networksUsed]

      reward, val, grads, pg, valLoss, entropy = optim.backward(
            self.rollouts, anns, valWeight=0.25, 
            entWeight=self.config.ENTROPY)
      self.grads = dict((idx, grad) for idx, grad in
            zip(self.networksUsed, grads))

      self.blobs = [r.feather.blob for r in self.rollouts.values()]
      self.rollouts = {}
      self.nGrads = 0
      self.networksUsed = set()

   def sendGradUpdate(self):
      grads = self.grads
      self.grads = None
      return grads
 
   def sendLogUpdate(self):
      blobs = self.blobs
      self.blobs = []
      return blobs

   def sendUpdate(self):
      if self.grads is None:
          return None, None
      return self.sendGradUpdate(), self.sendLogUpdate()

   def recvUpdate(self, update):
      for idx, paramVec in enumerate(update):
         setParameters(self.anns[idx], paramVec)
         zeroGrads(self.anns[idx])

   def collectStep(self, entID, atnArgs, val, reward):
      if self.config.TEST:
          return
      self.updates[entID].step(atnArgs, val, reward)

   def collectRollout(self, entID, ent):
      assert entID not in self.rollouts
      rollout = self.updates[entID]
      rollout.finish()
      self.nGrads += rollout.lifespan
      self.rollouts[entID] = rollout
      del self.updates[entID]

      # assert ent.annID == (hash(entID) % self.nANN)
      self.networksUsed.add(ent.annID)

      #Two options: fixed number of gradients or rollouts
      #if len(self.rollouts) >= self.nRollouts:
      if self.nGrads >= 100*32:
         self.backward()

   def decide(self, ent, stim):
      reward, entID, annID = 0, ent.entID, ent.annID
      action, arguments, atnArgs, val = self.anns[annID](ent, stim)
      self.collectStep(entID, atnArgs, val, reward)
      self.updates[entID].feather.scrawl(
            stim, ent, val, reward)
      return action, arguments, float(val)
