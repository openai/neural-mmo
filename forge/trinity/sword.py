from pdb import set_trace as T
from collections import defaultdict
import numpy as np

from forge import trinity
from forge.ethyr.torch.param import setParameters, zeroGrads
from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout
from forge.ethyr.torch import param

def mems():
   import torch, gc
   import numpy as np
   size = 0
   for obj in gc.get_objects():
       try:
           if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
               #print(type(obj), obj.size())
               size += obj.nelement()*obj.element_size()
       except:
           pass
   print('Tensor Size: ', size)

class Sword:
   def __init__(self, config, args):
      self.config, self.args = config, args
      self.nANN, self.h = config.NPOP, config.HIDDEN
      self.anns  = [trinity.ANN(config)
            for i in range(self.nANN)]

      self.init, self.networksUsed = True, set()
      self.nRollouts, self.updateFreq = config.NROLLOUTS, config.UPDATEFREQ
      self.updates, self.rollouts = defaultdict(Rollout), {}
      self.grads, self.nGrads = None, 0
      self.blobs = []
      self.nUpdates = 0

   def backward(self):
      ents = self.rollouts.keys()
      anns = [self.anns[idx] for idx in self.networksUsed]

      reward, val, pg, valLoss, entropy = optim.backward(
            self.rollouts, valWeight=0.25, 
            entWeight=self.config.ENTROPY)

      self.blobs += [r.feather.blob for r in self.rollouts.values()]
      self.rollouts = {}
      self.nGrads = 0
      self.networksUsed = set()
      self.nUpdates += 1

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
      #if self.nGrads >= self.batch:
      if len(self.rollouts) >= self.nRollouts:
         self.backward()
         if self.nUpdates >= self.updateFreq:
            self.nUpdates = 0
            self.grads = dict((idx, param.getGrads(ann, warn=False)) 
                  for idx, ann in enumerate(self.anns))


   def decide(self, env, ent):
      reward, entID, annID = 0, ent.entID, ent.annID
      actions, outs, val = self.anns[annID](env, ent)
      self.collectStep(entID, outs, val, reward)
      self.updates[entID].feather.scrawl(
            env, ent, val, reward)
      return actions, float(val)
