from pdb import set_trace as T
import ray
import pickle
import time

from forge.blade.core.realm import Realm
from forge.trinity.timed import Timed, runtime, waittime

class Trinity(Timed):
   def __init__(self, pantheon, god, sword):
      super().__init__()
      self.pantheon = pantheon
      self.god      = god
      self.sword    = sword

   def init(self, config, args):
      self.base = self.pantheon(self, config, args)
      self.disciples = [self.base]

   @runtime
   def step(self):
      return self.base.step()

   @waittime
   def sync(self, rets):
      return ray.get(rets)

#Cluster/Master logic
class Pantheon(Timed):
   def __init__(self, trinity, config, args):
      super().__init__()
      self.disciples = [trinity.god.remote(trinity, config, args) 
            for _ in range(config.NGOD)]

   def distrib(self, packet):
      rets = []
      for god in self.disciples:
         rets.append(god.step.remote(packet))
      return rets

   def step(self, packet=None):
      rets = self.distrib(packet)
      rets = self.sync(rets)
      return rets

   @waittime
   def sync(self, rets):
      return ray.get(rets)

#Environment logic
class God(Timed):
   def __init__(self, trinity, config, args):
      super().__init__()
      self.disciples = [trinity.sword.remote(trinity, config, args, idx) 
            for idx in range(args.nRealm)]

   def distrib(self, packet=None):
      rets = []
      for sword in self.disciples:
         ret  = sword.step.remote(packet)
         rets.append(ret)
      return rets

   def step(self, packet=None):
      rets = self.distrib(packet)
      rets = self.sync(rets)
      return rets

   @waittime
   def sync(self, rets):
      rets = ray.get(rets)
      #Bottleneck is here. All of it. Register custom serial?
      #Maybe, maybe not. Need to look into how Rapid does it.
      #If we can just send large blocks of (numpy) activations or gradients
      #around, would be easier...
      #Another approach would be to decouple action index selection from action
      #processing. Doing this in a dynamic action graph is hard though...
      return rets
      rets = [pickle.loads(e) for e in rets]
   
#Agent logic
class Sword(Timed):
   def __init__(self, trinity, config, args, idx):
      super().__init__()
      self.disciples = [Realm(config, args, idx)]
      self.env = self.disciples[0]
      self.env.spawn = self.spawn

   def getEnv(self):
      return self.env

   @runtime
   def step(self, atns):
      rets = self.sync(atns)
      return rets

   @waittime
   def sync(self, atns):
      rets = self.env.step(atns)
      return rets

   def sendUpdate(self): pass
   def recvUpdate(self, update): pass
   def collectRollout(self, entID, ent): pass
   def decide(self, entID, ent, stim): pass

