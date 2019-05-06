from pdb import set_trace as T
import ray, pickle
import time

from forge.blade.core import realm
from forge.trinity.timed import Timed, timed

class Trinity:
   def __init__(self, pantheon, god, sword):
      self.pantheon = pantheon
      self.god      = god
      self.sword    = sword

   def init(self, config, args):
      self.base = self.pantheon(self, config, args)

   def step(self):
      return self.base.distrib()

   def logs(self):
      return self.base.logs()

#Cluster/Master logic
class Pantheon(Timed):
   def __init__(self, trinity, config, args):
      super().__init__()
      self.disciples = [trinity.god.remote(trinity, config, args) 
            for _ in range(config.NGOD)]

   def step(self, packet):
      rets = []
      for god in self.disciples:
         rets.append(god.distrib.remote(packet))
      return ray.get(rets)

#Environment logic
class God(Timed):
   def __init__(self, trinity, config, args):
      super().__init__()
      self.disciples = [trinity.sword.remote(trinity, config, args) 
            for _ in range(args.nRealm)]

   def step(self, packet=None):
      rets = []
      for sword in self.disciples:
         ret  = sword.distrib.remote(packet)
         rets.append(ret)
      rets = ray.get(rets)
      rets = [pickle.loads(e) for e in rets]
      return rets
    
   def spawn(self): pass
   def send(self): pass
   def recv(self, pantheonUpdates): pass

#Agent logic
class Sword(Timed):
   def __init__(self, trinity, config, args):
      super().__init__()
      self.disciples = [realm.VecEnvRealm(config, args, idx=0)]
      self.env = self.disciples[0]

   def step(self, atns):
      return self.env.step(atns)

   def sendUpdate(self): pass
   def recvUpdate(self, update): pass
   def collectRollout(self, entID, ent): pass
   def decide(self, entID, ent, stim): pass

