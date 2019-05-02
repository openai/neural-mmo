class Trinity:
   def __init__(self, pantheon, god, sword):
      self.pantheon = pantheon
      self.god      = god
      self.sword    = sword

   def init(self, config, args):
      return self.pantheon(self, config, args)

#Cluster/Master logic
class Pantheon:
   def __init__(self, trinity, config, args):
      self.gods = [trinity.god(trinity, config, args) 
            for _ in range(config.NGOD)]
   
   def step(self, packet):
      rets = []
      for god in self.gods:
         rets.append(god.run(packet))
      return rets

#Environment logic
class God:
   def __init__(self, trinity, config, args):
      self.swords = [trinity.sword(trinity, config, args) 
            for _ in range(config.NSWORD)]

   def step(self, packet=None):
      rets = []
      for sword in self.swords:
         rets.append(sword.run(packet))
      return rets

     
   def spawn(self): pass
   def send(self): pass
   def recv(self, pantheonUpdates): pass

#Agent logic
class Sword:
   def __init__(self, trinity, config, args):
      self.env = realm.VecEnvRealm(config, args)

   def sendUpdate(self): pass
   def recvUpdate(self, update): pass
   def collectRollout(self, entID, ent): pass
   def decide(self, entID, ent, stim): pass

