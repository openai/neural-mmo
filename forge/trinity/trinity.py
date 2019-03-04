class Trinity:
   def __init__(self, pantheon, god, sword):
      self.pantheon = pantheon
      self.god      = god
      self.sword    = sword

#Cluster/Master logic
class Pantheon:
   def __init__(self, args): pass
   def step(self, recvs): pass

#Environment logic
class God:
   def __init__(self, args, idx): pass
   def spawn(self): pass
   def send(self): pass
   def recv(self, pantheonUpdates): pass

#Agent logic
class Sword:
   def __init__(self, args): pass
   def sendUpdate(self): pass
   def recvUpdate(self, update): pass
   def collectRollout(self, entID, ent): pass
   def decide(self, entID, ent, stim): pass

