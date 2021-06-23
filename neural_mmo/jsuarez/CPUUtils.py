#Utilities for Neural CPUs to interact with the set of actions.

from pdb import set_trace as T
from sim.lib import Utils
from sim.modules import Skill
from sim.entity.NPC import NPC
from sim.action import Action

class ActionStats:
   numMoves = 4
   numAttacks = 2
   numEntities = 1 + len(Utils.terminalClasses(NPC))
   numSkills = len(Utils.terminalClasses(Skill.Skill))
   numActions = 2 + numSkills

class ActionSpec:
   def __init__(self):
      self.prev = None
      self.roots = [Action.ActionNode, Action.ActionLeaf]
      #self.roots = [Actions.ActionLeaf]
   
   def edges(self):
      ret = []
      blacklist = (Action.ActionLeaf, Action.Args)
      for root in self.roots:
         for e in root.__subclasses__():
            if e not in blacklist:
               ret += [e]
      return ret

   def leaves(self):
      return Action.ActionLeaf.__subclasses__()

class SaveManager():
   def __init__(self, root):
      self.tl, self.ta, self.vl, self.va = [], [], [], []
      self.root = root
      self.stateDict = None
      self.lock = False

   def update(self, net, tl, ta, vl, va):
      nan = np.isnan(sum([t.sum(e) for e in net.state_dict().values()]))
      if nan or self.lock:
         self.lock = True
         print('NaN in update. Locking. Call refresh() to reset')
         return

      if self.epoch() == 1 or vl < np.min(self.vl):
         self.stateDict = net.state_dict().copy()
         t.save(net.state_dict(), self.root+'weights')

      self.tl += [tl]; self.ta += [ta]
      self.vl += [vl]; self.va += [va]

      np.save(self.root + 'tl.npy', self.tl)
      np.save(self.root + 'ta.npy', self.ta)
      np.save(self.root + 'vl.npy', self.vl)
      np.save(self.root + 'va.npy', self.va)

   def load(self, net, raw=False):
      stateDict = t.load(self.root+'weights')
      self.stateDict = stateDict
      if not raw:
         net.load_state_dict(stateDict)
      self.tl = np.load(self.root + 'tl.npy').tolist()
      self.ta = np.load(self.root + 'ta.npy').tolist()
      self.vl = np.load(self.root + 'vl.npy').tolist()
      self.va = np.load(self.root + 'va.npy').tolist()

   def refresh(self, net):
      self.lock = False
      net.load_state_dict(self.stateDict)

   def epoch(self):
      return len(self.tl)+1

    
