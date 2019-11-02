from pdb import set_trace as T
import numpy as np

from forge.blade.lib import utils, enums
from forge.blade.lib.utils import staticproperty
from forge.blade.io.action.node import Node, NodeType
from forge.blade.systems import combat

#ActionRoot
class Action(Node):
   nodeType = NodeType.SELECTION

   @staticproperty
   def edges():
      #return [Move, Attack, Exchange, Skill]
      return [Move, Attack]

   @staticproperty
   def n():
      return len(Action.actions)

   def args(stim, entity, config):
      return Static.edges

   #Called upon module import (see bottom of file)
   #Sets up serialization domain
   def hook():
      actions = Action.flat()
      Action.actions = actions

      for idx, atn in enumerate(actions):
         atn.serial = tuple([idx])
         atn.idx = idx 

   def flat(root=None):
      if root is None:
         root = Action

      rets = [root]
      if root.nodeType is NodeType.SELECTION:
         for edge in root.edges:
            rets += Action.flat(edge)

      return rets

class Move(Node):
   nodeType = NodeType.SELECTION
   def call(world, entity, rDelta, cDelta):
      r, c = entity.base.pos
      entity.history._lastPos = (r, c)
      rNew, cNew = r+rDelta, c+cDelta
      if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
         return
      if not utils.inBounds(rNew, cNew, world.shape):
         return
      if entity.status.freeze > 0:
         return

      entity.base.r.update(rNew)
      entity.base.c.update(cNew)
      entID = entity.entID
      
      r, c = entity.history.lastPos
      world.env.tiles[r, c].delEnt(entID)

      r, c = entity.base.pos
      world.env.tiles[r, c].addEnt(entID, entity)

   @staticproperty
   def edges():
      return [North, South, East, West]

   def args(stim, entity, config):
      return Move.edges

   @staticproperty
   def leaf():
      return True

#Todo: kill args on these.
class North(Node):
   priority = 0
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, -1, 0)

   def args(stim, entity, config):
      return [None]

class South(Node):
   priority = 0
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 1, 0)

   def args(stim, entity, config):
      return [None]

class East(Node):
   priority = 0
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 0, 1)

   def args(stim, entity, config):
      return [None]

class West(Node):
   priority = 0
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 0, -1)

   def args(stim, entity, config):
      return [None]


class Attack(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def n():
      return 3

   @staticproperty
   def edges():
      return [Melee, Range, Mage]

   @staticproperty
   def leaf():
      return True

   def inRange(entity, stim, config, N):
      R, C = stim.shape
      R, C = R//2, C//2

      rets = set([entity])
      for r in range(R-N, R+N+1):
         for c in range(C-N, C+N+1):
            for e in stim[r, c].ents.values():
               minWilderness = min(entity.status.wilderness.val, e.status.wilderness.val)

               selfLevel = combat.level(entity.skills)
               targLevel = combat.level(e.skills)
               #if abs(selfLevel - targLevel) <= minWilderness:
               rets.add(e)
      rets = list(rets)
      return rets

   def l1(pos, cent):
      r, c = pos
      rCent, cCent = cent
      return abs(r - rCent) + abs(c - cCent)

   def call(world, entity, targ, style, freeze=False):
      entity.history._attack = {}
      entity.history._attack['target'] = targ.entID
      entity.history._attack['style'] = style.__class__.__name__
      if entity.entID == targ.entID:
         entity.history._attack = None
         return

      dmg = combat.attack(entity, targ, style)
      #entity.applyDamage(dmg, style.__name__.lower())
      #targ.receiveDamage(dmg)
      return dmg

   def args(stim, entity, config):
      return [Melee, Range, Mage]

class AttackStyle(Node):
   pass

class Melee(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 0
   @staticproperty
   def edges():
      return None

   def call(world, entity, targ):
      #dmg = world.config.MELEEDAMAGE
      Attack.call(world, entity, targ, entity.skills.melee)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config, config.MELEERANGE)

class Range(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 1
   @staticproperty
   def edges():
      return None

   def call(world, entity, targ):
      #dmg = world.config.RANGEDAMAGE
      Attack.call(world, entity, targ, entity.skills.range);

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config, config.RANGERANGE)

class Mage(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 2
   @staticproperty
   def edges():
      return None

   def call(world, entity, targ):
      #dmg = world.config.MAGEDAMAGE
      dmg = Attack.call(world, entity, targ, entity.skills.mage, freeze=True)
      if dmg is not None and dmg > 0:
         targ.status.freeze.update(3)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config, config.MAGERANGE)

class Reproduce:
   pass

class Skill(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Harvest, Process]

   def args(stim, entity, config):
      return Skill.edges

class Harvest(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Fish, Mine]

   def args(stim, entity, config):
      return Harvest.edges

class Fish(Node):
   nodeType = NodeType.ACTION

class Mine(Node):
   nodeType = NodeType.ACTION

class Process(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Cook, Smith]

   def args(stim, entity, config):
      return Process.edges

class Cook(Node):
   nodeType = NodeType.ACTION

class Smith(Node):
   nodeType = NodeType.ACTION

class Exchange(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Buy, Sell, CancelOffer]

   def args(stim, entity, config):
      return Exchange.edges

class Buy(Node):
   nodeType = NodeType.ACTION

class Sell(Node):
   nodeType = NodeType.ACTION

class CancelOffer(Node):
   nodeType = NodeType.ACTION

class Message:
   pass

Action.hook()
