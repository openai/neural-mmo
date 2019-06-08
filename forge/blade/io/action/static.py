from pdb import set_trace as T
import numpy as np

from forge.blade.lib import utils, enums
from forge.blade.lib.utils import staticproperty
from forge.blade.io.action.node import Node, NodeType

#ActionRoot
class Static(Node):
   nodeType = NodeType.SELECTION

   @staticproperty
   def edges():
      #return [Move, Attack, Exchange, Skill]
      return [Move, Attack]

   @staticproperty
   def n():
      return len(Static.actions)

   def args(stim, entity, config):
      return Static.edges

   #Called upon module import (see bottom of file)
   #Sets up serialization domain
   def hook():
      actions = Static.flat()
      Static.actions = actions

      for idx, atn in enumerate(actions):
         atn.serial = tuple([idx])
         atn.idx = idx 

   def flat(root=None):
      if root is None:
         root = Static

      rets = [root]
      if root.nodeType is NodeType.SELECTION:
         for edge in root.edges:
            rets += Static.flat(edge)

      return rets

class Move(Node):
   nodeType = NodeType.SELECTION
   def call(world, entity, rDelta, cDelta):
      r, c = entity.pos
      entity._lastPos = entity.pos
      rNew, cNew = r+rDelta, c+cDelta
      if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
         return
      if not utils.inBounds(rNew, cNew, world.shape):
         return
      if entity.freeze > 0:
         return

      entity._r.update(rNew)
      entity._c.update(cNew)
      entID = entity.entID
      
      r, c = entity.lastPos
      world.env.tiles[r, c].delEnt(entID)

      r, c = entity.pos
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
      return []

class South(Node):
   priority = 0
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 1, 0)

   def args(stim, entity, config):
      return []

class East(Node):
   priority = 0
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 0, 1)

   def args(stim, entity, config):
      return []

class West(Node):
   priority = 0
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 0, -1)

   def args(stim, entity, config):
      return []


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

   def inRange(entity, stim, N):
      R, C = stim.shape
      R, C = R//2, C//2
      #R, C = entity.pos

      rets = []
      for r in range(R-N, R+N+1):
         for c in range(C-N, C+N+1):
            for e in stim[r, c].ents.values():
               rets.append(e)
      return rets

   def l1(pos, cent):
      r, c = pos
      rCent, cCent = cent
      return abs(r - rCent) + abs(c - cCent)

   def call(world, entity, targ, damageF, freeze=False):
      if entity.entID == targ.entID:
         entity._attack = None
         return
      #entity.targPos = targ.pos
      #entity.attkPos = entity.lastPos
      #entity.targ = targ
      damage = damageF(entity, targ)
      assert type(damage) == int
      if freeze and damage > 0:
         targ._freeze.update(3)
      return
      #return damage

   def args(stim, entity, config):
      return [Melee, Range, Mage]
      #return Melee.args(stim, entity, config) + Range.args(stim, entity, config) + Mage.args(stim, entity, config)

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
      damageF = world.config.MELEEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MELEERANGE)

class Range(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 1
   @staticproperty
   def edges():
      return None

   def call(world, entity, targ):
      damageF = world.config.RANGEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.RANGERANGE)

class Mage(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 2
   @staticproperty
   def edges():
      return None

   def call(world, entity, targ):
      damageF = world.config.MAGEDAMAGE
      dmg = Attack.call(world, entity, targ, damageF, freeze=True)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MAGERANGE)

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

Static.hook()
