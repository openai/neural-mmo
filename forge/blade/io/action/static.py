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
      return len(Action.arguments)

   def args(stim, entity, config):
      return Static.edges

   #Called upon module import (see bottom of file)
   #Sets up serialization domain
   def hook():
      idx = 0
      arguments = []
      for action in Action.edges:
         for args in action.edges:
            if not 'edges' in args.__dict__:
               continue
            for arg in args.edges: 
               arguments.append(arg)
               arg.serial = tuple([idx])
               arg.idx = idx 
               idx += 1
      Action.arguments = arguments

class Move(Node):
   priority = 0
   nodeType = NodeType.SELECTION
   def call(world, entity, direction):
      r, c = entity.base.pos
      entity.history.lastPos = (r, c)
      rDelta, cDelta = direction.delta
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
      return [Direction]

   @staticproperty
   def leaf():
      return True

class Direction(Node):
   @staticproperty
   def edges():
      return [North, South, East, West]

   def args(stim, entity, config):
      return Direction.edges

class North:
   delta = (-1, 0)

class South:
   delta = (1, 0)

class East:
   delta = (0, 1)

class West:
   delta = (0, -1)


class Attack(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def n():
      return 3

   @staticproperty
   def edges():
      return [Style, Target]

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

   def call(world, entity, style, targ):
      entity.history.attack = {}
      entity.history.attack['target'] = targ.entID
      entity.history.attack['style'] = style.__name__
      if entity.entID == targ.entID:
         entity.history.attack = None
         return

      dmg = combat.attack(entity, targ, style.skill(entity))
      if style.freeze and dmg is not None and dmg > 0:
         targ.status.freeze.update(3)

      #entity.applyDamage(dmg, style.__name__.lower())
      #targ.receiveDamage(dmg)
      return dmg

class Style(Node):
   @staticproperty
   def edges():
      return [Melee, Range, Mage]

   def args(stim, entity, config):
      return Style.edges


class Target(Node):
   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config, config.MELEERANGE)

class Melee(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 0
   freeze=False

   def skill(entity):
      return entity.skills.melee

class Range(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 1
   freeze=False

   def skill(entity):
      return entity.skills.range

class Mage(Node):
   priority = 1
   nodeType = NodeType.ACTION
   index = 2
   freeze=True

   def skill(entity):
      return entity.skills.mage

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

class BecomeSkynet:
   pass

Action.hook()
