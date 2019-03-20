from pdb import set_trace as T
from forge.blade.action import tree
from forge.blade.lib import utils, enums
import numpy as np

class staticproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class ActionRoot:
   @staticproperty
   def argType():
      return None

   @staticproperty
   def edges():
      return [Move, Attack]

class Move(ActionRoot):
   @staticproperty
   def argType():
      return tree.ConstDiscrete

   @staticproperty
   def n():
      return 5

   @staticproperty
   def edges():
      return None

   priority = 1
   def call(world, entity, rDelta, cDelta):
      r, c = entity.pos
      rNew, cNew = r+rDelta, c+cDelta
      if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
         return
      if not utils.inBounds(rNew, cNew, world.shape):
         return
      if entity.freeze > 0:
         return

      entity._pos = rNew, cNew
      entID = entity.entID
      
      r, c = entity.lastPos
      world.env.tiles[r, c].delEnt(entID)

      r, c = entity.pos
      world.env.tiles[r, c].addEnt(entID, entity)

   def args(stim, entity, config):
      rets = []
      for delta in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)):
         r, c = delta
         #r, c = Arg(r), Arg(c)
         rets.append((r, c))
      return rets

   @property
   def nArgs():
      return len(Move.args(None, None))

class Attack(ActionRoot):
   @staticproperty
   def argType():
      return tree.ConstDiscrete

   @staticproperty
   def n():
      return 3

   @staticproperty
   def edges():
      return [Melee, Range, Mage]

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
         targ._freeze = 3
      return
      #return damage

   def args(stim, entity, config):
      return [Melee, Range, Mage]
      #return Melee.args(stim, entity, config) + Range.args(stim, entity, config) + Mage.args(stim, entity, config)

class Melee(Attack):
   @staticproperty
   def argType():
      return tree.VariableDiscrete

   @staticproperty
   def edges():
      return None

   priority = 2
   def call(world, entity, targ):
      damageF = world.config.MELEEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MELEERANGE)

class Range(Attack):
   @staticproperty
   def argType():
      return tree.VariableDiscrete

   @staticproperty
   def edges():
      return None

   priority = 2
   def call(world, entity, targ):
      damageF = world.config.RANGEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.RANGERANGE)

class Mage(Attack):
   @staticproperty
   def argType():
      return tree.VariableDiscrete

   @staticproperty
   def edges():
      return None

   priority = 2
   def call(world, entity, targ):
      damageF = world.config.MAGEDAMAGE
      dmg = Attack.call(world, entity, targ, damageF, freeze=True)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MAGERANGE)

class Pass(Move):
   priority = 0

   @staticmethod
   def call(world, entity):
      return

   def args(stim, entity, config):
      return [()]

   @property
   def nArgs():
      return 1

class Reproduce:
   pass

class Skill:
   pass

class Harvest:
   pass

class Fish:
   pass

class Mine:
   pass

class Process:
   pass

class Cook:
   pass

class Smith:
   pass

class Exchange:
   pass

class Buy:
   pass

class Sell:
   pass

class CancelOffer:
   pass

class Message:
   pass
