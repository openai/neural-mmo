from pdb import set_trace as T
from forge.blade.action import tree
from forge.blade.lib import utils, enums
import numpy as np

class staticproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

#Traverses all edges without and decisions
class StaticNode:
   def argType():
      return None

   @staticproperty
   def edges():
      return []

#Picks an edge to follow
class SelectionNode:
   @staticproperty
   def argType():
      return tree.ConstDiscrete

   @staticproperty
   def args(stim, entity, config):
      return None

#Proposes an argument to execute
class VariableNode:
   @staticproperty
   def argType():
      return tree.VariableDiscrete

   @staticproperty
   def args(stim, entity, config):
      return None

#Proposes an argument to execute
class ConstantNode:
   @staticproperty
   def argType():
      return tree.VariableDiscrete

   @staticproperty
   def args(stim, entity, config):
      return None


class ActionRoot(StaticNode):
   @staticproperty
   def edges():
      return [Move, Attack, Exchange, Skill]

class Move(SelectionNode):
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

   @staticproperty
   def edges():
      return [Pass, North, South, East, West]

   def args(stim, entity, config):
      return Move.edges

class Pass(ConstantNode, Move):
   def call(world, entity):
      Move.call(world, entity, 0, 0)

class North(ConstantNode, Move):
   def call(world, entity):
      Move.call(world, entity, -1, 0)

class South(ConstantNode, Move):
   def call(world, entity):
      Move.call(world, entity, 1, 0)

class East(ConstantNode, Move):
   def call(world, entity):
      Move.call(world, entity, 0, 1)

class West(ConstantNode, Move):
   def call(world, entity):
      Move.call(world, entity, 0, -1)

class Attack(SelectionNode):
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

class Melee(ConstantNode, Attack):
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

class Range(ConstantNode, Attack):
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

class Mage(ConstantNode, Attack):
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

class Reproduce:
   pass

class Skill(SelectionNode):
   @staticproperty
   def edges():
      return [Harvest, Process]

   def args(stim, entity, config):
      return Skill.edges

class Harvest(SelectionNode):
   @staticproperty
   def edges():
      return [Fish, Mine]

   def args(stim, entity, config):
      return Harvest.edges

class Fish(ConstantNode, Harvest):
   pass

class Mine(ConstantNode, Harvest):
   pass

class Process(SelectionNode):
   @staticproperty
   def edges():
      return [Cook, Smith]

   def args(stim, entity, config):
      return Process.edges

class Cook(ConstantNode, Process):
   pass

class Smith(ConstantNode, Process):
   pass

class Exchange(SelectionNode):
   @staticproperty
   def edges():
      return [Buy, Sell, CancelOffer, Pass]

   def args(stim, entity, config):
      return Exchange.edges

class Buy(ConstantNode, Exchange):
   pass

class Sell(ConstantNode, Exchange):
   pass

class CancelOffer(ConstantNode, Exchange):
   pass

class Message:
   pass
