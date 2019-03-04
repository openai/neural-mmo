import numpy as np

from forge.blade.action import action
from forge.blade.systems import skill, droptable

class Entity():
   def __init__(self, pos):
      self.pos = pos
      self.alive = True
      self.skills = skill.Skills()
      self.entityIndex=0
      self.health = -1
      self.lastAttacker = None

   def act(self, world):
      pass

   def decide(self, stimuli):
      pass

   def death(self):
      pass

   def registerHit(self, attacker, dmg):
      self.lastAttacker = attacker
      self.health -= dmg

   def remove(self, ent):
      r, c = self.pos
      ent[r, c] = 0

   def isAlive(self):
      return self.health > 0

   @property
   def isPC(self):
      return False
