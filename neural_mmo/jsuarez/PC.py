from pdb import set_trace as T
import numpy as np

from sim.lib import MultiSet
from sim.modules import Inventory
from sim.entity.Entity import Entity
from sim.item import Food

class PC(Entity):
   def __init__(self, cpu, pos):
      super().__init__(pos)
      self.cpu = cpu
      self.inv = MultiSet.MultiSet(capacity=1024)

      self.hitRange = 4
      self.timeAlive = 0
      self.maxHealth = 10
      self.maxFood = 16
      self.maxWater = 16
      self.health = self.maxHealth
      self.food = self.maxFood
      self.water = self.maxWater
      self.decProb = 0.2
      self.index = 1
      self.lastAction = None

      self.equipment = Inventory.Inventory()
      self.inbox = []

   def act(self, world, actions):
      if (self.food > self.maxFood//2 and
            self.water > self.maxWater//2 and
         self.health < self.maxHealth):
            self.health += 1
      self.decrementFood()
      self.decrementWater()

      stim = world.stim(self.pos)
      action, args = self.decide(stim, actions)
      action(world, self, *(args))
      self.lastAction = action

      return self.post()

   #PCs interact with the world only through stimuli
   #to prevent cheating
   def decide(self, stimuli, actions):
      action, args = self.cpu.interact(self, stimuli, actions)
      return action, args

   def post(self):
      self.inbox = []
      self.timeAlive += 1
      #if self.timeAlive % 10 == 0:
      #   return self.cpu.reproduce()

   def receiveDrops(self, drops):
      for e in drops:
         item, num = e
         self.inv.add(item, num)

   def receiveMessage(self, sender, data):
      self.inbox += [(sender, data)]

   def eat(self, food):
      self.health = min(self.health+food.heal, self.maxHealth)
      self.inv.remove(food, 1)

   def equipArmor(self, item):
      if not self.equipment.armor.isBase:
         self.inv.add(self.equipment.armor)
      self.equipment.armor = item

   def equipMelee(self, item):
      if not self.equipment.melee.isBase:
         self.inv.add(self.equipment.melee)
      self.equipment.melee = item

   def equipRanged(self, item):
      if not self.equipment.ranged.isBase:
         self.inv.add(self.equipment.ranged)
      self.equipment.ranged = item

   def unequipArmor(self):
      if self.equipment.armor.isBase:
         return
      self.inv.add(self.equipment.armor)
      self.equipment.resetArmor()

   def unequipMelee(self):
      if self.equipment.melee.isBase:
         return
      self.inv.add(self.equipment.melee)
      self.equipment.resetMelee()

   def unequipRanged(self):
      if self.equipment.ranged.isBase:
         return
      self.inv.add(self.equipment.ranged)
      self.equipment.resetRanged()

   def decrementFood(self):
      if np.random.rand() < self.decProb:
         if self.food > 0:
            self.food -= 1
         else:
            self.health -= 1

   def decrementWater(self):
      if np.random.rand() < self.decProb:
         if self.water > 0:
            self.water -= 1
         else:
            self.health -= 1

   @property
   def isPC(self):
      return True
