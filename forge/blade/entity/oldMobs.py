#NPC definitions with AI overrides per NPC, as required.

from forge.blade.entity import NPC
from forge.blade.item import Item, RawMeat, Sword
from forge.blade.lib import AI
from forge.blade.lib.Enums import Material
from forge.blade.modules import DropTable, Skill


class Chicken(NPC.Passive):
   def __init__(self, pos):
      super().__init__(pos)
      self.index = 2
      self.maxHealth = 5
      self.health = self.maxHealth
      self.expOnKill = 10
   
      self.drops.add(RawMeat.Chicken, 1)
      self.drops.add(RawMeat.Chicken, 1, 0.5)
   
   def decide(self, world):
      action, args = AI.randomOnTurf(world, self, 
            [Material.GRASS.value])
      return action, args

class Goblin(NPC.PassiveAgressive):
   def __init__(self, pos):
      super().__init__(pos)
      self.index = 3
      self.searchRange = 5
      self.maxHealth = 15
      self.health = self.maxHealth
      self.expOnKill = 50
      self.skills.melee.exp = Skill.Skill.expCalculator.expAtLevel(5)
      self.skills.defense.exp = Skill.Skill.expCalculator.expAtLevel(5)

      self.drops.add(RawMeat.Goblin, 1)
      self.drops.add(Item.Gold, DropTable.Range(1, 100), 0.5)
      self.drops.add(Sword.Copper, 1, 0.2)
   
   def decide(self, world):
      #action, args = AI.turfSearchAndDestroy(world, self, 
      #      whitelist=[Material.FOREST.value])
      action, args = AI.randomOnTurf(world, self, 
            [Material.FOREST.value])
      return action, args

