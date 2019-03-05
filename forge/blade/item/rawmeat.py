from forge.blade.systems import Skill
from forge.blade.item import Item

class RawMeat(Item.Item):
   useSkill = Skill.Cooking
   alwaysSucceeds = False

class Chicken(RawMeat):
   useLevel = 1
   exp = 20

class Goblin(RawMeat):
   useLevel = 5 
   exp = 40

