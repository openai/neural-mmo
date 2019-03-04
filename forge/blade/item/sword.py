from forge.blade.systems.Recipe import Recipe
from forge.blade.systems import Skill
from forge.blade.item import Item, Ore


class Sword(Item.Weapon):
   createSkill = Skill.Smithing
   useSkill = Skill.Melee
   oreReq = 2

class Copper(Sword):
   createLevel = 1
   useLevel = 1
   exp = 10
   recipe = Recipe(Ore.Copper, Sword.oreReq)
   attack = 10
   strength = 9
   
class Tin(Sword):
   createLevel = 5
   useLevel = 5
   exp = 20
   recipe = Recipe(Ore.Tin, Sword.oreReq)
   attack = 15
   strength = 14

class Iron(Sword):
   createLevel = 10
   useLevel = 10
   exp = 30
   recipe = Recipe(Ore.Iron, Sword.oreReq)
   attack = 19
   strength = 14

