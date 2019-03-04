from forge.blade import systems
from forge.blade.systems import skill
from forge.blade.item import item, ore

class Knife(item.Weapon):
   createSkill = skill.Smithing
   useSkill = skill.Ranged
   amtMade = 10
   oreReq  = 1
   
class Copper(Knife):
   createLevel = 1
   useLevel = 1
   exp = 10
   attack = 5
   strength = 4
   recipe = systems.Recipe(ore.Copper, Knife.oreReq, amtMade=Knife.amtMade)

class Tin(Knife):
   createLevel = 5
   useLevel = 5
   exp = 20
   recipe = systems.Recipe(ore.Tin, Knife.oreReq, amtMade=Knife.amtMade)
   attack = 8
   strength = 7

class Iron(Knife):
   createLevel = 10
   useLevel = 10
   exp = 30
   recipe = systems.Recipe(ore.Iron, Knife.oreReq, amtMade=Knife.amtMade)
   attack = 10
   strength = 8
