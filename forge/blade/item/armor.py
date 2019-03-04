from pdb import set_trace as T
from forge.blade import systems
from forge.blade.systems import skill
from forge.blade.item import item, ore

class Armor(item.Item):
   createSkill = skill.Smithing
   useSkill = skill.Defense
   alwaysSucceeds = True
   oreReq = 4
   defense = 0

class Base(Armor):
   useLevel = 1
   defense = 0

class Copper(Armor):
   createLevel = 1
   useLevel = 1
   exp = 40
   recipe = systems.Recipe(ore.Copper, Armor.oreReq)
   defense = 55

class Tin(Armor):
   createLevel = 5
   useLevel = 5
   exp = 80
   recipe = systems.Recipe(ore.Tin, Armor.oreReq)
   defense = 90

class Iron(Armor):
   createLevel = 10
   useLevel = 10
   exp = 120
   recipe = systems.Recipe(ore.Iron, Armor.oreReq)
   defense = 105

