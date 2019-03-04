from forge.blade.systems import skill
from forge.blade.item import item

class Ore(item.Item):
   createSkill = skill.Mining
   useSkill = skill.Smithing

class Copper(Ore):
   createLevel = 1
   useLevel = 1
   exp = 10 
class Tin(Ore):
   createLevel = 5
   useLevel = 5
   exp = 20
   
class Iron(Ore):
   createLevel = 10
   useLevel = 10
   exp = 30


