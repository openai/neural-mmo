from forge.blade.systems import skill
from forge.blade.item import item

class RawFish(item.Item):
   createSkill = skill.Fishing
   useSkill = skill.Cooking
   alwaysSucceeds = False

class Shrimp(RawFish):
   createLevel = 1
   useLevel = 1
   exp = 10

class Sardine(RawFish):
   createLevel = 5
   useLevel = 5 
   exp = 20

class Herring(RawFish):
   createLevel = 10
   useLevel = 10
   exp = 30


