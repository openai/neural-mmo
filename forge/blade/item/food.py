from forge.blade.systems import Skill
from forge.blade.item import Item

class Food(Item.Item):
   createSkill = Skill.Cooking
   useSkill = Skill.Constitution
   heal = None

class Ration(Food):
   useLevel = 1
   exp = 0
   heal = 1

class Shrimp(Food):
   createLevel = 1
   useLevel = 1
   exp = 10
   heal = 2

class Sardine(Food):
   createLevel = 5
   useLevel = 5
   exp = 20
   heal = 3

class Herring(Food):
   createLevel = 10
   useLevel = 10
   exp = 30
   heal = 5

class Chicken(Food):
   createLevel = 1
   useLevel = 1
   exp = 10
   heal = 3

class Goblin(Food):
   createLevel = 1
   useLevel = 5
   exp = 10
   heal = 5
