from forge.blade.systems import skill

class Item:
   createSkill = None
   useSkill = None

   createLevel = None
   useLevel = None

   exp = 0
   amtMade = 1
   alwaysSucceeds = True
   recipe = None 

class Tool(Item): pass

class Weapon(Item):
   alwaysSucceeds = True
   createSkill = skill.Smithing

#Used as a stand in when no weapon is present
class Base(Weapon):
   useLevel = 1
   attack = 0
   strength = 0
   ammo = None

class Gold(Item):
   pass

class Hammer(Tool): pass
class Tinderbox(Tool): pass
class Pickaxe(Tool): pass
class Rod(Tool): pass
