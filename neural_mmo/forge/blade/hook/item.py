#Iterates through all items and hooks them to their respective skills
from neural_mmo.forge.blade.item import item
from neural_mmo.forge.blade.lib import utils

#Filled in at runtime with items
class ItemList:
   items = []

def hook():
   cls = item.Item
   for e in utils.terminalClasses(cls):
      skill = e.createSkill
      if skill is not None:
         if type(skill.skillItems) != list:
            skill.skillItems = []
         skill.skillItems += [e]
         ItemList.items += [e]
