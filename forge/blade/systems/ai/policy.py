from pdb import set_trace as T

from forge.blade.systems.ai import behavior, move, attack, utils
from forge.blade.io.action import static as Action
from forge.blade.systems import combat

def passive(realm, entity):
   behavior.update(entity)
   actions = {}

   if not entity.attacker:
      behavior.meander(realm, actions, entity)
   else:
      entity.target = entity.attacker
      behavior.hunt(realm, actions, entity)

   return actions

def neutral(realm, entity):
   behavior.update(entity)
   actions = {}

   if not entity.attacker:
      behavior.meander(realm, actions, entity)
   else:
      entity.target = entity.attacker
      behavior.hunt(realm, actions, entity)

   return actions

def hostile(realm, entity):
   behavior.update(entity)
   actions = {}

   #This is probably slow
   if not entity.target:
      entity.target = utils.closestTarget(entity, realm.map.tiles, rng=entity.vision)

   if not entity.target:
      behavior.meander(realm, actions, entity)
   else:
      behavior.hunt(realm, actions, entity)

   return actions

def baseline(realm, entity):
   behavior.update(entity)
   actions = {}

   if entity.resources.food <= 15 or entity.resources.water <= 15:
      behavior.forage(realm, actions, entity)
   else:
      behavior.inward(realm, actions, entity)
   
   return actions

