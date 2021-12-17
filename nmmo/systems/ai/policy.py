from pdb import set_trace as T

from nmmo.systems.ai import behavior, utils

def passive(realm, entity):
   behavior.update(entity)
   actions = {}

   behavior.meander(realm, actions, entity)

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

   # This is probably slow
   if not entity.target:
      entity.target = utils.closestTarget(entity, realm.map.tiles,
                                          rng=entity.vision)

   if not entity.target:
      behavior.meander(realm, actions, entity)
   else:
      behavior.hunt(realm, actions, entity)

   return actions
