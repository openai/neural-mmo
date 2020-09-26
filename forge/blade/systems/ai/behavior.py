from pdb import set_trace as T
import numpy as np

from forge.blade.systems.ai import move, attack, utils
from forge.blade.io.action import static as Action

def update(entity):
   '''Update validity of tracked entities'''
   if not utils.validTarget(entity, entity.attacker, entity.vision):
      entity.attacker = None
   if not utils.validTarget(entity, entity.target, entity.vision):
      entity.target = None
   if not utils.validTarget(entity, entity.closest, entity.vision):
      entity.closest = None

   if entity.__class__.__name__ != 'Player':
      return

   if (not utils.validResource(entity, entity.food, entity.vision) or
         not utils.validResource(entity, entity.water, entity.vision)):
      entity.food  = None
      entity.water = None
      
def pathfind(realm, actions, entity, target):
   actions[Action.Move]   = {Action.Direction: move.pathfind(realm.map.tiles, entity, target)}

def inward(realm, actions, entity):
   R, C = realm.shape
   r, c = entity.pos
   
   #Direction 
   rr   = int(r + np.clip(R//2-r, -entity.vision, entity.vision))
   cc   = int(c + np.clip(C//2-c, -entity.vision, entity.vision))

   tile = realm.map.tiles[rr, cc]
   pathfind(realm, actions, entity, tile)
    

def meander(realm, actions, entity):
   actions[Action.Move] = {Action.Direction: move.habitable(realm.map.tiles, entity)}

def evade(realm, actions, entity):
   actions[Action.Move] = {Action.Direction: move.antipathfind(realm.map.tiles, entity, entity.attacker)}

def hunt(realm, actions, entity):
   actions[Action.Attack] = {Action.Style: entity.skills.style,
                             Action.Target: entity.target}

   
   distance = utils.distance(entity, entity.target)
   if distance == 0:
      direction = move.random()
   elif distance == 1:
      return
   else:
      direction = move.pathfind(realm.map.tiles, entity, entity.target)

   actions[Action.Move]   = {Action.Direction: direction}

def forage(realm, actions, entity):
   direction = utils.forage(realm.map.tiles, entity)
   actions[Action.Move]   = {Action.Direction: move.towards(direction)}
