from pdb import set_trace as T
import numpy as np
import random as rand

from neural_mmo.forge.blade.systems.ai import utils
from neural_mmo.forge.blade.io.action import static as Action

def random():
   return rand.choice(Action.Direction.edges)

def randomSafe(tiles, ent):
   r, c  = ent.base.pos
   cands = []
   if not tiles[r-1, c].lava:
      cands.append(Action.North)
   if not tiles[r+1, c].lava:
      cands.append(Action.South)
   if not tiles[r, c-1].lava:
      cands.append(Action.West)
   if not tiles[r, c+1].lava:
      cands.append(Action.East)
   
   return rand.choice(cands)

def habitable(tiles, ent):
   r, c  = ent.base.pos
   cands = []
   if tiles[r-1, c].vacant:
      cands.append(Action.North)
   if tiles[r+1, c].vacant:
      cands.append(Action.South)
   if tiles[r, c-1].vacant:
      cands.append(Action.West)
   if tiles[r, c+1].vacant:
      cands.append(Action.East)
   
   if len(cands) == 0:
      return Action.North

   return rand.choice(cands)

def towards(direction):
   if direction == (-1, 0):
      return Action.North
   elif direction == (1, 0):
      return Action.South
   elif direction == (0, -1):
      return Action.West
   elif direction == (0, 1):
      return Action.East
   else:
      return random()

def bullrush(ent, targ):
   direction = utils.directionTowards(ent, targ)
   return towards(direction)

def pathfind(tiles, ent, targ):
   direction = utils.aStar(tiles, ent.pos, targ.pos)
   return towards(direction)

def antipathfind(tiles, ent, targ):
   er, ec = ent.pos
   tr, tc = targ.pos
   goal   = (2*er - tr , 2*ec-tc)
   direction = utils.aStar(tiles, ent.pos, goal)
   return towards(direction)

  


