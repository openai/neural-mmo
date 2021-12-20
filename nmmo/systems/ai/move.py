from pdb import set_trace as T
import numpy as np
import random as rand

import nmmo
from nmmo.systems.ai import utils

def random():
   return rand.choice(nmmo.action.Direction.edges)

def randomSafe(tiles, ent):
   r, c  = ent.base.pos
   cands = []
   if not tiles[r-1, c].lava:
      cands.append(nmmo.action.North)
   if not tiles[r+1, c].lava:
      cands.append(nmmo.action.South)
   if not tiles[r, c-1].lava:
      cands.append(nmmo.action.West)
   if not tiles[r, c+1].lava:
      cands.append(nmmo.action.East)
   
   return rand.choice(cands)

def habitable(tiles, ent):
   r, c  = ent.base.pos
   cands = []
   if tiles[r-1, c].vacant:
      cands.append(nmmo.action.North)
   if tiles[r+1, c].vacant:
      cands.append(nmmo.action.South)
   if tiles[r, c-1].vacant:
      cands.append(nmmo.action.West)
   if tiles[r, c+1].vacant:
      cands.append(nmmo.action.East)
   
   if len(cands) == 0:
      return nmmo.action.North

   return rand.choice(cands)

def towards(direction):
   if direction == (-1, 0):
      return nmmo.action.North
   elif direction == (1, 0):
      return nmmo.action.South
   elif direction == (0, -1):
      return nmmo.action.West
   elif direction == (0, 1):
      return nmmo.action.East
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

  


