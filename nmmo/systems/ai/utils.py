from pdb import set_trace as T
import numpy as np
import random

from nmmo.lib.utils import inBounds
from nmmo.systems import combat
from nmmo.lib import material
from queue import PriorityQueue, Queue

from nmmo.systems.ai.dynamic_programming import map_to_rewards, \
   compute_values, max_value_direction_around

def validTarget(ent, targ, rng):
   if targ is None or not targ.alive:
      return False
   if lInfty(ent.pos, targ.pos) > rng:
      return False
   return True


def validResource(ent, tile, rng):
   return tile is not None and tile.state.tex in (
      'forest', 'water') and distance(ent, tile) <= rng


def directionTowards(ent, targ):
   sr, sc = ent.base.pos
   tr, tc = targ.base.pos

   if abs(sc - tc) > abs(sr - tr):
      direction = (0, np.sign(tc - sc))
   else:
      direction = (np.sign(tr - sr), 0)

   return direction


def closestTarget(ent, tiles, rng=1):
   sr, sc = ent.base.pos
   for d in range(rng+1):
      for r in range(-d, d+1):
         for e in tiles[sr+r, sc-d].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr + r, sc + d].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr - d, sc + r].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr + d, sc + r].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

def distance(ent, targ):
   return l1(ent.pos, targ.pos)

def lInf(ent, targ):
   sr, sc = ent.pos
   gr, gc = targ.pos
   return abs(gr - sr) + abs(gc - sc)


def adjacentPos(pos):
   r, c = pos
   return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]


def cropTilesAround(position: (int, int), horizon: int, tiles):
   line, column = position

   return tiles[max(line - horizon, 0): min(line + horizon + 1, len(tiles)),
          max(column - horizon, 0): min(column + horizon + 1, len(tiles[0]))]


def inSight(dr, dc, vision):
    return (
          dr >= -vision and
          dc >= -vision and
          dr <= vision and
          dc <= vision)

def vacant(tile):
   from nmmo.io.stimulus.static import Stimulus
   Tile     = Stimulus.Tile
   occupied = Observation.attribute(tile, Tile.NEnts)
   matl     = Observation.attribute(tile, Tile.Index)

   lava    = material.Lava.index
   water   = material.Water.index
   grass   = material.Grass.index
   scrub   = material.Scrub.index
   forest  = material.Forest.index
   stone   = material.Stone.index
   orerock = material.Orerock.index

   return matl in (grass, scrub, forest) and not occupied

def meander(obs):
   from nmmo.io.stimulus.static import Stimulus

   agent  = obs.agent
   Entity = Stimulus.Entity
   Tile   = Stimulus.Tile

   r = Observation.attribute(agent, Entity.R)
   c = Observation.attribute(agent, Entity.C)

   cands = []
   if vacant(obs.tile(-1, 0)):
      cands.append((-1, 0))
   if vacant(obs.tile(1, 0)):
      cands.append((1, 0))
   if vacant(obs.tile(0, -1)):
      cands.append((0, -1))
   if vacant(obs.tile(0, 1)):
      cands.append((0, 1))
   if not cands:
      return (-1, 0)
   return random.choice(cands)

# A* Search
def l1(start, goal):
   sr, sc = start
   gr, gc = goal
   return abs(gr - sr) + abs(gc - sc)

def l2(start, goal):
   sr, sc = start
   gr, gc = goal
   return 0.5*((gr - sr)**2 + (gc - sc)**2)**0.5

#TODO: unify lInfty and lInf
def lInfty(start, goal):
   sr, sc = start
   gr, gc = goal
   return max(abs(gr - sr), abs(gc - sc))

def aStar(tiles, start, goal, cutoff=100):
   if start == goal:
      return (0, 0)

   pq = PriorityQueue()
   pq.put((0, start))

   backtrace = {}
   cost = {start: 0}

   closestPos = start
   closestHeuristic = l1(start, goal)
   closestCost = closestHeuristic

   while not pq.empty():
      # Use approximate solution if budget exhausted
      cutoff -= 1
      if cutoff <= 0:
         if goal not in backtrace:
            goal = closestPos
         break

      priority, cur = pq.get()

      if cur == goal:
         break

      for nxt in adjacentPos(cur):
         if not inBounds(*nxt, tiles.shape):
            continue
         if tiles[nxt].occupied:
            continue

         newCost = cost[cur] + 1
         if nxt not in cost or newCost < cost[nxt]:
            cost[nxt] = newCost
            heuristic = lInfty(goal, nxt)
            priority = newCost + heuristic

            # Compute approximate solution
            if heuristic < closestHeuristic or (
                    heuristic == closestHeuristic and priority < closestCost):
               closestPos = nxt
               closestHeuristic = heuristic
               closestCost = priority

            pq.put((priority, nxt))
            backtrace[nxt] = cur

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   sr, sc = start
   gr, gc = goal

   return (gr - sr, gc - sc)
# End A*

# Adjacency functions
def adjacentTiles(tiles, ent):
   r, c = ent.base.pos


def adjacentDeltas():
   return [(-1, 0), (1, 0), (0, 1), (0, -1)]


def l1Deltas(s):
   rets = []
   for r in range(-s, s + 1):
      for c in range(-s, s + 1):
         rets.append((r, c))
   return rets


def posSum(pos1, pos2):
   return pos1[0] + pos2[0], pos1[1] + pos2[1]


def adjacentEmptyPos(env, pos):
   return [p for p in adjacentPos(pos)
           if inBounds(*p, env.size)]


def adjacentTiles(env, pos):
   return [env.tiles[p] for p in adjacentPos(pos)
           if inBounds(*p, env.size)]


def adjacentMats(tiles, pos):
   return [type(tiles[p].state) for p in adjacentPos(pos)
           if inBounds(*p, tiles.shape)]


def adjacencyDelMatPairs(env, pos):
   return zip(adjacentDeltas(), adjacentMats(env.tiles, pos))
###End###
