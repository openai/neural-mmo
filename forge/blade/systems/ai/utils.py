from pdb import set_trace as T
import numpy as np
import random

from forge.blade.lib.utils import inBounds
from forge.blade.systems import combat
from forge.blade.lib import material
from forge.blade.io.stimulus.static import Stimulus
from queue import PriorityQueue, Queue

from forge.blade.systems.ai.dynamic_programming import map_to_rewards, \
   compute_values, max_value_direction_around

class Observation:
    def __init__(self, config, obs):
        self.config = config
        self.obs    = obs
        self.delta  = config.NSTIM

        self.tiles  = self.obs['Tile']['Continuous']
        self.agents = self.obs['Entity']['Continuous']
        self.n      = int(self.obs['Entity']['N']) 

    def tile(self, rDelta, cDelta):
        #return self.tiles[int(r*self.config.WINDOW + c)]
        return self.tiles[self.config.WINDOW * (self.delta + rDelta) + self.delta + cDelta]

    @property
    def agent(self):
        return self.agents[0]

    #@property
    #def agentID(self):
    #    return self.attribute(self.agent, Stimulus.Entity.ID

    @staticmethod
    def attribute(ary, attr):
        return float(ary[attr.index])

def validTarget(ent, targ, rng):
   if targ is None or not targ.alive:
      return False
   if lInf(ent, targ) > rng:
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

'''
def closestTarget(config, ob):
   shortestDist = np.inf
   closestAgent = None

   Entity = Stimulus.Entity
   agent  = ob.agent

   sr = Observation.attribute(agent, Entity.R)
   sc = Observation.attribute(agent, Entity.C)
   start = (sr, sc)

   for target in ob.agents:
      exists = Observation.attribute(target, Entity.Self)
      if not exists:
         continue

      tr = Observation.attribute(target, Entity.R)
      tc = Observation.attribute(target, Entity.C)

      goal = (tr, tc)
      dist = l1(start, goal)

      if dist < shortestDist and dist != 0:
          shortestDist = dist
          closestAgent = target

   if closestAgent is None:
      return None, None

   targID = Observation.attribute(closestAgent, Entity.ID)
   return int(targID), shortestDist
'''

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


def forageDP(tiles, entity):
   horizon = entity.vision
   line, column = entity.pos

   tiles = cropTilesAround((line, column), horizon, tiles)

   reward_matrix = map_to_rewards(tiles, entity)
   value_matrix = compute_values(reward_matrix)

   max_value_line, max_value_column = max_value_direction_around(
      (min(horizon, len(value_matrix) - 1),
       min(horizon, len(value_matrix[0]) - 1)), value_matrix)

   return max_value_line, max_value_column

def inSight(dr, dc, vision):
    return (
          dr >= -vision and
          dc >= -vision and
          dr <= vision and
          dc <= vision)

def vacant(tile):
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


def forageDijkstra(config, ob, food_max, water_max, cutoff=100):
   vision = config.NSTIM
   Entity = Stimulus.Entity
   Tile   = Stimulus.Tile

   agent  = ob.agent
   food   = Observation.attribute(agent, Entity.Food)
   water  = Observation.attribute(agent, Entity.Water)

   best      = -1000 
   start     = (0, 0)
   goal      = (0, 0)

   reward    = {start: (food, water)}
   backtrace = {start: None}

   queue = Queue()
   queue.put(start)

   while not queue.empty():
      cutoff -= 1
      if cutoff <= 0:
         break

      cur = queue.get()
      for nxt in adjacentPos(cur):
         if nxt in backtrace:
            continue

         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = Observation.attribute(tile, Tile.Index)
         occupied = Observation.attribute(tile, Tile.NEnts)

         if occupied:
            continue

         if matl in (material.Lava.index, material.Water.index, material.Stone.index, material.Orerock.index):
            continue

         food, water = reward[cur]
         food  = max(0, food - 1)
         water = max(0, water - 1)

         if matl == material.Forest.index:
            food = min(food+food_max//2, food_max)
         for pos in adjacentPos(nxt):
            if not inSight(*pos, vision):
               continue

            tile = ob.tile(*pos)
            matl = Observation.attribute(tile, Tile.Index)
 
            if matl == material.Water.index:
               water = min(water+water_max//2, water_max)
               break

         reward[nxt] = (food, water)

         total = min(food, water)
         if total > best or (
                 total == best and max(food, water) > max(reward[goal])):
            best = total
            goal = nxt

         queue.put(nxt)
         backtrace[nxt] = cur

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   return goal


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
         if tiles[nxt].impassible:
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

def aStars(config, ob, actions, rr, cc, cutoff=100):
   Entity = Stimulus.Entity
   Tile   = Stimulus.Tile
   vision = config.NSTIM

   if rr == 0 and cc == 0:
      return (0, 0)

   start = (0, 0)
   goal  = (rr, cc)

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
         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = Observation.attribute(tile, Tile.Index)
         occupied = Observation.attribute(tile, Tile.NEnts)

         if occupied:
            continue

         if matl in (material.Lava.index, material.Water.index, material.Stone.index, material.Orerock.index):
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

   if goal not in backtrace:
      goal = closestPos

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   tile     = ob.tile(*goal)
   matl     = Observation.attribute(tile, Tile.Index)

   #if goal not in ((-1, 0), (1, 0), (0, -1), (0, 1)):
   #    T()

   return goal


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
