#Various high level routines and tools for building quick
#NPC AIs. Returns only action arguments where possible. More
#complex routines return (action, args) pairs where required.

import numpy as np
from queue import Queue
from neural_mmo.forge.blade.lib import enums, utils
from neural_mmo.forge.blade.io.action import static as Action

# Flighty 
# PassiveAggressive
# Hostile 

def turfSearchAndDestroy(world, entity, whitelist):
   for e in sortedByl1(world.env.ent, world.size, entity):
     if e.entityIndex == enums.Entity.NEURAL.value.data:
         if isAdjacent(entity.pos, e.pos):
            tree = Actions.ActionTree(world, entity, rootVersion=Actions.MeleeV2) 
            tree.decideArgs(e)
            return tree.actionArgPair()
         move = routePath(entity.pos, e.pos)
         if inWhitelist(world.env.tiles, entity.pos, move, whitelist):
            tree = Actions.ActionTree(world, entity, rootVersion=Actions.MoveV2) 
            tree.decideArgs(move)
            return tree.actionArgPair()
   return randomOnTurf(world, entity, whitelist)

def randomMove(world, entity):
   tree = Actions.ActionTree(world, entity, rootVersion=Actions.MoveV2) 
   tree.randomArgs()
   return tree.actionArgPair()

def randomOnTurf(world, entity, whitelist):
   delMatPairs = adjacencyDelMatPairs(world.env, entity.pos)
   moves = whitelistByBlock(delMatPairs, whitelist)
   if len(moves) == 0: return Actions.Pass(), Actions.EmptyArgs()
   ind = np.random.randint(0, len(moves))

   tree = Actions.ActionTree(world, entity, rootVersion=Actions.MoveV2) 
   tree.decideArgs(moves[ind])
   return tree.actionArgPair()

def inWhitelist(env, pos, delta, whitelist):
   r, c = posSum(pos, delta)
   return env[r, c] in whitelist

def whitelistByBlock(delMatPairs, whitelist):
   ret = []
   for deli, mati in delMatPairs:
      if mati in whitelist:
         ret += [deli]
   return ret

def l1(pos1, pos2):
   r1, c1 = pos1
   r2, c2 = pos2
   return abs(r1-r2) + abs(c1-c2)

def sortedByl1(ent, sz, startEnt):
   targs = l1Range(ent, sz, startEnt.pos, startEnt.searchRange)
   targs = sorted(targs, key=lambda targ: l1(startEnt.pos, targ.pos))
   return targs

def l1Range(ent, sz, start, rng):
   R, C = sz, sz
   rs, cs = start
   rt = max(0, rs-rng)
   rb = min(R, rs+rng)
   cl = max(0, cs-rng)
   cr = min(C, cs+rng)
   ret = []
   for r in range(rt, rb):
      for c in range(cl, cr):
         if len(ent[r, c]) > 0:
            ret += ent[r, c]
   return ret

def isAdjacent(pos1, pos2):
   rs, re = pos1
   es, ee = pos2
   return np.logical_xor(abs(re - rs) == 1, abs(ee - es) == 1) 

def posSum(pos1, pos2):
   return pos1[0] + pos2[0], pos1[1] + pos2[1]

def routePath(start, end):
   sr, sc = start
   er, ec = end
   if abs(sc - ec) > abs(sr - er):
      return (0, np.sign(ec - sc))
   return (np.sign(er - sr), 0)

 
def inRange(env, start, targ, rng):
   R, C = env.shape
   rs, cs = start
   rt = max(0, rs-rng)
   rb = min(R, rs+rng)
   cl = max(0, cs-rng)
   cr = min(C, cs+rng)
   return targ in env[rt:rb, cl:cr]

#Fix this
def findNearest(env, start, targ, rng=4 ):
   #Quick check
   rs, ts = start
   if not inRange(env, start, targ, rng):
      return

   #Expensive search
   cur = Queue()
   visited = {}
   cur.push(start)
   while not cur.empty:
      r, c = cur.pop()
      if (r, c) in visited:
         continue

      if env[r, c] == targ:
         return (r, c)

      visited[(r, c)] = 1
      if rs - r < targ:
         cur.push(r+1,  c)
      if cs - c < targ:
         cur.push(r,  c+1)
      if r - rs < targ:
         cur.push(r-1, c) 
      if c - cs < targ:
         cur.push(r, c-1)

   return None
   
class RageClock:
   def __init__(self, ticks):
      self.ticks = ticks

   def tick(self):
      self.ticks -= 1

   def isActive(self):
      return self.ticks > 0

