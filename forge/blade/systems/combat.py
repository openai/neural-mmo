#Various utilities for managing combat, including hit/damage

from pdb import set_trace as T

import numpy as np
from forge.blade.systems import skill as Skill

def level(skills):
   hp = skills.constitution.level
   defense = skills.defense.level
   melee = skills.melee.level
   ranged = skills.range.level
   mage   = skills.mage.level
   
   base = 0.25*(defense + hp)
   final = np.floor(base + 0.5*max(melee, ranged, mage))
   return final

def attack(entity, targ, skill):
   attackLevel  = skill.level
   defenseLevel = targ.skills.defense.level + targ.loadout.defense
   skill = skill.__class__.__name__

   #1 dmg on a miss, max hit on success
   dmg = 1
   if np.random.rand() < accuracy(attackLevel, defenseLevel):
      dmg = damage(skill, attackLevel)
      
   entity.applyDamage(dmg, skill.lower())
   targ.receiveDamage(entity, dmg)
   return dmg

#Compute maximum damage roll
def damage(skill, level):
   if skill == 'Melee':
      return np.floor(5 + level * 45 / 99)
   if skill == 'Range':
      return np.floor(3 + level * 32 / 99)
   if skill == 'Mage':
      return np.floor(1 + level * 24 / 99)

#Compute maximum attack or defense roll (same formula)
#Max attack 198 - min def 1 = 197. Max 198 - max 198 = 0
#REMOVE FACTOR OF 2 FROM ATTACK AFTER IMPLEMENTING WEAPONS
def accuracy(attkLevel, defLevel):
   return 0.5 + (2*attkLevel - defLevel) / 197

def wilderness(config, pos):
   cent = config.TERRAIN_SIZE // 2

   #Distance from border terrain to center
   R = cent - abs(pos[0] - cent) - config.TERRAIN_BORDER
   C = cent - abs(pos[1] - cent) - config.TERRAIN_BORDER
  
   #Normalize
   dist = min(R, C) / (cent - config.TERRAIN_BORDER)
   wild = int(100 * dist) - 1

   #Convert to distance from center?
   if config.INVERT_WILDERNESS:
      R = 1 - R
      C = 1 - C

   return wild
