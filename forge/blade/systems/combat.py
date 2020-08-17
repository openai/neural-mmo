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
   rCent = config.R//2
   cCent = config.C//2

   R = abs(pos[0] - rCent)
   C = abs(pos[1] - cCent)

   #Circle crop with 0 starting at 10 squares from
   #center and increasing one level every 5 tiles
   wild = config.R//2 - max(R, C)
   wild = (wild - 17) // 5

   wild = int(np.clip(wild, -1, 99))

   return wild
