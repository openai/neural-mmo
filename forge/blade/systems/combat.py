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

def attack(entity, targ, skillFn):
   config      = entity.config
   entitySkill = skillFn(entity)
   targetSkill = skillFn(targ)

   targetDefense = targ.skills.defense.level + targ.loadout.defense

   roll = np.random.randint(1, config.DICE_SIDES+1)
   dc   = accuracy(config, entitySkill.level, targetSkill.level, targetDefense)
   crit = roll == config.DICE_SIDES

   dmg = 1 #Chip dmg on a miss
   if roll >= dc or crit:
      dmg = damage(entitySkill.__class__, entitySkill.level)
      
   entity.applyDamage(dmg, entitySkill.__class__.__name__.lower())
   targ.receiveDamage(entity, dmg)
   return dmg

#Compute maximum damage roll
def damage(skill, level):
   if skill == Skill.Melee:
      return np.floor(5 + level * 45 / 99)
   if skill == Skill.Range:
      return np.floor(3 + level * 32 / 99)
   if skill == Skill.Mage:
      return np.floor(1 + level * 24 / 99)

#Compute maximum attack or defense roll (same formula)
#Max attack 198 - min def 1 = 197. Max 198 - max 198 = 0
#REMOVE FACTOR OF 2 FROM ATTACK AFTER IMPLEMENTING WEAPONS
def accuracy(config, entAtk, targAtk, targDef):
   alpha   = config.DEFENSE_WEIGHT

   attack  = entAtk
   defense = alpha*targDef + (1-alpha)*targAtk
   dc      = defense - attack + config.DICE_SIDES//2

   return dc

def danger(config, pos, full=False):
   cent = config.TERRAIN_SIZE // 2

   #Distance from center
   R   = int(abs(pos[0] - cent + 0.5))
   C   = int(abs(pos[1] - cent + 0.5))
   mag = max(R, C)

   #Distance from border terrain to center
   if config.INVERT_WILDERNESS:
      R   = cent - R - config.TERRAIN_BORDER
      C   = cent - C - config.TERRAIN_BORDER
      mag = min(R, C)

   #Normalize
   norm = mag / (cent - config.TERRAIN_BORDER)

   if full:
      return norm, mag
   return norm
      
def wilderness(config, pos):
   norm, raw = danger(config, pos, full=True)
   wild      = int(100 * norm) - 1

   if not config.WILDERNESS:
      return 99
   if wild < config.WILDERNESS_MIN:
      return -1
   if wild > config.WILDERNESS_MAX:
      return config.WILDERNESS_MAX

   return wild
