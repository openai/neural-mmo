#Various utilities for managing combat, including hit/damage

import numpy as np

def combatLevel(skills):
   hp = skills.constitution.level
   defense = skills.defense.level
   melee = skills.melee.level
   ranged = skills.ranged.level
   
   base = 0.25*(defense + hp)
   meleeAdjust = 0.65*melee
   rangeAdjust = 0.325*(np.floor(ranged/2)+ranged)
   final = np.floor(base + max(meleeAdjust, rangeAdjust))
   return final

def attack(entity, targ, skill):
   attackLevel  = skill.level
   defenseLevel = targ.skills.defense.level

   attackBonus, strengthBonus = 0, 0
   if entity.isPC:
      if skill.isMelee:
         equip = entity.equipment.melee
      elif skill.isRanged:
         equip = entity.equipment.ranged
         if equip.ammo is not None and equip.ammo <= 0:
            return
      attackBonus, strengthBonus  = equip.attack, equip.strength

   defenseBonus = 0
   if targ.isPC:
      defenseBonus = targ.equipment.armor.defense
 
   dmg = 0
   if isHit(attackLevel, attackBonus, defenseLevel, defenseBonus):
      dmg = damage(attackLevel, strengthBonus)

   if entity.isPC:
      entity.skills.addCombatExp(skill, dmg)
   if entity.isPC:
      targ.skills.addCombatExp(targ.skills.defense, dmg)
   targ.registerHit(entity, dmg)

#Compute maximum damage roll
def maxHit(effectiveLevel, equipmentBonus):
   return np.floor(0.5 + (8+effectiveLevel)*(equipmentBonus+64.0)/640.0)

#Compute maximum attack or defense roll (same formula)
def maxAttackDefense(effectiveLevel, equipmentBonus):
   return effectiveLevel*(equipmentBonus+64)

#Compute hit chance from max attack and defense
def accuracy(atk, dfn):
   if atk > dfn:
      return 1 - (dfn+2) / (2*(atk+1))
   return atk/(2*(dfn+1))

def isHit(attackLevel, attackBonus, defenseLevel, defenseBonus):
   maxAttack  = maxAttackDefense(attackLevel, attackBonus)
   maxDefense = maxAttackDefense(defenseLevel, defenseBonus)
   acc = accuracy(maxAttack, maxDefense)
   return np.random.rand() < acc 

def damage(strengthLevel, strengthBonus):
   mmax = maxHit(strengthLevel, strengthBonus)
   return np.random.randint(1, 1 + mmax)

