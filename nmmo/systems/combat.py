#Various utilities for managing combat, including hit/damage

from pdb import set_trace as T

import numpy as np
from nmmo.systems import skill as Skill

def level(skills):
    melee = skills.melee.level.val
    ranged = skills.range.level.val
    mage   = skills.mage.level.val

    return max(melee, ranged, mage)

def damage_multiplier(config, skill, targ):
    skills = [targ.skills.melee, targ.skills.range, targ.skills.mage]
    idx    = np.argmax([s.level for s in skills])
    targ   = skills[idx]

    if type(targ) == skill.weakness:
        return config.DAMAGE_MULTIPLIER

    return 1.0

def attack(entity, targ, skillFn):
    config     = entity.config
    skill      = skillFn(entity)
    skill_type = type(skill)

    # Note: the below damage calculation only holds for ammo
    # granting a bonus to a single combat style
    ammunition = entity.equipment.ammunition
    if skill_type == Skill.Melee:
        offense = entity.equipment.total(lambda e: e.melee_attack)
        defense = entity.equipment.total(lambda e: e.melee_defense)
        #if type(ammunition) == Item.Scrap:
        #    ammunition.fire(entity)
    elif skill_type == Skill.Range:
        offense = entity.equipment.total(lambda e: e.range_attack)
        defense = entity.equipment.total(lambda e: e.range_defense)
        #if type(ammunition) == Item.Shaving:
        #    ammunition.fire(entity)
    elif skill_type == Skill.Mage:
        offense = entity.equipment.total(lambda e: e.mage_attack)
        defense = entity.equipment.total(lambda e: e.mage_defense)
        #if type(ammunition) == Item.Shard:
        #    ammunition.fire(entity)
    elif __debug__:
        assert False, 'Attack skill must be Melee, Range, or Mage'

    #Total damage calculation
    damage = config.DAMAGE_BASE + offense - defense

    entity.applyDamage(damage, skill.__class__.__name__.lower())
    targ.receiveDamage(entity, damage)

    return damage

def danger(config, pos, full=False):
   border = config.TERRAIN_BORDER
   center = config.TERRAIN_CENTER
   r, c   = pos
  
   #Distance from border
   rDist  = min(r - border, center + border - r - 1)
   cDist  = min(c - border, center + border - c - 1)
   dist   = min(rDist, cDist)
   norm   = 2 * dist / center

   if full:
      return norm, mag

   return norm

def spawn(config, dnger):
    border = config.TERRAIN_BORDER
    center = config.TERRAIN_CENTER
    mid    = center // 2

    dist       = dnger * center / 2
    max_offset = mid - dist
    offset     = mid + border + np.random.randint(-max_offset, max_offset)

    rng = np.random.rand()
    if rng < 0.25:
        r = border + dist
        c = offset
    elif rng < 0.5:
        r = border + center - dist - 1
        c = offset
    elif rng < 0.75:
        c = border + dist
        r = offset
    else:
        c = border + center - dist - 1
        r = offset

    if __debug__:
        assert dnger == danger(config, (r,c)), 'Agent spawned at incorrect radius'

    r = int(r)
    c = int(c)

    return r, c
