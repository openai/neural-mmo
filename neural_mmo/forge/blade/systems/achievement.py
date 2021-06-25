from pdb import set_trace as T

class Tier:
   EASY   = 4
   NORMAL = 10
   HARD   = 25

class Diary:
   def __init__(self, config):
      self.achievements = []

      if config.game_system_enabled('Achievement'):
         self.achievements = [PlayerKills, Equipment, Exploration, Foraging]

      self.achievements = [a(config) for a in self.achievements]

   @property
   def stats(self):
      return [a.stats for a in self.achievements]

   def score(self, aggregate=True):
      score = [a.score for a in self.achievements]
      if score and aggregate:
         return sum(score)
      return score

   def update(self, realm, entity, aggregate=True, dry=False):
      scores = [a.update(realm, entity, dry) for a in self.achievements]
      if scores and aggregate:
         return sum(scores)
      return scores


class Achievement:
   def __init__(self, easy=None, normal=None, hard=None):
      self.progress = 0

      self.easy     = easy
      self.normal   = normal
      self.hard     = hard

   @property
   def stats(self):
      return self.__class__.__name__, self.progress

   @property
   def score(self):
      if self.hard and self.progress >= self.hard:
         return Tier.HARD
      elif self.normal and self.progress >= self.normal:
         return Tier.NORMAL
      elif self.easy and self.progress >= self.easy:
         return Tier.EASY
      return 0

   def update(self, value, dry):
      if value <= self.progress:
         return 0

      old = self.score

      if not dry:
         self.progress = value

      new = self.score

      if old == new:
         return 0

      return new - old
      
class PlayerKills(Achievement):
   def __init__(self, config):
      super().__init__(easy   = config.PLAYER_KILLS_EASY,
                       normal = config.PLAYER_KILLS_NORMAL,
                       hard   = config.PLAYER_KILLS_HARD)

   def update(self, realm, entity, dry):
      return super().update(entity.history.playerKills, dry)

class Equipment(Achievement):
   def __init__(self, config):
      super().__init__(easy   = config.EQUIPMENT_EASY,
                       normal = config.EQUIPMENT_NORMAL,
                       hard   = config.EQUIPMENT_HARD)

   def update(self, realm, entity, dry):
      return super().update(entity.loadout.defense, dry)

class Exploration(Achievement):
   def __init__(self, config):
      super().__init__(easy   = config.EXPLORATION_EASY,
                       normal = config.EXPLORATION_NORMAL,
                       hard   = config.EXPLORATION_HARD)

   def update(self, realm, entity, dry):
      return super().update(entity.history.exploration, dry)

class Foraging(Achievement):
   def __init__(self, config):
      super().__init__(easy   = config.FORAGING_EASY,
                       normal = config.FORAGING_NORMAL,
                       hard   = config.FORAGING_HARD)

   def update(self, realm, entity, dry):
      lvl = (entity.skills.fishing.level + entity.skills.hunting.level)/2.0
      return super().update(lvl, dry)

