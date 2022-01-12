from pdb import set_trace as T

from typing import Callable
from dataclasses import dataclass


@dataclass
class Task:
    condition: Callable
    target: float = None
    reward: float = 0


class Diary:
   def __init__(self, tasks):
      self.achievements = []
      for task in tasks:
         self.achievements.append(Achievement(task.condition, task.target, task.reward))

   @property
   def completed(self):
      return sum(a.completed for a in self.achievements)

   @property
   def cumulative_reward(self, aggregate=True):
      return sum(a.reward * a.completed for a in self.achievements)

   def update(self, realm, entity):
      return {a.name: a.update(realm, entity) for a in self.achievements}


class Achievement:
   def __init__(self, condition, target, reward):
      self.completed = False

      self.condition = condition
      self.target    = target
      self.reward    = reward

   @property
   def name(self):
      return '{}_{}'.format(self.condition.__name__, self.target)

   def update(self, realm, entity):
      if self.completed:
         return 0

      metric = self.condition(realm, entity)

      if metric >= self.target:
         self.completed = True
         return self.reward

      return 0
