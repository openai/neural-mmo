from typing import List

#from forge.blade.core import material
from nmmo.systems import ai

import math

import numpy as np


def map_to_rewards(tiles, entity) -> List[List[float]]:
   lava_reward = stone_reward = water_reward = float('-inf')
   forest_reward = 1.0 + math.pow(
      (1 - entity.resources.food.val / entity.resources.food.max) * 15.0,
      1.25)
   scrub_reward = 1.0
   around_water_reward = 1.0 + math.pow(
      (1 - entity.resources.water.val / entity.resources.water.max) * 15.0,
      1.25)

   reward_matrix = np.full((len(tiles), len(tiles[0])), 0.0)

   for line in range(len(tiles)):
      tile_line = tiles[line]
      for column in range(len(tile_line)):
         tile_val = tile_line[column].state.tex
         if tile_val == 'lava':
            reward_matrix[line][column] += lava_reward

         if tile_val == 'stone':
            reward_matrix[line][column] += stone_reward

         if tile_val == 'forest':
            reward_matrix[line][column] += forest_reward

         if tile_val == 'water':
            reward_matrix[line][column] += water_reward

         #TODO: Make these comparisons work off of the water Enum type
         #instead of string compare
         if 'water' in ai.utils.adjacentMats(tiles, (line, column)):
            reward_matrix[line][column] += around_water_reward

         if tile_val == 'scrub':
            reward_matrix[line][column] += scrub_reward

   return reward_matrix


def compute_values(reward_matrix: List[List[float]]) -> List[List[float]]:
   gamma_factor = 0.8  # look ahead ∈ [0, 1]
   max_delta = 0.01  # maximum allowed approximation

   value_matrix = np.full((len(reward_matrix), len(reward_matrix[0])), 0.0)

   delta = float('inf')
   while delta > max_delta:
      old_value_matrix = np.copy(value_matrix)
      for line in range(len(reward_matrix)):
         for column in range(len(reward_matrix[0])):
            reward = reward_matrix[line][column]
            value_matrix[line][
               column] = reward + gamma_factor * max_value_around(
               (line, column), value_matrix)

      delta = np.amax(
         np.abs(np.subtract(old_value_matrix, value_matrix)))
   return value_matrix


def values_around(position: (int, int), value_matrix: List[List[float]]) -> (
        float, float, float, float):
   line, column = position

   if line - 1 >= 0:
      top_value = value_matrix[line - 1][column]
   else:
      top_value = float('-inf')

   if line + 1 < len(value_matrix):
      bottom_value = value_matrix[line + 1][column]
   else:
      bottom_value = float('-inf')

   if column - 1 >= 0:
      left_value = value_matrix[line][column - 1]
   else:
      left_value = float('-inf')

   if column + 1 < len(value_matrix[0]):
      right_value = value_matrix[line][column + 1]
   else:
      right_value = float('-inf')

   return top_value, bottom_value, left_value, right_value


def max_value_around(position: (int, int),
                     value_matrix: List[List[float]]) -> float:
   return max(values_around(position, value_matrix))


def max_value_position_around(position: (int, int),
                              value_matrix: List[List[float]]) -> (int, int):
   line, column = position
   top_value, bottom_value, left_value, right_value = values_around(position,
                                                                    value_matrix)

   max_value = max(top_value, bottom_value, left_value, right_value)

   if max_value == top_value:
      return line - 1, column
   elif max_value == bottom_value:
      return line + 1, column
   elif max_value == left_value:
      return line, column - 1
   elif max_value == right_value:
      return line, column + 1


def max_value_direction_around(position: (int, int),
                               value_matrix: List[List[float]]) -> (int, int):
   top_value, bottom_value, left_value, right_value = values_around(position,
                                                                    value_matrix)

   max_value = max(top_value, bottom_value, left_value, right_value)

   if max_value == top_value:
      return -1, 0
   elif max_value == bottom_value:
      return 1, 0
   elif max_value == left_value:
      return 0, -1
   elif max_value == right_value:
      return 0, 1
