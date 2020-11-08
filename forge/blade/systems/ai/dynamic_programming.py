from typing import List

from forge.blade.lib.enums import Material
from forge.blade.systems import ai

import math


def map_to_rewards(tiles, entity) -> List[List[float]]:
   default_reward = -1.0  # grass & co
   lava_reward = float('-inf')
   stone_reward = float('-inf')
   forest_reward = math.pow(
      (1 - entity.resources.food._val / entity.resources.food._max) * 15,
      1.25)
   scrub_reward = 0.0
   water_reward = float('-inf')
   around_water_reward = math.pow(
      (1 - entity.resources.water._val / entity.resources.water._max) * 15,
      1.25)

   reward_matrix = [[default_reward for x in range(len(tiles[0]))] for y in
                    range(len(tiles))]

   for line in range(len(tiles)):
      tile_line = tiles[line]
      for column in range(len(tile_line)):
         tile_val = tile_line[column].state.tex
         if tile_val == 'lava':
            reward_matrix[line][column] = lava_reward
         elif tile_val == 'stone':
            reward_matrix[line][column] = stone_reward
         elif tile_val == 'forest':
            reward_matrix[line][column] = forest_reward
         elif tile_val == 'water':
            reward_matrix[line][column] = water_reward
         elif Material.WATER.value in ai.utils.adjacentMats(tiles,
                                                            (line, column)):
            reward_matrix[line][column] = around_water_reward
         elif tile_val == 'scrub':
            reward_matrix[line][column] = scrub_reward

   return reward_matrix


def compute_values(reward_matrix: List[List[float]]) -> List[List[float]]:
   gamma_factor = 0.8  # look ahead ∈ [0, 1]
   max_delta = 0.01  # maximum allowed approximation
   default_reward = -1.0  # grass & co

   value_matrix = [[default_reward for x in range(len(reward_matrix[0]))] for
                   y in range(len(reward_matrix))]

   delta = float('inf')
   while delta > max_delta:
      delta = float('-inf')
      for line in range(len(reward_matrix)):
         for column in range(len(reward_matrix[0])):
            reward = reward_matrix[line][column]
            old_value = value_matrix[line][column]

            value_matrix[line][
               column] = reward + gamma_factor * max_value_around(
               (line, column), value_matrix)

            delta = max(delta, abs(old_value - value_matrix[line][column]))

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
