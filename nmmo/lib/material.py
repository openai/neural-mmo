from pdb import set_trace as T

from nmmo.systems import item

class Material:
   harvestable = False
   capacity    = 1
   def __init__(self, config):
      pass

   def __eq__(self, mtl):
      return self.index == mtl.index

   def __equals__(self, mtl):
      return self == mtl

class Lava(Material):
   tex   = 'lava'
   index = 0

class Water(Material):
   tex   = 'water'
   index = 1

   def __init__(self, config):
       self.deplete = __class__
       self.respawn  = 1.0

   def harvest(self):
       return droptable.Empty()

class Grass(Material):
   tex   = 'grass'
   index = 2

class Scrub(Material):
   tex = 'scrub'
   index = 3

class Forest(Material):
   tex   = 'forest'
   index = 4

   deplete = Scrub
   def __init__(self, config):
      if config.game_system_enabled('Resource'):
         self.capacity = config.RESOURCE_FOREST_CAPACITY
         self.respawn  = config.RESOURCE_FOREST_RESPAWN

   def harvest(self):
       return droptable.Empty()

class Stone(Material):
   tex   = 'stone'
   index = 5

class Slag(Material):
   tex   = 'slag'
   index = 6

class Ore(Material):
   tex   = 'ore'
   index = 7

   tool    = item.Pickaxe
   deplete = Slag

   def __init__(self, config):
       self.respawn = config.ORE_RESPAWN

   def harvest(self):
       return droptable.Ammunition(item.Scrap)

class Stump(Material):
   tex   = 'stump'
   index = 8

class Tree(Material):
   tex   = 'tree'
   index = 9

   tool    = item.Chisel
   deplete = Stump

   def __init__(self, config):
      if config.game_system_enabled('Resource'):
         self.capacity = config.RESOURCE_TREE_CAPACITY
         self.respawn  = config.RESOURCE_TREE_RESPAWN

   def harvest(self):
       return droptable.Ammunition(item.Shaving)

class Fragment(Material):
   tex   = 'fragment'
   index = 10

class Crystal(Material):
   tex   = 'crystal'
   index = 11

   tool    = item.Arcane
   deplete = Fragment

   def __init__(self, config):
      if config.game_system_enabled('Resource'):
         self.capacity = config.RESOURCE_CRYSTAL_CAPACITY
         self.respawn  = config.RESOURCE_CRYSTAL_RESPAWN

   def harvest(self):
       return droptable.Ammunition(item.Shard)

class Weeds(Material):
   tex   = 'weeds'
   index = 12

class Herb(Material):
   tex   = 'herb'
   index = 13

   tool    = item.Gloves
   deplete = Weeds

   def __init__(self, config):
      if config.game_system_enabled('Resource'):
         self.capacity = config.RESOURCE_HERB_CAPACITY
         self.respawn  = config.RESOURCE_HERB_RESPAWN

   def harvest(self):
       return droptable.Ammunition(item.Poultice)

class Ocean(Material):
   tex   = 'ocean'
   index = 14

class Fish(Material):
   tex   = 'fish'
   index = 15

   tool    = item.Rod
   deplete = Ocean

   def __init__(self, config):
      if config.game_system_enabled('Resource'):
         self.capacity = config.RESOURCE_FISH_CAPACITY
         self.respawn  = config.RESOURCE_FISH_RESPAWN

   def harvest(self):
       return droptable.Ammunition(item.Ration)

class Meta(type):
   def __init__(self, name, bases, dict):
      self.indices = {mtl.index for mtl in self.materials}

   def __iter__(self):
      yield from self.materials

   def __contains__(self, mtl):
      if isinstance(mtl, Material):
         mtl = type(mtl)
      if isinstance(mtl, type):
         return mtl in self.materials
      return mtl in self.indices

class All(metaclass=Meta):
   '''List of all materials'''
   materials = {
      Lava, Water, Grass, Scrub, Forest,
      Stone, Slag, Ore, Stump, Tree,
      Fragment, Crystal, Weeds, Herb, Ocean, Fish}

class Impassible(metaclass=Meta):
   '''Materials that agents cannot walk through'''
   materials = {Lava, Water, Stone, Ocean, Fish}

class Habitable(metaclass=Meta):
   '''Materials that agents cannot walk on'''
   materials = {Grass, Scrub, Forest, Ore, Tree, Crystal, Weeds, Herb}

class Harvestable(metaclass=Meta):
   '''Materials that agents can harvest'''
   materials = {Water, Forest, Ore, Tree, Crystal, Herb, Fish}
