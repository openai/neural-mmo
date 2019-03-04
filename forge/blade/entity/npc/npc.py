from sim.entity.Entity import Entity
from sim.lib import Enums, AI
from sim.modules import DropTable


#NPC
class NPC(Entity):
   def __init__(self, pos):
      super().__init__(pos)
      self.expOnKill = 10
      self.drops = DropTable.DropTable()

   #NPCs have direct access to the world
   def act(self, world):
      action, args = self.decide(world)
      if type(args) == Actions.EmptyArgs:
         args = []
      action(world, self, *args)
      self.lastAction = action

   def yieldDrops(self):
      self.lastAttacker.receiveDrops(self.drops.roll())

#Wanders aimlessly. Ignores attacks
class Passive(NPC):
   def __init__(self, pos):
      super().__init__(pos)

   def decide(self, env):
      return Actions.move4(), EmptyArgs

#Adds basic agressive behavior
class PassiveAgressive(NPC):
   def __init__(self, pos, rageTime=5):
      super().__init__(pos)
      self.rageClock = AI.RageClock(0)
      self.rageTime = rageTime

   def decide(self, world):
      if not self.rageClock.isActive():
         return Actions.move4(), EmptyArgs

      self.rageClock.tick()
      action = AI.searchAndDestroy, (world,
            self.pos, Enums.Entity.NEURAL)
      return action, []

   def registerHit(self, attacker, dmg):
      super(PassiveAgressive, self).registerHit(attacker, dmg)
      self.rageClock = AI.RageClock(self.rageTime)


