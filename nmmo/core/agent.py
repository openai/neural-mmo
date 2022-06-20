from pdb import set_trace as T

from nmmo.lib import colors

class Agent:
    scripted = False
    policy   = 'Neural'

    color    = colors.Neon.CYAN
    pop      = 0

    def __init__(self, config, idx):
       '''Base class for agents

       Args:
          config: A Config object
          idx: Unique AgentID int
       '''
       self.config = config
       self.iden   = idx
       self.pop    = Agent.pop

    def __call__(self, obs):
       '''Used by scripted agents to compute actions. Override in subclasses.

       Args:
           obs: Agent observation provided by the environment
       '''
       pass

class Random(Agent):
    '''Moves randomly, including bumping into things and falling into lava'''
    def __call__(self, obs):
        return {Action.Move: {Action.Direction: rand.choice(Action.Direction.edges)}}
