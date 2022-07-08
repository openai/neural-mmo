from .version import __version__

import os
motd = r'''      ___           ___           ___           ___
     /__/\         /__/\         /__/\         /  /\      Version {:<8}
     \  \:\       |  |::\       |  |::\       /  /::\ 
      \  \:\      |  |:|:\      |  |:|:\     /  /:/\:\    An open source
  _____\__\:\   __|__|:|\:\   __|__|:|\:\   /  /:/  \:\   project originally
 /__/::::::::\ /__/::::| \:\ /__/::::| \:\ /__/:/ \__\:\  founded by Joseph Suarez
 \  \:\~~\~~\/ \  \:\~~\__\/ \  \:\~~\__\/ \  \:\ /  /:/  and formalized at OpenAI
  \  \:\  ~~~   \  \:\        \  \:\        \  \:\  /:/
   \  \:\        \  \:\        \  \:\        \  \:\/:/    Now developed and
    \  \:\        \  \:\        \  \:\        \  \::/     maintained at MIT in
     \__\/         \__\/         \__\/         \__\/      Phillip Isola's lab '''.format(__version__)

from . import scripting
from .lib import material, spawn
from .overlay import Overlay, OverlayRegistry
from .io import action
from .io.stimulus import Serialized
from .io.action import Action
from .core import config, agent
from .core.agent import Agent
from .core.env import Env, Replay
from . import scripting, emulation, integrations
from .systems.achievement import Task
from .core.terrain import MapGenerator, Terrain

__all__ = ['Env', 'config', 'scripting', 'emulation', 'integrations', 'agent', 'Agent', 'MapGenerator', 'Terrain',
        'Serialized', 'action', 'Action', 'scripting', 'material', 'spawn',
        'Task', 'Overlay', 'OverlayRegistry', 'Replay']

try:
    import openskill
    from .lib.rating import OpenSkillRating
    __all__.append('OpenSkillRating')
except:
    print('Warning: OpenSkill not installed. Ignore if you do not need this feature')
