from .version import __version__

import os
motd = '''      ___           ___           ___           ___
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
from .lib import material
from .lib.rating import OpenSkillRating
from .overlay import Overlay, OverlayRegistry
from .io import action
from .io.stimulus import Serialized
from .io.action import Action
from .core import config, agent
from .core.agent import Agent
from .core.env import Env
from .systems.achievement import Task
from .core.terrain import MapGenerator, Terrain

__all__ = ['Env', 'config', 'scripting', 'agent', 'Agent', 'MapGenerator', 'Terrain',
        'Serialized', 'action', 'Action', 'scripting', 'material',
        'Task', 'Overlay', 'OverlayRegistry']

