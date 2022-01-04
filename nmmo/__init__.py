from .version import __version__

import os
motd = open(os.path.dirname(__file__) + '/resource/ascii.txt').read().format(__version__)

from . import scripting
from .lib import material
from .overlay import Overlay, OverlayRegistry
from .io import action
from .io.stimulus import Serialized
from .io.action import Action
from .core import config, agent
from .core.agent import Agent
from .core.env import Env
from .systems import achievement

__all__ = ['Env', 'config', 'scripting', 'agent', 'Agent',
        'Serialized', 'action', 'Action', 'scripting', 'material',
        'achievement', 'Overlay', 'OverlayRegistry']
