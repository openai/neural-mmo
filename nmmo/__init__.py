import os

version = '1.5.3.1'
motd = open(os.path.dirname(__file__) + '/resource/ascii.txt').read().format(version)


from . import scripting
from .lib import material
from .overlay import Overlay, OverlayRegistry
from .io import action
from .io.stimulus import Serialized
from .io.action import Action
from .core import config, agent
from .core.agent import Agent
from .core.env import Env

__all__ = ['Env', 'config', 'scripting', 'agent', 'Agent',
        'Serialized', 'action', 'Action', 'scripting', 'material',
        'Overlay', 'OverlayRegistry']
