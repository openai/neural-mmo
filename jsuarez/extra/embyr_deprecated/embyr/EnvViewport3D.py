from pdb import set_trace as T
import numpy as np
import time
from scipy.misc import imresize, imsave
from enum import Enum

import pygame
from pygame import Surface

from forge.embyr import embyr
from forge.embyr import utils as renderutils
from forge.embyr import render
from forge.embyr.texture import TextureInitializer
from forge.blade.lib.enums import Neon, Color256, Defaults
from forge.blade.action.v2 import Attack
from forge.blade.action import action

from pdb import set_trace as T
import numpy as np

import os
import kivy3
from kivy.app import App
from kivy3 import Scene, Renderer, PerspectiveCamera
from kivy3.loaders import OBJMTLLoader
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config
from kivy.graphics import opengl as gl
from kivy.graphics import Mesh as KivyMesh
from kivy3.core.object3d import Object3D
from kivy3.materials import Material
from kivy.core.image import Image
from copy import deepcopy
from forge.embyr.embyr import Application as KivyApp
import pywavefront as pywave
import pytmx
from forge.blade.lib import enums
from forge.embyr.transform import Transform
from forge.embyr import embyr3D

root = 'forge/embyr/'
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
shaderf = 'tex/default.glsl'
pi = 3.14159265
NANIM = 4

class EnvViewport3D(embyr3D.Widget):
   def __init__(self, root, realm, **kwargs):
      super().__init__(root)
      self.root = 'forge/embyr/'
      self.rs = embyr3D.RS(root, **kwargs)
      self.setupScene()

   def setupScene(self):
        self.map = embyr3D.Map(self.rs)

        obj = embyr3D.OBJLoader.load(self.root + 'tex/nn.obj')

        ent = embyr3D.Ent()
        ent.pos.x = 40
        ent.pos.y = 10
        ent.pos.z = 40
        self.vecent = ent
        self.rs.add(ent)

        ent = embyr3D.Ent()
        ent.pos.x = 8
        ent.pos.y = 20
        ent.pos.z = 8
        self.cament = ent
        self.rs.add(ent)

   def render(self, dt):
        #self.client.render(dt)
        #self.step()
        x, y, z = self.rs.update(dt)
        self.vecent.update_pos(x, 3, z)

        '''
        desciples = sorted(self.realm.desciples.items())
        if len(desciples) == 0:
            return
        ent = desciples[0][1]
        z, x = 32, 32 #ent.server.pos
        self.ent.update_pos(x, self.ent.pos.y, z)
        '''

   def refresh(self, trans, iso):
      self.iso = iso
      mmap = self.map.refresh(trans, self.iso)
      ent  = self.ent.refresh(trans, self.iso)

      self.blit(mmap, (0, 0))
      self.blit(ent, (0, 0))

      self.flip()
      return self.surf
