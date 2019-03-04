from pdb import set_trace as T
import numpy as np

import os
import kivy3
from kivy.app import App
from kivy3 import Scene, Renderer, PerspectiveCamera
from kivy3.loaders import OBJMTLLoader
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
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
from forge.embyr.client import Client, Canvas
from forge.embyr.EnvViewport3D import EnvViewport3D

root = 'forge/embyr/'
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
shaderf = 'tex/default.glsl'
pi = 3.14159265
NANIM = 4

class Application(KivyApp):
    def __init__(self, size, realm, step, conf):
        super().__init__(size)
        self.W, self.H = size
        self.appSize = size
        self.root = 'forge/embyr/'
        self.title = 'Projekt: Godsword'
        self.blocks = dict((mat.value.tex, mat.value) 
              for mat in enums.Material)

        self.realm = realm
        self.step = step
        self.conf = conf

    def build(self):
        root = FloatLayout()
        W, H, side = self.W, self.H, 256
        dims = (self.W-side, self.H-side, side)
        #self.env1 = EnvViewport3D(root, self.realm, (W, W))
        #self.env2 = EnvViewport3D(root, self.realm, (W, W))
        canvas = Canvas(self.appSize, root, self.realm, dims, self.conf)
        #self.client = Client(canvas, self.appSize, self.realm,
        #      self.step, dims, NANIM)
        self.canvas = canvas
        
        root.add_widget(canvas)

        #root.add_widget(self.env1)
        #root.add_widget(self.env2)

        self.loop(self.update)
        return root

    def update(self, dt):
        self.step()
        #self.env1.render()
        #self.env2.render()
        self.canvas.render(dt)

        '''
        desciples = sorted(self.realm.desciples.items())
        if len(desciples) == 0:
            return
        ent = desciples[0][1]
        z, x = 32, 32 #ent.server.pos
        self.ent.update_pos(x, self.ent.pos.y, z)
        '''
 
