
"""
The MIT License (MIT)

Copyright (c) 2013 Niko Skrypnik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Renderer class
=============

Unlike of THREE.js we may provide only one renderer which is the
Kivy widget and uses Kivy canvas and FBO concept for drawing graphics.
You may use this class as usual widget and place it wherever you need
in your application
"""

import os
import kivy3


from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics.fbo import Fbo
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST
from kivy.graphics import Callback, PushMatrix, PopMatrix, \
                          Rectangle, Canvas, UpdateNormalMatrix


kivy3_path = os.path.abspath(os.path.dirname(kivy3.__file__))


class RendererError(Exception):
    pass


class Renderer(Widget):

    def __init__(self, **kw):
        self.shader_file = kw.pop("shader_file", None)
        self.canvas = Canvas()
        super(Renderer, self).__init__(**kw)

        with self.canvas:
            self._viewport = Rectangle(size=self.size, pos=self.pos)
            self.fbo = Fbo(size=self.size,
                           with_depthbuffer=True, compute_normal_mat=True,
                           clear_color=(0., 0., 0., 0.))
        self._config_fbo()
        self.texture = self.fbo.texture
        self.camera = None
        self.scene = None

    def _config_fbo(self):
        # set shader file here
        self.fbo.shader.source = self.shader_file or \
            os.path.join(kivy3_path, "default.glsl")
        with self.fbo:
            Callback(self._setup_gl_context)
            PushMatrix()
            # instructions set for all instructions
            self._instructions = InstructionGroup()
            PopMatrix()
            Callback(self._reset_gl_context)

    def _setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)
        self.fbo.clear_buffer()

    def _reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def render(self, scene, camera):
        self.scene = scene
        self.camera = camera
        self.camera.bind_to(self)
        self._instructions.add(scene.as_instructions())
        Clock.schedule_once(self._update_matrices, -1)

    def add(self, obj):
        self._instructions.add(obj.as_instructions())

    def on_size(self, instance, value):
        self.fbo.size = value
        self._viewport.texture = self.fbo.texture
        self._viewport.size = value
        self._viewport.pos = self.pos
        self._update_matrices()

    def on_pos(self, instance, value):
        self._viewport.pos = self.pos
        self._update_matrices()

    def on_texture(self, instance, value):
        self._viewport.texture = value

    def _update_matrices(self, dt=None):
        if self.camera:
            self.fbo['projection_mat'] = self.camera.projection_matrix
            self.fbo['modelview_mat'] = self.camera.modelview_matrix
        else:
            raise RendererError("Camera is not defined for renderer")

    def set_clear_color(self, color):
        self.fbo.clear_color = color
