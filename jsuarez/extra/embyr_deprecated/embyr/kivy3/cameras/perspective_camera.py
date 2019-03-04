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
Perspective camera module
=============

Implements perspective camera.
"""

__all__ = ('PerspectiveCamera', )

from kivy.properties import NumericProperty
from kivy.graphics.transformation import Matrix
from .camera import Camera


class PerspectiveCamera(Camera):
    """
    Implementation of the perspective camera.
    """

    aspect = NumericProperty()

    def __init__(self, fov, aspect, near, far, **kw):

        super(PerspectiveCamera, self).__init__(**kw)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.update_projection_matrix()
        self.bind(aspect=self._on_aspect)

    def _on_aspect(self, inst, value):
        self.update_projection_matrix()
        self.update()

    def update_projection_matrix(self):
        m = Matrix()
        m.perspective(self.fov * 0.5, self.aspect, self.near, self.far)
        self.projection_matrix = m
