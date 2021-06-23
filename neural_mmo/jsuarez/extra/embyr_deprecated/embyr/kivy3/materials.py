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

from kivy.graphics import ChangeState

# Map for material attributes to shader
# uniform variables
MATERIAL_TO_SHADER_MAP = {
                          "color": "Ka",
                          "transparency": "Tr",
                          "diffuse": "Kd",
                          "specular": "Ks",
                          "shininess": "Ns",  # specular coefficient
                          }


def set_attribute_to_uniform(attr_name, uniform_var):
    MATERIAL_TO_SHADER_MAP[attr_name] = uniform_var


class Material(ChangeState):

    def __init__(self, map=None, transparency=1.0, color=(1, 1, 1),
                 diffuse=(0, 0, 0), specular=(0, 0, 0),
                 shininess=10.0, **kwargs):
        self.map = map
        super(Material, self).__init__()
        transparency = float(transparency)
        color = tuple(float(c) for c in color)
        diffuse = tuple(float(d) for d in diffuse)
        specular = tuple(float(s) for s in specular)
        shininess = float(shininess)

        # set attribute from locals
        for k, v in locals().items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        if k in MATERIAL_TO_SHADER_MAP:
            uniform_var = MATERIAL_TO_SHADER_MAP[k]
            self.changes[uniform_var] = v
        else:
            if type(v) in [float, int, str, list]:
                self.changes[k] = v
        super(Material, self).__setattr__(k, v)
