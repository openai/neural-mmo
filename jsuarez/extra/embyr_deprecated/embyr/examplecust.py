"""
The MIT License (MIT)

Copyright (c) 2014 Niko Skrypnik

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
Same as example1 but with using default shader file and colorizing
of the objects
"""

import os
#import kivy3
from kivy.app import App
from kivy3 import Scene, Renderer, PerspectiveCamera
from kivy3.loaders import OBJLoader
from kivy.uix.floatlayout import FloatLayout
from pdb import set_trace as T


class MainApp(App):

    def build(self):
        root = FloatLayout()
        self.renderer = Renderer(shader_file='examples/simple.glsl')
        scene = Scene()
        camera = PerspectiveCamera(30, 1, 1, 1000)
        # load obj file
        loader = OBJLoader()
        obj_path = os.path.join(os.path.dirname(__file__), "cube.obj")
        obj = loader.load(obj_path)
        cube = obj.children[0]
        #scene.add(*obj.children)
        #for obj in scene.children:

        scene.add(cube)
        cube.pos.z = -20
        cube.rot.y = -45
        cube.rot.x = 45
        cube.material.specular = .35, .35, .35

        # set colors to 3d objects
        scene.children[0].material.color = 0., .7, 0.  # green
        scene.children[0].material.diffuse = 0., .7, 0.  # green

        self.renderer.render(scene, camera)
        root.add_widget(self.renderer)
        self.renderer.bind(size=self._adjust_aspect)
        return root

    def _adjust_aspect(self, inst, val):
        rsize = self.renderer.size
        aspect = rsize[0] / float(rsize[1])
        self.renderer.camera.aspect = aspect


if __name__ == '__main__':
    MainApp().run()
