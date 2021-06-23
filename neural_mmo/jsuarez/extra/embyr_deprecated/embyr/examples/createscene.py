
import os
import kivy3
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy3 import Mesh, Material
from kivy3 import Scene, Renderer, PerspectiveCamera
from kivy3.extras.geometries import BoxGeometry


class SceneApp(App):

    def build(self):
        root = FloatLayout()

        self.renderer = Renderer()
        self.renderer.set_clear_color((.2, .2, .2, 1.))
        scene = Scene()
        geometry = BoxGeometry(1, 1, 1)
        material = Material(color=(0., 0., 1.), diffuse=(1., 1., 0.),
                            specular=(.35, .35, .35))
        self.cube = Mesh(geometry, material)
        self.cube.pos.z = -5
        camera = PerspectiveCamera(75, 0.3, 1, 1000)

        scene.add(self.cube)
        self.renderer.render(scene, camera)

        root.add_widget(self.renderer)
        Clock.schedule_interval(self._rotate_cube, 1 / 20)
        self.renderer.bind(size=self._adjust_aspect)

        return root

    def _adjust_aspect(self, inst, val):
        rsize = self.renderer.size
        aspect = rsize[0] / float(rsize[1])
        self.renderer.camera.aspect = aspect

    def _rotate_cube(self, dt):
        self.cube.rotation.x += 1
        self.cube.rotation.y += 1
        self.cube.rotation.z += 1


if __name__ == '__main__':
    SceneApp().run()
