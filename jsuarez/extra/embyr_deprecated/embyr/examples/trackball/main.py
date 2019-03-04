
import os
import math
from kivy.app import App
from kivy.clock import Clock
from kivy3 import Scene, Renderer, PerspectiveCamera
from kivy3.loaders import OBJLoader
from kivy.uix.floatlayout import FloatLayout

# Resources pathes
_this_path = os.path.dirname(os.path.realpath(__file__))
obj_file = os.path.join(_this_path, "./MQ-27.obj")


class ObjectTrackball(FloatLayout):

    def __init__(self, camera, radius, *args, **kw):
        super(ObjectTrackball, self).__init__(*args, **kw)
        self.camera = camera
        self.radius = radius
        self.phi = 90
        self.theta = 0
        self._touches = []
        self.camera.pos.z = radius
        camera.look_at((0, 0, 0))

    def define_rotate_angle(self, touch):
        theta_angle = (touch.dx / self.width) * -360
        phi_angle = -1 * (touch.dy / self.height) * 360
        return phi_angle, theta_angle

    def on_touch_down(self, touch):
        touch.grab(self)
        self._touches.append(touch)

    def on_touch_up(self, touch):
        touch.ungrab(self)
        self._touches.remove(touch)

    def on_touch_move(self, touch):
        if touch in self._touches and touch.grab_current == self:
            if len(self._touches) == 1:
                self.do_rotate(touch)
            elif len(self._touches) == 2:
                pass

    def do_rotate(self, touch):
        d_phi, d_theta = self.define_rotate_angle(touch)
        self.phi += d_phi
        self.theta += d_theta

        _phi = math.radians(self.phi)
        _theta = math.radians(self.theta)
        z = self.radius * math.cos(_theta) * math.sin(_phi)
        x = self.radius * math.sin(_theta) * math.sin(_phi)
        y = self.radius * math.cos(_phi)
        self.camera.pos = x, y, z


class MainApp(App):

    def build(self):
        self.renderer = Renderer()
        scene = Scene()
        camera = PerspectiveCamera(15, 1, 1, 1000)
        loader = OBJLoader()
        obj = loader.load(obj_file)
        self.obj3d = obj
        self.camera = camera
        root = ObjectTrackball(camera, 1500)

        scene.add(obj)

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
