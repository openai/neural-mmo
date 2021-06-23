from pdb import set_trace as T
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color

class CornerRectangleWidget(Widget):
    def __init__(self, **kwargs):
        super(CornerRectangleWidget, self).__init__(**kwargs)

        with self.canvas:
            Color(1, 0, 0, 1)  # set the colour to red
            self.rect = Rectangle(pos=self.center,
                                  size=(self.width/2.,
                                        self.height/2.))

class MyApp(App):
    def build(self):
        game = CornerRectangleWidget()
        #Clock.schedule_interval(game.update, 1.0/1000.0)
        return game

if __name__ == '__main__':
    MyApp().run()
