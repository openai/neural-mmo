from pdb import set_trace as T

import inspect
from collections import defaultdict

from neural_mmo.forge.blade.lib.enums import Neon, Solid

class Iterable(type):
   def __iter__(cls):
      stack = list(cls.__dict__.items())
      while len(stack) > 0:
         name, attr = stack.pop()
         if name.startswith('__'):
            continue
         if not inspect.isclass(attr) and type(attr) == classmethod:
            continue
         yield name, attr
  
class I(metaclass=Iterable):
   pass

class Theme(metaclass=Iterable):
   @classmethod
   def dict(cls):
     data = defaultdict(dict)

     for group, c in Theme:
        for attr, val in c:
           data[group][attr] = val

     for group, c in cls:
        for attr, val in c:
           data[group][attr] = val

     return dict(data)

   class Figure(I):
      outline_line_cap = "square"
      outline_line_width    = 2.0

   class Grid(I):
      grid_line_dash  = [6, 4]
      grid_line_width = 1.1

   class Title(I):
      text_font_size = "20pt"
      align          = "center"

   class Axis(I):
      axis_label_text_font_size  = "16pt"
      major_label_text_font_size = "14pt"
      major_tick_line_cap   = None
      minor_tick_line_cap   = None
      major_tick_line_width = 2.0
      minor_tick_line_width = 1.0
      
   class Legend(I):
      background_fill_alpha = 0.75
      border_line_width     = 0
      border_line_alpha     = 1.0
      label_text_font_size  = "14pt"
      click_policy          = "hide"
      location              = "bottom_right"

   class Line(I):
      line_width = 2.5
      line_cap   = 'square' 

   class Circle(I):
      fill_alpha = 0.3
      line_width = 1.0
      size       = 5.0

   class Quad(I):
      fill_alpha = 0.5

   class VArea(I):
      fill_alpha = 0.3

   class Patch(I):
      fill_alpha = 0.3
   

class Web(Theme):
   def __init__(self, config):
      self.index  = config.PATH_THEME_WEB
      self.colors = Neon.color12()
   
   class Figure(I):
      background_fill_color = "#000000"
      border_fill_color     = "#000e0e"
      outline_line_color    = "#005050"

   class Grid(I):
      grid_line_color = "#005050"
      grid_line_alpha = 0.0

   class Title(I):
      text_color     = "#00bbbb"

   class Axis(I):
      axis_line_color            = "#005050"
      axis_label_text_color      = "#00bbbb"
      major_label_text_color     = "#005050"
      major_tick_line_color      = "#005050"
      minor_tick_line_color      = "#005050"

   class Legend(I):
      background_fill_color = "#000e0e"
      inactive_fill_color   = "#000e0e"
      border_line_color     = "#005050"
      label_text_color      = "#00bbbb"


class Publication(Theme):
   def __init__(self, config):
      self.index  = config.PATH_THEME_PUB
      self.colors = Solid.color10()
 
   class Figure(I):
      background_fill_color = "#ffffff"
      border_fill_color     = "#ffffff"
      outline_line_color    = "#000000"

   class Grid(I):
      grid_line_color = "#505050"
      grid_line_alpha = 0.0

   class Title(I):
      text_color = "#000000"

   class Axis(I):
      axis_line_color        = "#000000"
      axis_label_text_color  = "#000000"
      major_label_text_color = "#000000"
      major_tick_line_color  = "#000000"
      minor_tick_line_color  = "#000000"

   class Legend(I):
      background_fill_color = "#bbbbbb"
      inactive_fill_color   = "#bbbbbb"
      border_line_color     = "#000000"
      label_text_color      = "#000000"
