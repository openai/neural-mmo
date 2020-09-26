from pdb import set_trace as T

from collections import defaultdict

from forge.blade.io.comparable import IterableTypeCompare, Iterable

class I(metaclass=Iterable):
   pass

class Theme(metaclass=IterableTypeCompare):
   @classmethod
   def dict(cls):
     data = defaultdict(dict)

     for (group,), c in Theme:
        for attr, val in c:
           data[group][attr] = val

     for (group,), c in cls:
        for attr, val in c:
           data[group][attr] = val

     return dict(data)

   class Figure(I):
      outline_line_cap = "square"
 
   class Grid(I):
      grid_line_dash  = [6, 4]
      grid_line_width = 1.1

   class Title(I):
      text_font_size = "20pt"
      align          = "center"

   class Axis(I):
      axis_label_text_font_size  = "20pt"
      major_label_text_font_size = "16pt"

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

   class Quad(I):
      fill_alpha = 0.5

   class VArea(I):
      fill_alpha = 0.3

   class Patch(I):
      fill_alpha = 0.3
   

class Web(Theme):
   class Figure(I):
      background_fill_color = "#000000"
      border_fill_color     = "#000e0e"
      outline_line_color    = "#000e0e"

   class Grid(I):
      grid_line_color = "#005050"
      grid_line_alpha = 1.0

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
   class Figure(I):
      background_fill_color = "#ffffff"
      border_fill_color     = "#ffffff"
      outline_line_color    = "#000000"
      outline_line_width    = 2.0

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
