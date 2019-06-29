import { Vector2 } from "three";

export const textAlign = {
  center: new Vector2(0, 0),
  left: new Vector2(1, 0),
  topLeft: new Vector2(1, -1),
  topRight: new Vector2(-1, -1),
  right: new Vector2(-1, 0),
  bottomLeft: new Vector2(1, 1),
  bottomRight: new Vector2(-1, 1),
}


var fontHeightCache: { [id: string]: number; } = {};

export function getFontHeight (fontStyle: string) {
  var result = fontHeightCache[fontStyle];

  if (!result)
  {
    var body = document.getElementsByTagName('body')[0];
    var dummy = document.createElement('div');

    var dummyText = document.createTextNode('MÉq');
    dummy.appendChild(dummyText);
    dummy.setAttribute('style', `font:${ fontStyle };position:absolute;top:0;left:0`);
    body.appendChild(dummy);
    result = dummy.offsetHeight;

    fontHeightCache[fontStyle] = result;
    body.removeChild(dummy);
  }

  return result;
}
