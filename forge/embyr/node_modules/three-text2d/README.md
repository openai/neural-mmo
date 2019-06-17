

three-text2d
===

Render texture from canvas into THREE's Mesh or Sprite.

Usage
---

**Mesh**

```javascript
import { MeshText2D, textAlign } from 'three-text2d'

var text = new MeshText2D("RIGHT", { align: textAlign.right, font: '30px Arial', fillStyle: '#000000', antialias: true })
scene.add(text)
```

**Sprite**

```javascript
import { SpriteText2D, textAlign } from 'three-text2d'

var sprite = new SpriteText2D("SPRITE", { align: textAlign.center,  font: '40px Arial', fillStyle: '#000000' , antialias: false })
scene.add(sprite)
```

License
---

MIT
