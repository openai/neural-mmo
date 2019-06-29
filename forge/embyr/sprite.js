import * as textsprite from "./textsprite.js";
export {Overhead}

class Overhead extends THREE.Object3D {
   constructor(params, engine) {
      super()
      this.initName(params);
      this.initSpriteStats(params, engine);
   }

   update(params) {
      this.stats.update(params);
   }

   updateFast() {
      this.stats.updateFast();
   }

   initSpriteStats(params, engine) {
      this.stats = new SpriteStats(params, engine);
      this.add(this.stats);
   }

   initName(params) {
      var sprite = textsprite.makeTextSprite(params['name'], "16", params['color']);
      //sprite.scale.set( 64, 30, 1 );
      sprite.position.y = 32;
      this.add(sprite);
   }
}

class SpriteStats extends THREE.Object3D {
   constructor(params, engine) {
      super();
      this.barHeight = 8;
      this.engine = engine

      this.health = this.initBar(0x00ff00, params['health']['max'], 0)
      this.water  = this.initBar(0x0000ff, params['water']['max'], 8)
      this.food   = this.initBar(0xd4af37, params['food']['max'], 16)
   }

   update(params) {
      this.health.update(params['health']['val']);
      this.water.update(params['water']['val']);
      this.food.update(params['food']['val']);
   }

   updateFast() {
      this.quaternion.copy(this.engine.camera.quaternion);
   }

   initBar(color, maxVal, heightOffset) {
      var bar = new StatBar(color, maxVal, this.engine);
      bar.position.y = heightOffset
      this.add(bar)
      return bar
   }
}

class StatBar extends THREE.Object3D {
   constructor(color, maxVal, engine) {
      super();
      this.height = 8;
      this.width = 64;
      this.maxVal = maxVal;

      this.engine = engine
      this.bar = this.initSprite(color);
      this.bar.center = new THREE.Vector2(0, 0);
      this.update(maxVal);
   }

   initSprite(hexColor) {
      //var material = new THREE.SpriteMaterial({color: hexColor});
      var clr = new THREE.Color(hexColor);
      //this.valBar.scale.set(this.width, this.height, 1);
      //var color = new THREE.Vector3(clr[0], clr[1], clr[2]);
      //color:   { value: color},
      var customUniforms = {
         color:   { value: clr},
         width:   { type: "f", value: this.width},
         val:     { type: "f", value: this.maxVal},
      };

      var material = new THREE.ShaderMaterial(
      {
         uniforms: customUniforms,
         vertexShader:   document.getElementById('statVertexShader').textContent,
         fragmentShader: document.getElementById('statFragmentShader').textContent,
      });
      this.material = material;
      var sprite = new THREE.Sprite(material)
      this.sprite = sprite
      this.add(sprite)
      return sprite
   }

   update(val) {
      this.bar.scale.set(this.width, this.height, 1);
      this.material.uniforms.val.value = val;
      this.material.needsUpdate = true;
   }

}

