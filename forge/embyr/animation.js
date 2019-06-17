import * as textsprite from "./textsprite.js";

export {Move, Damage, Melee, Range, Mage}

//We dont exactly have animation tracks for this project
class ProceduralAnimation {

   constructor() {
      this.clock = new THREE.Clock()
      this.elapsedTime = 0.0;
      this.delta = 0.0;
      this.setup()
      this.scheduler = setTimeout(this.update.bind(this), 1000*tick/nAnim);
   }

   update() {
      this.delta = this.clock.getDelta();
      var time = this.elapsedTime + this.delta;
      this.elapsedTime = Math.min(time, tick);
      this.step(this.delta, this.elapsedTime);
      if (this.elapsedTime < tick) {
         this.scheduler = setTimeout(this.update.bind(this), 1000*tick/nAnim);
      }
      else {
         this.finish();
      }
   }

   //Abstract
   step(delta, elapsedTime) {
      throw new Error(
              'Must override abstract step method of ProceduralAnimation');
   }

   //Optional call before animation
   setup() {
   }

   //Optional call upon animation termination
   finish() {
   }

   cancel() {
      this.finish();
      clearTimeout(this.scheduler);
   }
}

class Move extends ProceduralAnimation {
   constructor(ent, targ) {
      super();
      this.pos  = ent.obj.position.clone();
      this.targ = ent.coords(targ);
      this.isTarget = false;
      this.ent = ent;
   }

   step(delta, elapsedTime) {
      var moveFrac = elapsedTime / tick;
      var x = this.pos.x + moveFrac * (this.targ.x - this.pos.x);
      var y = this.pos.y + moveFrac * (this.targ.y - this.pos.y);
      var z = this.pos.z + moveFrac * (this.targ.z - this.pos.z);
      var pos = new THREE.Vector3(x, y, z);
      this.ent.obj.position.copy(pos);

      if (this.isTarget) {
         engine.camera.position.copy(pos);
         engine.controls.target.copy(pos);//this.ent.obj.position);
         engine.controls.update();
      }
   }
}

class Damage extends ProceduralAnimation {
   constructor(ent, damage) {
      super();
      this.dmg = textsprite.makeTextSprite(damage, "16", '#ff0000');
      this.dmg.scale.set(0.007, 0.017, 1);
      this.height = 128
      this.dmg.position.y = this.height
      this.ent = ent;
      ent.obj.add(this.dmg)
   }

   step(delta, elapsedTime) {
      var moveFrac = elapsedTime / tick;
      this.dmg.position.y = this.height+32*moveFrac;
   }

   finish() {
      this.ent.obj.remove(this.dmg);
   }
}

class Attack extends ProceduralAnimation {
   constructor(scene, orig, targ) {
      super();
      this.orig = orig.obj;
      this.targ = targ.obj;
      this.scene = scene;

      var attkMatl = new THREE.MeshBasicMaterial( {
           color: this.color} );
      var attkMesh = new THREE.Mesh(this.attkGeom, attkMatl);
      this.attk = attkMesh
      this.attk.position.x = this.orig.x;
      this.attk.position.y = 128;
      this.attk.position.z = this.orig.z;
      scene.add(this.attk);
   }

   step(delta, elapsedTime) {
      var moveFrac = elapsedTime / tick;
      var x = this.orig.position.x + moveFrac * (this.targ.position.x - this.orig.position.x) + 16;
      var y = 96;
      var z = this.orig.position.z + moveFrac * (this.targ.position.z - this.orig.position.z) + 16;
      var pos = new THREE.Vector3(x, y, z)
      this.attk.position.copy(pos);
   }

   finish() {
      this.scene.remove(this.attk);
   }
}

class Melee extends Attack{
   setup() {
      this.attkGeom = new THREE.BoxGeometry(10, 10, 10);
      this.color = '#ff0000';
   }
}

class Range extends Attack{
   setup() {
      this.attkGeom = new THREE.OctahedronGeometry(10);
      this.color = '#00ff00';
   }
}

class Mage extends Attack{
   setup() {
      this.attkGeom = new THREE.IcosahedronGeometry(10);
      this.color = '#0000ff';
   }
}

class StatBar extends THREE.Object3D {
   constructor(color, width, height) {
      super();
      this.valBar = this.initSprite(color);
      this.valBar.center = new THREE.Vector2(1, 0);

      this.redBar = this.initSprite(0xff0000);
      this.redBar.center = new THREE.Vector2(0, 0);

      this.offset = 64;
      this.height = height;
      this.width = width;
      this.update(width);
   }

   initSprite(hexColor) {
      var material = new THREE.SpriteMaterial({
         color: hexColor, sizeAttenuation: false});
      var sprite = new THREE.Sprite(material)
      this.add(sprite)
      return sprite
   }

   update(val) {
      this.valBar.scale.set(val, this.height, 1);
      this.redBar.scale.set(this.width - val, this.height, 1);
   }
}

