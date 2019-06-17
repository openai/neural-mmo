import * as engineM from './engine.js';
import * as playerM from './player.js';
import * as terrainM from './terrain.js';
import * as countsM from './counts.js';
import * as valuesM from './values.js';
import * as entityM from './entitybox.js';
import * as textsprite from './textsprite.js';

var client, counts, values, stats, box;
var CURRENT_VIEW = views.CLIENT;


THREE.Object3D.prototype.rotateAroundWorldAxis = function() {

    // rotate object around axis in world space (the axis passes through point)
    // axis is assumed to be normalized
    // assumes object does not have a rotated parent

    var q = new THREE.Quaternion();

    return function rotateAroundWorldAxis( point, axis, angle ) {

        q.setFromAxisAngle( axis, angle );

        this.applyQuaternion( q );

        this.position.sub( point );
        this.position.applyQuaternion( q );
        this.position.add( point );

        return this;

    }

}();

class Client{
   // interface for client, viewer, and counts
   constructor (my_container) {
      this.engine = new engineM.Engine(modes.ADMIN, client_container);
      my_container.innerHTML = ""; // get rid of the text after loading
      my_container.appendChild( this.engine.renderer.domElement );

      this.handler = new playerM.PlayerHandler(this.engine);
      this.init = true;

      var scope = this; // javascript quirk... don't touch this
      function onMouseDown( event ) { scope.onMouseDown( event ); }
      my_container.addEventListener( 'click', onMouseDown, false );

      this.axPos = tileSz*nTiles/2
      this.p = new THREE.Vector3(this.axPos, 0, this.axPos);
      this.ax = new THREE.Vector3(0, 1, 0);
      this.tick = 0
      this.rotateTime = 0
      this.r = 2750
      
      this.controls = this.engine.controls
      this.camera   = this.engine.camera
      this.camera.position.set(this.axPos, 4000, this.axPos)
      this.controls.update()
   }

   update() {
      var delta = this.engine.clock.getDelta();
      this.engine.update(delta);
      this.handler.updateFast();
      this.updatePacket();

      //this.engine.scene.rotateAroundWorldAxis(this.p, this.ax, 0.0025)
      /*
      if (this.tick > 200) {
         var x = this.axPos + this.r * Math.cos(this.rotateTime)
         var z = this.axPos + this.r * Math.sin(this.rotateTime)
         this.camera.position.set(x, 500, z)
         this.rotateTime += 0.0025
      }
      */

      var packet = this.packet;
      if (packet) {
         this.tick += 1
         // Receive packet, begin translating based on the received position
         packet = JSON.parse(packet);

         var map = packet['map']
         if (this.init) { 
            this.terrain = new terrainM.Terrain(map, this.engine);
            this.values = new valuesM.Values(this.terrain.material);
            this.counts = new countsM.Counts(this.terrain.material)
            displayOnlyCurrent(client.engine);
            this.values.update(map, packet['values']);
            this.counts.update(packet['counts']);
            this.init = false;
         }
         this.handler.updateData(packet['ent']);
         this.terrain.update(map)
         if (CURRENT_VIEW == views.VALUES) {
            this.values.update(map, packet['values']);
         }
         if (CURRENT_VIEW == views.COUNTS) {
            this.counts.update(packet['counts']);
         }
      }

      if (this.terrain) {
         this.terrain.updateFast();
      }
   }

   onWindowResize () {
      this.engine.onWindowResize();
   }

   updatePacket() {
      if (inbox.length > 0) {
         this.packet = inbox.pop();
      } else {
         this.packet = null;
      }
   }

   onMouseDown(event) {
      // handle player event first
      var minDistance = 1000000; // large number
      var minPlayer = null;

      for (var id in this.handler.players) {
         // Player subclasses Object3D
         var player = this.handler.players[id];
         var coords = this.engine.raycast(event.clientX, event.clientY,
                 player);
         if (coords) {
            var distance = this.engine.camera.position.distanceTo(coords);
            if (distance < minDistance) {
               minDistance = distance;
               minPlayer = player;
            }
         }
      }

      if (minPlayer) {
         // now we've identified the closest player
         console.log("Clicked player", minPlayer.entID);
         if (!box) {
            box = new entityM.EntityBox();
         }
         box.setPlayer(minPlayer);
         box.showAll();

         if (this.engine.mode == modes.SPECTATOR) {
            // follow this player
         }
      }

      // then handle translate event (if self is player)
      if (this.engine.mode == modes.PLAYER) {
         //var pos = this.engine.raycast(event.clientX, event.clientY);
         //this.engine.controls.target.set(pos);
      }
   }
}

function webglError() {
   if ( WEBGL.isWebGLAvailable() === false ) {
      document.body.appendChild( WEBGL.getWebGLErrorMessage() );
   }
}

function toggleVisualizers() {
   CURRENT_VIEW = (CURRENT_VIEW + 1) % 3;
   displayOnlyCurrent();
}

function displayOnlyCurrent() {
   client_container.style.display = "block";

   // now update the current view
   switch ( CURRENT_VIEW ) {
      case views.CLIENT:
         client.terrain.reset()
         break;
      case views.COUNTS:
         client.counts.reset()
         break;
      case views.VALUES:
         client.values.reset()
         break;
   }
}

function onKeyDown(event) {
   switch ( event.keyCode ) {
      case 84: // T
         toggleVisualizers();
         break;
   }
}

function onWindowResize() {
   client.onWindowResize();
   if (counts) {counts.onWindowResize();}
   if (values) {values.onWindowResize();}
}

function init() {
   webglError();
   var client_container = document.getElementById("client_container");
   var values_container = document.getElementById("values_container");

   client = new Client(client_container);

   stats  = new Stats();
   client_container.appendChild(stats.dom);

   // Start by setting these to none

   var blocker = document.getElementById("blocker");
   var instructions = document.getElementById("instructions");

   instructions.addEventListener("click", function() {
	   client.engine.controls.enabled = true;
	   client.engine.controls.update();

	   instructions.style.display = "none";
      blocker.style.display = "none";
   }, false);

   window.addEventListener( 'resize', onWindowResize, false );
   window.addEventListener( 'keydown', onKeyDown, false );

   animate();
}

function animate() {
   requestAnimationFrame( animate );
   client.update();
   stats.update();
   if (stats) { stats.update();}
   if (box) { box.update();}
}

// Main
init();



            /*
            var pkt1 = {
               'pos': [18, 11],
               'entID': 394,
               'color':'#ff8000',
               'name': 'Neural_394',
               'food':25, 
               'water':12,
               'health':16,
               'maxFood':32,
               'maxWater':32,
               'maxHealth':32,
               'damage':2,
            }

            var pkt2 = {
               'pos': [19, 13],
               'entID': 383,
               'color':'#8000ff',
               'name': 'Neural_383',
               'food':8, 
               'water':28,
               'health':23,
               'maxFood':32,
               'maxWater':32,
               'maxHealth':32,
               'damage':0,
               'attack':'Range',
               'target':'394',
            }
        
            var player1 = new playerM.Player(this.handler, 0, pkt1)
            this.engine.scene.add(player1);
            player1.updateData(this.engine, pkt1, {})
            this.player1 = player1

            var player2 = new playerM.Player(this.handler, 0, pkt2)
            this.engine.scene.add(player2);
            player2.updateData(this.engine, pkt2, {394: player1})
            this.player2 = player2

            var attkGeom = new THREE.IcosahedronGeometry(10);
            var attkMatl = new THREE.MeshBasicMaterial({color: '#0000ff'});
            var attkMesh = new THREE.Mesh(attkGeom, attkMatl);

            var p2 = player2.obj.position
            var p1 = player1.obj.position
            var moveFrac = 0.75
            var y = 96;

            var x = p1.x + moveFrac * (p2.x - p1.x) + 16;
            var z = p1.z + moveFrac * (p2.z - p1.z) + 16;
            var pos = new THREE.Vector3(x, y, z)
            attkMesh.position.x = x
            attkMesh.position.y = y
            attkMesh.position.z = z
            this.attkMesh = attkMesh
            this.engine.scene.add(attkMesh);

            var dmg = textsprite.makeTextSprite(2, "200", '#ff0000');
            dmg.scale.set( 10, 30, 1 );
            dmg.position.x = p2.x
            dmg.position.y = p2.y + 128
            dmg.position.z = p2.z
            this.dmg = dmg
            this.engine.scene.add(dmg);
            */


