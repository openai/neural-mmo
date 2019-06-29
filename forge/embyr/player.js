import * as textsprite from "./textsprite.js";
import * as OBJ from "./obj.js";
import * as Animation from "./animation.js";
import * as Sprite from "./sprite.js";

export {Player, PlayerHandler};


class PlayerHandler {
   /*
    * The PlayerHandler receives packets from the server containing player
    * information (other players' movements, interactions with our player)
    * and disperses the signals appropriately.
    */

   constructor(engine) {
      this.players = {};
      this.engine = engine;
      this.load();
      this.finishedLoading = false;
   }

   load() {
      this.nnObjs = {}
      var promises = [];
      var scope = this;

      for (var name in Neon) {
         var color = Neon[name];
         var loadedPromise = OBJ.loadNN(color);

         loadedPromise.then( function (result) {
            scope.nnObjs[result.myColor] = result.obj;
         });

         promises.push(loadedPromise);
      }

      Promise.all(promises).then(function () {
         scope.finishedLoading = true;
         console.log("PlayerHandler: Finished loading all meshes.");
      });
   }

   addPlayer(id, params) {
      var player = new Player(this, params)
      this.players[id] = player;
      player.castShadow = true;
      this.engine.scene.add(player);
   }

   removePlayer( playerIndex ) {
      this.engine.scene.remove(this.players[playerIndex])
      delete this.players[playerIndex];
   }

   updateFast() {
      for (var id in this.players) {
         this.players[id].updateFast();
      }
   }

   updateData(ents) {
      if (!this.finishedLoading) {
         return;
      }

      for (var id in this.players) {
         if (!(id in ents)) {
            this.removePlayer(id);
         }
      }

      for (var id in ents) {
         if (!(id in this.players)) {
            this.addPlayer(id, ents[id])
         }
         this.players[id].updateData(this.engine, ents[id], this.players);
      }
   }
}

class Player extends THREE.Object3D {
   constructor(handler, params)  {
      super();
      this.translateState = false;
      this.translateDir = new THREE.Vector3(0.0, 0.0, 0.0);
      this.moveTarg = [0, 0];
      this.height = sz;    // above grass, below mountains
      this.entID = params['entID'];
      this.engine = handler.engine;
      this.color = null;
      this.attackMap = null;

      this.initObj(params, handler);
      this.initOverhead(params);
   }

   initObj(params, handler) {
      var pos = params['pos'];
      //this.obj = OBJ.loadNN(params['color']);
      this.color = params['color']
      this.obj = handler.nnObjs[this.color].clone();
      this.obj.position.y = this.height;
      this.obj.position.copy(this.coords(pos));
      this.target = this.obj.position.clone();
      this.add(this.obj)
      this.anims = [];
   }

   initOverhead(params) {
      this.overhead = new Sprite.Overhead(params, this.engine);
      this.obj.add(this.overhead)
      this.overhead.position.y = sz;
   }

   //Format: pos = (r, c)
   coords(pos) {
      return new THREE.Vector3(pos[1]*sz+sz+sz/2, this.height, pos[0]*sz+sz+sz/2);
   }

   cancelAnims() {
      for (var anim in this.anims) {
         this.anims[anim].cancel()
      }
   }

   updateData(engine, packet, players) {
      this.cancelAnims();

      var move = packet['pos'];
      //console.log("Move: ", move)
      this.anims.push(new Animation.Move(this, move));

      var damage = packet['damage'];
      if (damage != null) {
         this.anims.push(new Animation.Damage(this, packet['damage']));
      }

      this.overhead.update(packet)

      if (packet['attack'] != null) {
         var targ = packet['attack']['target'];
         var targID = parseInt(targ, 10);
         if (this.entID != targID && targID in players) {
            var attk;
            switch (packet['attack']['style']) {
               case 'Melee':
                  attk = new Animation.Melee(engine.scene, this, players[targID]);
                  break;
               case 'Range':
                  attk = new Animation.Range(engine.scene, this, players[targID]);
                  break;
               case 'Mage':
                  attk = new Animation.Mage(engine.scene, this, players[targID]);
                  break;
            }
            this.anims.push(attk);
         }
      }
      this.attackMap = packet['attackMap'];
   }

   updateFast() {
      this.overhead.updateFast()
   }

   sendMove() {
      var packet = JSON.stringify({
         "pos" : this.moveTarg
      });
      ws.send(packet);
   }
}

