export {EntityBox}


class EntityBox {

   constructor() {
      // HTML stuff
      this.container = document.createElement("div");
      this.container.id = "entity_container";
      this.container.style.cssText =  // opacity:0.5. z-index puts this in front
		   'position:fixed;left:0;top:66%;z-index:10000';

      this.panels = [];

		var pp = new PlayerPanel();
		var fp = new FlatPanel("#000000");
		var tp = new TilePanel(0, 0);

		this.addPanel(fp);
		this.addPanel(tp);
		this.addPanel(pp);

      this.dom = this.container;
      this.mode = -1;
		document.body.appendChild(this.dom);

      var scope = this;
      function onMouseDown( event ) {scope.onClick(event);}
      this.container.addEventListener( "click", onMouseDown, false);
   }

   onClick (event) {
      this.container.style.display = "none";
   }

   addPanel (panel) {
      this.container.appendChild(panel.dom);
		this.panels.push(panel);
   }

   showAll() {
		this.container.style.display = 'block';
   }

   showPanel( id ) {
		for ( var i = 0; i < this.container.children.length; i ++ ) {
			this.container.children[i].style.display = i === id ? 'block' : 'none';
		}
		this.mode = id;
	}

   setPlayer( player ) {
      for (var i = 0; i < this.panels.length; i++) {
         this.panels[i].setPlayer(player);
      }
   }

   setColor(color) {
      for (var i = 0; i < this.panels.length; i++) {
         this.panels[i].setColor(color);
      }
   }

   setText(text) {
      for (var i = 0; i < this.panels.length; i++) {
         this.panels[i].setText(text);
      }
   }

   update(delta) {
      for (var i = 0; i < this.panels.length; i++) {
         this.panels[i].update(delta);
      }
   }
}


class Panel {

   constructor() {
		this.PR = Math.round( window.devicePixelRatio || 1 );
   }

   setColor(color) {
   }

   setText(text) {
   }

   setPlayer(player) {
      this.player = player;
      this.setColor(player.color);
      this.setText("Player #" + player.entID);
   }

   update(delta) {
      // Do nothing
   }
}

class TilePanel extends Panel {

   constructor(i, j) {
      super();
		this.TILESZ = 20;
		var canvas = document.createElement( 'canvas' );
      canvas.id = "tile_panel";
		canvas.width = this.TILESZ * 7;
		canvas.height = this.TILESZ * 7;
		canvas.style.cssText = 'position:fixed;left:5%;top:80%;width:'
         + canvas.width + 'px;height:' + canvas.height + 'px';
		this.canvas = canvas;
	   this.context = this.canvas.getContext( '2d' );
		this.dom = canvas;
      this.tileMap = new TileMap();
	}

	drawTiles() {
		for (var c = 0; c < this.tileMap.cols; c++) {
			for (var r = 0; r < this.tileMap.rows; r++) {
				var tile = this.tileMap.getTile(r, c);
				if (tile.getHex() !== 0) { // 0 => empty tile
					this.context.fillStyle = "#" + tile.getHexString();
					this.context.fillRect(
						  c * this.TILESZ, // target x
						  r * this.TILESZ, // target y
						  this.TILESZ, // target width
						  this.TILESZ // target height
					);
				}
			}
		}
   }

   update( delta ) {
      super.update(delta);
      this.context.fillStyle = "#000000";
		this.context.fillRect( 0, 0, this.canvas.width, this.canvas.height );

      this.tileMap.setTiles(this.player.attackMap);
      this.drawTiles();
   }
}


class TileMap {
	constructor() {
      /* Implements row-major tile map.
       */
      this.array = null; // array is N rows by 3xN cols
	   this.rows = 0;
      this.cols = 0;
      this.maxCount = 0;
   }

   setTiles(array) {
      this.array = array;
      var center = array[3][3];
      array[3][3] = [0,0,0];
      function getMax( a ) {
         return Math.max(...a.map(e => Array.isArray(e) ? getMax(e) : e));
      }
      this.maxCount = getMax(array);
      array[3][3] = center;
      this.rows = array.length;
      this.cols = array[0].length;
      console.log(this.array);
   }

   getTile(i, j) {
      var rgb = this.array[i][j];
      var r = rgb[0]; var g = rgb[1]; var b = rgb[2];
      return new THREE.Color(
            Math.min(1.0, r/this.maxCount),
            Math.min(1.0, g/this.maxCount),
            Math.min(1.0, b/this.maxCount));
   }
}


class FlatPanel extends Panel {

   constructor(fgColor) {
      /* fgColor will be the color of the text.
       * bgColor will be the color of the background, and possibly the color
       * of the graph.
       */
      super();

      this.fgColor = fgColor;  // this shouldn't change

		this.WIDTH = window.innerWidth / 5 * this.PR;
      this.HEIGHT = window.innerHeight / 3 * this.PR;
		this.TEXT_X = 20 * this.PR;
      this.TEXT_Y = 50 * this.PR;

		var canvas = document.createElement( 'canvas' );
      canvas.id = "flat_panel";
		canvas.width = this.WIDTH;
		canvas.height = this.HEIGHT;
		canvas.style.cssText = 'position:fixed;left:0;top:67%;width:'
         + this.WIDTH/this.PR + 'px;height:' + this.HEIGHT/this.PR + 'px';
		this.canvas = canvas;

		var context = this.canvas.getContext( '2d' );
		context.font = ( 40 * this.PR ) + 'px DragonSlapper';
		context.textBaseline = 'top';
      this.context = context;

		this.dom = canvas;
	}

   setText( text ) {
      this.text = text;
      this.repaint();
   }

	setColor( color ) {
      this.bgColor = color;
      this.repaint();
	}

   repaint() {
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);

      var grd = this.context.createLinearGradient(0, 0, this.canvas.width, 0);
      grd.addColorStop(0, this.bgColor);
      var alphaColor = this.bgColor + "20";
      grd.addColorStop(0.8, alphaColor);
      grd.addColorStop(1, "transparent");
      this.context.fillStyle = grd;
		this.context.fillRect( 0, 0, this.WIDTH, this.HEIGHT );
		this.context.fillStyle = this.fgColor;
		this.context.fillText( this.text, this.TEXT_X, this.TEXT_Y );
   }

   update ( delta ) {
		// currently we do nothing
		// this.context.drawImage(
		//		canvas, GRAPH_X + PR, GRAPH_Y, GRAPH_WIDTH - PR, GRAPH_HEIGHT,
		//		GRAPH_X, GRAPH_Y, GRAPH_WIDTH - PR, GRAPH_HEIGHT );
   }
}

class RenderedPanel extends Panel {

   constructor() {
      super();
		var canvas = document.createElement( 'canvas' );
      canvas.id = "rendered_panel";
		canvas.width = window.innerWidth/6 * this.PR;
		canvas.height = window.innerHeight/6 * this.PR;
		canvas.style.cssText = 'position:fixed;left:10%;top:80%;width:'
         + canvas.width/this.PR + 'px;height:' + canvas.height/this.PR + 'px';
		this.canvas = canvas;

      // engine stuff in a panel
      this.scene = new THREE.Scene();

      this.camera = new THREE.PerspectiveCamera(
              60, window.innerWidth / window.innerHeight, 1, 20000);

      this.renderer = new THREE.WebGLRenderer( {
         antialias: true, canvas: this.canvas, alpha: true} );
      this.renderer.setClearColor( 0x000000, 0);
      this.renderer.setPixelRatio( window.devicePixelRatio );
      this.renderer.setSize( window.innerWidth/6, window.innerHeight/6 );
      this.renderer.shadowMap.enabled = true;

      var pointLight = new THREE.PointLight( 0xffffff, 1.5, 0, 2);
      pointLight.position.set(64*40, 1500, 64*40);
      pointLight.caseShadow = true;
      pointLight.shadow.camera.far = 0;
      this.scene.add(pointLight);

      this.controls = new THREE.OrbitControls(this.camera, this.canvas);
      //this.controls.lookVertical = true;
      this.controls.enableZoom = true;
      //this.controls.autoRotate = true;
      this.controls.update();
		this.dom = this.canvas;
   }

   setColor(color) {
      //this.scene.background = new THREE.Color( color );
   }

   update(delta) {
      this.renderer.render( this.scene, this.camera );
   }
}

class AttackPanel extends TilePanel {}

class StatsPanel extends TilePanel {}

class PlayerPanel extends RenderedPanel {

   setPlayer( player ) {
      super.setPlayer(player);
      this.player = player.obj.clone({recursive: false});
      this.scene.add(this.player);
      this.player.position.set(0, 0, 0);
      this.controls.target.copy(this.player.position);
      this.camera.position.z -= 5;
      this.camera.lookAt(this.player.position);
   }

   update(delta) {
      if (this.player) {
         this.controls.update();
      }
      super.update(delta);
   }
}


