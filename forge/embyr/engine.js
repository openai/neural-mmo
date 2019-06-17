export {Engine};

class Engine {

   constructor(mode, aContainer) {
      this.mode = mode;
      this.container = aContainer;
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(seal_sky);

      this.mesh = null; // we'll initialize from the server packet

      this.camera = new THREE.PerspectiveCamera(
              60, window.innerWidth / window.innerHeight, 1, 20000 );
      this.camera.position.y = 2 * sz;
      this.camera.position.z = 10;

      var width  = window.innerWidth; 
      var height = window.innerHeight;
      var aspect = width/height;
      var near = 10;
      var far = 1000;
      var fov = 90;

      this.renderer = new THREE.WebGLRenderer( { antialias: true } );
      //this.renderer.setPixelRatio( window.devicePixelRatio );
      this.renderer.setPixelRatio(2); //Antialias x2
      this.renderer.setSize( window.innerWidth, window.innerHeight );


      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
      //this.renderer.shadowMap.renderSingleSided = false;
      //this.renderer.shadowMap.renderReverseSided = false;

      this.initializeControls();

      this.mouse = new THREE.Vector2();
      this.raycaster = new THREE.Raycaster();
      this.clock = new THREE.Clock();

      //initialize lights
      //var ambientLight = new THREE.AmbientLight( 0xcccccc, 0.25);
      //this.scene.add( ambientLight );

      var pointLight = new THREE.PointLight( 0xffffff, 1, 0, 0.5 );
      pointLight.position.set( 64*40, 500, 64*40 );
      //pointLight.position.set( 0, 1500, 0 );
      pointLight.castShadow = true;
      pointLight.shadow.camera.far = 0;
      this.scene.add(pointLight);

      var clip = 40*64;

      var light = new THREE.DirectionalLight(0xffffff, 1.0);
      light.position.set(clip,300,clip);
      light.target.position.set(clip, 0, clip)
      light.target.updateMatrixWorld()
      light.shadow.camera.near = near;       
      light.shadow.camera.far = far;      
      light.shadow.camera.left = -clip;
      light.shadow.camera.bottom = -clip;
      light.shadow.camera.right = clip;
      light.shadow.camera.top = clip;
      light.castShadow = true;
      light.shadow.mapSize.width = 2048;
      light.shadow.mapSize.height = 2048;
      this.scene.add(light)

      var geometry = new THREE.SphereGeometry(100, 32, 32);
      //var material = new THREE.MeshPhongMaterial({color: 0x0000ff, side: THREE.DoubleSide});

      /*
      var vertShader = document.getElementById(
            'phongVertexShader').textContent;
      var fragShader = document.getElementById(
            'phongFragmentShader').textContent;
      //var fragShader = THREE.ShaderChunk[ 'meshphong_frag' ]


      var defines = {};
      defines[ "USE_MAP" ] = "";

      //defines: defines,
      var material = new THREE.ShaderMaterial(
      {
         color: 0x0000ff,
         uniforms: THREE.ShaderLib.phong.uniforms,
         vertexShader: THREE.ShaderChunk['meshphong_vert'],
         fragmentShader: THREE.ShaderChunk['meshphong_frag'],
         name: 'custom-material',
         lights: true,
      });
      //vertexShader: vertShader,
      //fragmentShader: fragShader,
 

      var crosscap = new THREE.Mesh(geometry, material);
      crosscap.receiveShadow = true;
      crosscap.castShadow = true;
      crosscap.position.y = 400;
      crosscap.position.x = 4380;
      crosscap.position.z = 4380;
      this.scene.add(crosscap);

      var directionalLight = new THREE.DirectionalLight( 0xffffff, 2 );
      directionalLight.position.set( 1, 0.5, 0 ).normalize();
      directionalLight.castShadow = true
      this.scene.add( directionalLight );

      var spotLight = new THREE.SpotLight( 0xffffff, 2 );
      spotLight.castShadow = true
      this.scene.add( spotLight);
      */

      document.body.appendChild( this.renderer.domElement );
   }

   initializeControls() {
      var controls = new THREE.OrbitControls(this.camera, this.container);
      controls.mouseButtons = {
         LEFT: THREE.MOUSE.MIDDLE, // rotate
         // RIGHT: THREE.MOUSE.LEFT // pan
      }
      controls.target.set( 40*sz, 0, 40*sz );
      controls.minPolarAngle = 0.0001;
      controls.maxPolarAngle = Math.PI / 2.0 - 0.1;

      controls.movementSpeed = 1000;
      controls.lookSpeed = 0.125;
      controls.lookVertical = true;

      if ( this.mode == modes.ADMIN ) {
         controls.enableKeys = true;
         controls.enablePan = true;
      }

      controls.enabled = false;
      controls.update();
      this.controls = controls;
   }

   onWindowResize() {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.controls.update();
      this.renderer.setSize( window.innerWidth, window.innerHeight );
   }

   raycast(clientX, clientY, mesh) {
      this.mouse.x = (
            clientX / this.renderer.domElement.clientWidth ) * 2 - 1;
      this.mouse.y = - (
            clientY / this.renderer.domElement.clientHeight ) * 2 + 1;
      this.raycaster.setFromCamera( this.mouse, this.camera );

      // See if the ray from the camera into the world hits one of our meshes
      var intersects = this.raycaster.intersectObject( mesh, true );
      // recursive=true
      // for normal use: this.mesh?

      // Toggle rotation bool for meshes that we clicked
      if ( intersects.length > 0 ) {
         //var x = intersects[ 0 ].point.x;
         //var y = intersects[ 0 ].point.y;
         //var z = intersects[ 0 ].point.z;

         //x = Math.min(Math.max(0, Math.floor(x/sz)), worldWidth);
         // new terrain gen uses +x, -z
         //z = Math.max(Math.min(0, Math.floor(z/sz)), -worldDepth);
         // z = Math.min(Math.max(0, Math.floor(z/sz)), worldDepth);
         return intersects[0].point;
      }
      return false;
   }

   update(delta) {
      this.controls.update( delta );
      this.renderer.render( this.scene, this.camera );
   }

}

