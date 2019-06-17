export {loadObj, loadNN};

function loadObj(objf, mtlf) {
    var contain = new THREE.Object3D();
    var obj;

    function onMTLLoad( materials ) {
        materials.preload();

        var objLoader = new THREE.OBJLoader();
        objLoader.setMaterials( materials );
        //objLoader.setPath( path );

        function onOBJLoad(object) {
           obj = object;
           obj.scale.x = 50;
           obj.scale.y = 50;
           obj.scale.z = 50;
           contain.add(obj)
        }
        objLoader.load( objf, onOBJLoad);
    }

    var mtlLoader = new THREE.MTLLoader();
    //mtlLoader.setPath( path );
    mtlLoader.load( mtlf, onMTLLoad);
    return contain
}

function promisifyLoader ( loader ) {

  function promiseLoader ( url ) {
    return new Promise( ( resolve, reject ) => {
      loader.load( url, resolve, Function.prototype, reject );
    } );
  }

  return {
    originalLoader: loader,
    load: promiseLoader,
  };

}

function loadNN(color) {
    var objf = 'resources/nn.obj';
    var mtlf = 'resources/nn.mtl';
    var contain = new THREE.Object3D();
    var obj;

	function onOBJLoad( object ) {
	   obj = object;
	   obj.scale.x = 50;
	   obj.scale.y = 50;
	   obj.scale.z = 50;
	   obj.children[0].material.color.setHex(parseInt(color.substr(1), 16));
      obj.children[0].castShadow = true;

      obj.traverse( function ( child ) {

          if ( child instanceof THREE.Mesh ) {

              //child.material.map = texture;
              child.castShadow = true;

          }

      } );
         contain.add(obj);
       return new Promise( ( resolve, reject ) => {
          resolve( { myColor: color, obj: contain} );
       } ); // ugly but gets the job done
	}

	function onMTLLoad( materials ) {
		materials.preload();

		var objLoader = new THREE.OBJLoader();
		objLoader.setMaterials(materials);
		var OBJPromiseLoader = promisifyLoader(objLoader);
		var objPromise = OBJPromiseLoader.load(objf);
		return objPromise;
	}

	var MTLPromiseLoader = promisifyLoader(new THREE.MTLLoader());
	var mtlPromise = MTLPromiseLoader.load(mtlf);
	var finalPromise = mtlPromise.then(materials=>onMTLLoad(materials))
			                     .then(object=>onOBJLoad(object));
	return finalPromise;
}
