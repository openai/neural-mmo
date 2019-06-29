export {Values};

//var worldWidth = 256, worldDepth = 256,
//worldHalfWidth = worldWidth / 2, worldHalfDepth = worldDepth / 2;
var width  = 80;
var height = 80;
var resolution = 3;

class Values {
   constructor(material) {
       /*
        * Adds terrain which shades water, grass, dirt, and mountains
        * based on a heightmap given by the server.
        */
   
      this.material = material
      this.vertShader = document.getElementById(
            'phongValueVertexShader').textContent;
      this.fragShader = document.getElementById(
            'phongValueFragmentShader').textContent;
   }

   reset() {
      this.material.vertexShader = this.vertShader
      this.material.fragmentShader = this.fragShader
      this.material.needsUpdate = true
   }

   valueTex(map){
      var tileMap = this.generateVals(map);
      var tileTexture = this.dataTexture(tileMap, width, height);
      return tileTexture
   }

   update(map, values) {
      var valueTex    = this.valueTex(values);

      // use "this." to create global object
      var custUniforms = {
         valueTex:      { type: "t", value: valueTex},
      };
      var customUniforms = Object.assign( 
            custUniforms, this.material.uniforms);
      this.material.uniforms = customUniforms
      this.material.needsUpdate = true;
   }

   generateVals(map) {
      var mapSz = map.length;
      var data = new Uint8Array( 3*mapSz*mapSz );
      var k = 0;
      for ( var r = 0; r < mapSz; r ++ ) {
         for ( var c = 0; c < mapSz; c ++ ) {
            data[k] = map[r][c][0];
            data[k+1] = map[r][c][1];
            data[k+2] = map[r][c][2];
            k += 3;
         }
      }
      return data;
   }

   texture(fname) {
      var texture = this.loader.load(fname);
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      return texture
   }

   dataTexture(map, width, height) {
      var texture = new THREE.DataTexture(
               map, width, height, THREE.RGBFormat);
      texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.needsUpdate = true;
      return texture
   }


}
