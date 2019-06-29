export {Counts};

var width  = 80;
var height = 80;
var resolution = 3;

class Counts {
   constructor(material) {
       /*
        * Adds terrain which shades water, grass, dirt, and mountains
        * based on a heightmap given by the server.
        */

      this.material = material
      this.vertShader = document.getElementById(
            'phongCountsVertexShader').textContent;
      this.fragShader = document.getElementById(
'phongCountsFragmentShader').textContent;
   }

   reset() {
      this.material.vertexShader   = this.vertShader
      this.material.fragmentShader = this.fragShader
      this.material.needsUpdate = true
   }

   generateCounts(map, idx) {
      var mapSz = map.length;
      var data = new Uint8Array( 3*mapSz*mapSz );
      var k = 0;
      for ( var r = 0; r < mapSz; r ++ ) {
         for ( var c = 0; c < mapSz; c ++ ) {
            data[k] = map[r][c][idx];
            data[k+1] = map[r][c][idx];
            data[k+2] = map[r][c][idx];
            k += 3;
         }
      }
      return data;
   }

   countTex(map, idx){
      var tileMap = this.generateCounts(map, idx);
      var tileTexture = this.dataTexture(tileMap, width, height);
      return tileTexture
   }

   update(counts) {
      var countZeroTex  = this.countTex(counts, 0);
      var countOneTex   = this.countTex(counts, 1);
      var countTwoTex   = this.countTex(counts, 2);
      var countThreeTex = this.countTex(counts, 3);
      var countFourTex  = this.countTex(counts, 4);
      var countFiveTex  = this.countTex(counts, 5);
      var countSixTex   = this.countTex(counts, 6);
      var countSevenTex = this.countTex(counts, 7);

      // use "this." to create global object
      var custUniforms = {
         countZeroTex:  { type: "t", value: countZeroTex },
         countOneTex:   { type: "t", value: countOneTex },
         countTwoTex:   { type: "t", value: countTwoTex },
         countThreeTex: { type: "t", value: countThreeTex },
         countFourTex:  { type: "t", value: countFourTex },
         countFiveTex:  { type: "t", value: countFiveTex },
         countSixTex:   { type: "t", value: countSixTex },
         countSevenTex: { type: "t", value: countSevenTex },
      };
      var customUniforms = Object.assign( 
            custUniforms, this.material.uniforms)
      this.material.uniforms = customUniforms

      countZeroTex.needsUpdate = true
      countOneTex.needsUpdate = true
      countTwoTex.needsUpdate = true
      countThreeTex.needsUpdate = true
      countFourTex.needsUpdate = true
      countFiveTex.needsUpdate = true
      countSixTex.needsUpdate = true
      countSevenTex.needsUpdate = true

      this.material.needsUpdate = true;
      this.material.uniforms.needsUpdate = true;
      this.material.uniforms.countZeroTex.value = countZeroTex;
      this.material.uniforms.countOneTex.value = countOneTex;
      this.material.uniforms.countTwoTex.value = countTwoTex;
      this.material.uniforms.countThreeTex.value = countThreeTex;
      this.material.uniforms.countFourTex.value = countFourTex;
      this.material.uniforms.countFiveTex.value = countFiveTex;
      this.material.uniforms.countSixTex.value = countSixTex;
      this.material.uniforms.countSevenTex.value = countSevenTex;
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
