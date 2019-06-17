uniform sampler2D bumpTexture;
uniform float tileScale;

varying float vAmount; varying vec2 vUV;
varying vec3 newPosition;

vec2 tilePos;
float tile;
float border;
float val;
float tol = 0.025;
float mag = 1.0;
float nTiles = 80.00001;
//varying vec3 vViewPosition;
//varying vec3 vNormal;

void main()
{
    //vec3 position = vec3(position.x + 64.0*40.0, position.y+64.0*40.0, position.z);
    vUV = uv;
    tilePos = floor(position.xz / 64.0);
    vAmount = 255.0*texture2D(bumpTexture, tilePos/nTiles).r;

    //Check if we are on a border
    vec2 err = position.xz/64.0 - tilePos;


    if (err.x < tol) {
       val = 255.0*texture2D(bumpTexture, vec2(tilePos.x-1.0, tilePos.y)/nTiles).r;
       if (val == 1.0) { vAmount = mag; }
    }

    if (err.y < tol) {
       val = 255.0*texture2D(bumpTexture, vec2(tilePos.x, tilePos.y-1.0)/nTiles).r;
       if (val == 1.0) { vAmount = mag; }
    }

    if (err.x < tol && err.y < tol) {
       val = 255.0*texture2D(bumpTexture, vec2(tilePos.x-1.0, tilePos.y-1.0)/nTiles).r;
       if (val == 1.0) { vAmount = mag; }
    }


    if (position.x == 0.0 || position.z == 0.0 || 
          position.x / 64.0 == 80.0 || position.z / 64.0 == 80.0) {
       vAmount = 0.0;
    }

    // move the position along the normal
    newPosition = position + normal * tileScale * vAmount;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(
        newPosition, 1.0 );
}
