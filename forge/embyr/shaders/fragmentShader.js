uniform sampler2D oceanTexture;
uniform sampler2D sandyTexture;
uniform sampler2D grassTexture;
uniform sampler2D forestTexture;
uniform sampler2D lavaTexture;
uniform sampler2D scrubTexture;
uniform sampler2D stoneTexture;
uniform sampler2D tileTexture;

vec2 tilePos;
float tile;
float tol = 0.05;

varying vec2 vUV;

varying float vAmount;
varying vec3 newPosition;
float nTiles = 80.0001;
vec4 water;

void main() {
    //Check 80 minus
    //tilePos = 80.0 - floor(newPosition.xy / 64.0);
    //vec2 tileUV = newPosition.xz/64.0 - floor(newPosition.xz / 64.00);
    tilePos = floor(newPosition.xz / 64.0);
    vec2 tileUV = newPosition.xz/64.0;
    tile = 255.0 * texture2D(tileTexture, tilePos/nTiles ).r;

    vec4 grass  = texture2D( grassTexture,  tileUV* 1.0 );
    vec4 forest = texture2D( forestTexture, tileUV* 1.0 );
    vec4 lava   = texture2D( lavaTexture,   tileUV* 1.0 );
    vec4 scrub  = texture2D( scrubTexture,  tileUV* 1.0 );
    vec4 stone  = texture2D( stoneTexture,  tileUV* 1.0 );
    vec4 sandy  = texture2D( sandyTexture,  tileUV* 1.0 );
    /*
    vec4 sandy = (
        smoothstep(0.24, 0.27, vAmount) -
        smoothstep(0.28, 0.31, vAmount)
        ) * texture2D( sandyTexture, tileUV* 10.0 );
    */

    lava = lava * float(tile==0.0);
    water = sandy * float(tile == 1.0);
    grass = grass * float(tile == 2.0 );
    scrub = scrub * float(tile == 3.0);
    forest = forest * float(tile==4.0);
    stone = stone * float(tile==5.0);
    gl_FragColor = (vec4(0.0, 0.0, 0.0, 1.0) + 
          water + grass +
          forest + scrub + stone + lava);

    float modx = tilePos.x - (2.0 * floor(tilePos.x/2.0));
    float mody = tilePos.y - (2.0 * floor(tilePos.y/2.0));
    gl_FragColor = 0.8*gl_FragColor + 0.2*gl_FragColor*(abs(modx-mody));
    //Check if we are on a border
    vec2 err = newPosition.xz/64.0 - tilePos;

    //float xcolor = newPosition.x /10000.0;
    //float ycolor = newPosition.z /10000.0;
    //gl_FragColor = vec4(xcolor, ycolor, 0.0, 1.0);

    //Debug shader
    /*
    if (err.x < tol) {
       gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
    }

    if (err.y < tol) {
       gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }

    if (err.x < tol && err.y < tol) {
       gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
    }
    */
}
