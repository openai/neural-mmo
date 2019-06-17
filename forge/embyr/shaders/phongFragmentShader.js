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

#define PHONG

uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <uv2_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

void main() {

   #include <clipping_planes_fragment>

   vec4 diffuseColor = vec4( diffuse, opacity );
   ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
   vec3 totalEmissiveRadiance = emissive;

   #include <logdepthbuf_fragment>
   #include <map_fragment>
   #include <color_fragment>
   #include <alphamap_fragment>
   #include <alphatest_fragment>
   #include <specularmap_fragment>
   #include <normal_fragment_begin>
   #include <normal_fragment_maps>
   #include <emissivemap_fragment>

   // accumulation
   #include <lights_phong_fragment>
   #include <lights_fragment_begin>
   #include <lights_fragment_maps>
   #include <lights_fragment_end>

   // modulation
   #include <aomap_fragment>

   vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;

   #include <envmap_fragment>

   gl_FragColor = vec4( outgoingLight, diffuseColor.a );

            tilePos = floor(newPosition.xz / 64.0);
            vec2 tileUV = newPosition.xz/64.0;
            tile = 255.0 * texture2D(tileTexture, tilePos/nTiles ).r;

            vec4 grass  = texture2D( grassTexture,  tileUV* 2.0 );
            vec4 forest = texture2D( forestTexture, tileUV* 2.0 );
            vec4 lava   = texture2D( lavaTexture,   tileUV* 2.0 );
            vec4 scrub  = texture2D( scrubTexture,  tileUV* 2.0 );
            vec4 stone  = texture2D( stoneTexture,  tileUV* 3.0 );
            vec4 sandy  = texture2D( sandyTexture,  tileUV* 2.0 );
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
            gl_FragColor = 0.9*gl_FragColor + 0.1*gl_FragColor*(abs(modx-mody));

            gl_FragColor = vec4(gl_FragColor.xyz * outgoingLight, diffuseColor.a);

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
 
   #include <tonemapping_fragment>
   #include <encodings_fragment>
   #include <fog_fragment>
   #include <premultiplied_alpha_fragment>
   #include <dithering_fragment>

}
